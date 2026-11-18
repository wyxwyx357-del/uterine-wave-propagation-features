"""Fixed-version cohort screening and local video review; no outcome inputs.

Only case_id is read from the inclusion CSV. Historic caches supply timestamps
and background correction, never legacy R2 frequencies or feature trajectories.
Patient data and rendered outputs must remain local.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
import tempfile
import shutil
import html
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from uterine_wave.geometry import layout
from uterine_wave.deformation import compute_spacetime
from uterine_wave.radial_events import radial_candidates_from_maps
from uterine_wave.quality import tracking_quality, radial_quality_maps, candidate_quality
from uterine_wave.io import jsonable


def save(path,data):
    path.write_text(json.dumps(jsonable(data),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def load(path):
    with np.load(path,allow_pickle=False) as data:return dict(data)


def prepare(args):
    with args.cohort.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.reader(stream);index=next(reader).index('case_id')
        included={row[index] for row in reader}
    with args.manifest.open(encoding='utf-8-sig',newline='') as stream:
        manifest=list(csv.DictReader(stream))
    with args.cache_manifest.open(encoding='utf-8-sig',newline='') as stream:
        cache={row['case_id']:row['cache_path'] for row in csv.DictReader(stream)}
    eligible=[];audit=[]
    with zipfile.ZipFile(args.annotations) as archive:
        names=archive.namelist()
        for row in manifest:
            case=row['case_id'];record=dict(case_id=case,status='eligible')
            if case not in included:record['status']='not_in_included_cohort'
            elif case not in cache or not Path(cache[case]).exists():record['status']='cache_unavailable'
            else:
                cached=load(Path(cache[case]));t=cached['time_s']
                duration=float(t[-1]-t[0]+np.median(np.diff(t)))
                record['duration_s']=duration
                entries=[n for n in names if '/'+case+'/' in n and n.endswith('__start__f0000.json')]
                source=args.legacy_root/'runtime_aliases'/row['video'].replace('\\','/').split('/runtime_aliases/',1)[1]
                if duration<60:record['status']='duration_under_60_s'
                elif len(entries)!=1:record['status']='first_frame_geometry_unavailable'
                elif not source.exists():record['status']='video_unavailable'
                else:
                    document=json.loads(archive.read(entries[0]).decode('utf-8-sig'))
                    labels={s['label'] for s in document['shapes']}
                    if not {'ant_wall','post_wall','cervix_point','fundus_point'}.issubset(labels):
                        record['status']='anatomical_labels_incomplete'
                    else:
                        eligible.append(dict(case_id=case,review_id='case_'+case.split('-')[0],
                            video=str(source),cache=cache[case],annotation_entry=entries[0],
                            duration_s=duration,frames=len(t)))
            audit.append(record)
    # Identity ordering only: no outcome, prior feature, activity or quality ranking.
    pool=sorted((r for r in eligible if r['case_id'].split('-')[0] not in args.exclude_review_ids),
                key=lambda r:int(r['case_id'].split('-')[0]))
    ids=np.unique(np.rint(np.linspace(0,len(pool)-1,min(args.limit,len(pool)))).astype(int))
    selected=[pool[i] for i in ids]
    algorithm_files=['tracking.py','_lk.py','geometry.py','deformation.py','amplitude.py','radial_events.py','quality.py']
    protocol=dict(code_commit='94b5fe2f5c596ce4f5c8e51f80a8652da53fd3c4',
        source_hashes={p:digest(ROOT/'uterine_wave'/p) for p in algorithm_files},
        cohort_csv_sha256=digest(args.cohort),cohort_identifier_count=len(included),
        cohort_read_columns=['case_id'],manifest_rows=len(manifest),eligible_count=len(eligible),
        selection='Equally spaced numeric case IDs, before new tracking; exclude prior wave-development cases',
        selected=selected,interval_s=2.,time_factor=1.,
        noise_floor_rate=json.loads(args.noise_reference.read_text())['all_texture_threshold'],
        noise_reference_sha256=digest(args.noise_reference),
        background_source='existing cached poses/pose_valid/epoch; new frozen A0 target tracks',
        event_frequency=None,export_selection='Review examples ranked after screening, not representative prevalence')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'protocol.json').exists():
        old=json.loads((args.output/'protocol.json').read_text(encoding='utf-8'))
        if old!=protocol:raise ValueError('Output protocol differs; use a separate output directory')
    save(args.output/'protocol.json',protocol);save(args.output/'cohort_audit.json',audit)
    print('PREPARED',len(included),'included',len(eligible),'eligible','selected',[r['review_id'] for r in selected],flush=True)
    return protocol


class ProgressFrames(list):
    def __getitem__(self,index):
        if isinstance(index,int) and index>getattr(self,'_last',0)+400:
            self._last=index;print('TRACK_FRAME',index,len(self),flush=True)
        return super().__getitem__(index)


def process(row,args,protocol,cv2):
    from uterine_wave.tracking import track_segments
    destination=args.output/row['review_id'];destination.mkdir(exist_ok=True)
    if (destination/'summary.json').exists():return json.loads((destination/'summary.json').read_text())
    started=time.monotonic();base=load(Path(row['cache']))
    with zipfile.ZipFile(args.annotations) as archive:
        raw=archive.read(row['annotation_entry']);document=json.loads(raw.decode('utf-8-sig'))
    names=('ant_wall','post_wall','cervix_point','fundus_point')
    shapes={name:np.asarray(next(s['points'] for s in document['shapes'] if s['label']==name),float)*.5 for name in names}
    grid=layout(shapes);frames=ProgressFrames()
    # Windows OpenCV paths use a local ASCII staging path, consistently.
    with tempfile.TemporaryDirectory(prefix='wave_cohort_decode_') as td:
        staged=Path(td)/'source.mp4';shutil.copyfile(row['video'],staged)
        cap=cv2.VideoCapture(str(staged));fps=cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(cv2.resize(frame,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY))
        cap.release()
    t=base['time_s']
    if len(frames)!=len(t) or not np.allclose(np.diff(t),1/fps,atol=1e-6,rtol=0):
        raise ValueError('Cached time axis and native video frames differ')
    print('TRACK',row['review_id'],len(t),'points',len(grid['A_points'])+len(grid['B_points']),flush=True)
    tracks=track_segments(frames,np.vstack((grid['A_points'],grid['B_points'])),t,base['poses'],
                          pose_epoch=base['epoch'],pose_valid=base['pose_valid'])
    maps=compute_spacetime(tracks,grid,interval_s=2.)
    report,arrays=radial_candidates_from_maps(maps,noise_floor=protocol['noise_floor_rate'])
    quality=tracking_quality(tracks,grid)
    quality_maps=radial_quality_maps(tracks,grid,maps)
    np.savez_compressed(destination/'tracks.npz',**tracks)
    np.savez_compressed(destination/'grid.npz',**grid)
    np.savez_compressed(destination/'maps.npz',**maps)
    np.savez_compressed(destination/'candidate_maps.npz',**arrays)
    save(destination/'candidates.json',report)
    save(destination/'tracking_quality.json',quality)
    save(destination/'candidate_quality.json',candidate_quality(quality_maps,report))
    np.savez_compressed(destination/'quality_maps.npz',**quality_maps)
    direction=[dict(e,wall=w['wall']) for w in report['walls'] for e in w['events'] if e['classification'] in ('C2F','F2C')]
    # Ranking selects review material only; it never changes detection or counts waves.
    def score(e):return e.get('association_weight',0.) if e['evidence_kind']=='paired_phase' else 0.
    ranked=sorted(direction,key=score,reverse=True)
    summary=dict(review_id=row['review_id'],case_id=row['case_id'],status='screened',frames=len(t),fps=fps,
        duration_s=len(t)/fps,columns_per_wall=len(grid['u']),actual_interval_s=float(np.median(maps['interval_dt_s'])),
        radial_pcc_on_valid_native_steps=quality['radial_points'],
        pair_valid_fraction=maps['valid_mask'].mean((0,2)).tolist(),pose_valid_fraction=float(base['pose_valid'].mean()),
        candidates_per_wall=[w['candidate_count'] for w in report['walls']],
        directional_structures_per_wall=[w['traveling_candidate_count'] for w in report['walls']],
        paired_directional_structures=sum(e['evidence_kind']=='paired_phase' for e in direction),
        review_rank_score=max([score(e) for e in direction] or [0.]),
        suggested_structures=[dict(event_id=e['event_id'],wall=e['wall'],classification=e['classification'],
             evidence_kind=e['evidence_kind'],arrival_time_s=e['arrival_time_s'],sections=e['sections'],
             spatial_span=e['spatial_span'],quality_flags=e['quality_flags']) for e in ranked[:3]],
        annotation_sha256=hashlib.sha256(raw).hexdigest(),video_sha256=digest(Path(row['video'])),
        cache_sha256=digest(Path(row['cache'])),official_wave_count=None,seconds_elapsed=time.monotonic()-started)
    save(destination/'summary.json',summary)
    print('DONE',row['review_id'],summary['directional_structures_per_wall'],'valid',summary['pair_valid_fraction'],
          'seconds',round(summary['seconds_elapsed'],1),flush=True)
    return summary


def render(row,summary,args,cv2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=args.output/row['review_id'];maps=load(folder/'candidate_maps.npz');grid=load(folder/'grid.npz')
    tracks=load(folder/'tracks.npz');report=json.loads((folder/'candidates.json').read_text(encoding='utf-8'))
    t=maps['interval_center_s'];s=maps['s'];a=maps['filtered_A'];finite=abs(a[np.isfinite(a)])
    scale=float(np.quantile(finite,.97)) if len(finite) else .001
    fig,axes=plt.subplots(2,1,figsize=(9.4,7.6),sharex=True,constrained_layout=True)
    for wall,ax in enumerate(axes):
        ax.pcolormesh(t,s,np.ma.masked_invalid(a[:,wall].T),cmap='RdBu_r',vmin=-scale,vmax=scale,shading='nearest')
        ax.set_facecolor('.65');ax.set_ylim(1,0);ax.set_ylabel('Cervix -> fundus s')
        ax.set_title('Anterior' if wall==0 else 'Posterior');ax.set_xlim(0,summary['duration_s'])
        for event in report['walls'][wall]['events']:
            if event['classification'] not in ('C2F','F2C'):continue
            color='lime' if event['classification']=='C2F' else 'magenta'
            single=event['evidence_kind']=='single_phase'
            ax.plot(event['arrival_time_s'],s[event['sections']],'^--' if single else 'o-',color=color,ms=3,lw=1)
            j=len(event['sections'])//2
            ax.text(event['arrival_time_s'][j],s[event['sections'][j]],event['event_id'].split('_')[-1],fontsize=7,
                    bbox=dict(facecolor='white',alpha=.7,edgecolor='none'),clip_on=True)
    axes[-1].set_xlabel('Acquisition time (s)');fig.canvas.draw()
    chart=np.asarray(fig.canvas.buffer_rgba())[:,:,:3][:,:,::-1].copy()
    xleft,xright=[ax.get_window_extent().x0 for ax in axes],[ax.get_window_extent().x1 for ax in axes]
    ybounds=[(chart.shape[0]-ax.get_window_extent().y1,chart.shape[0]-ax.get_window_extent().y0) for ax in axes]
    fig.savefig(folder/'spacetime.png',dpi=150)
    seed=grid['A_points'];h,w=chart.shape[:2]
    staging=tempfile.TemporaryDirectory(prefix='wave_cohort_render_')
    staged_source=Path(staging.name)/'source.mp4';shutil.copyfile(row['video'],staged_source)
    cap=cv2.VideoCapture(str(staged_source));fps=cap.get(cv2.CAP_PROP_FPS)
    ok,first=cap.read()
    if not ok:raise ValueError('Cannot decode source for rendering')
    # Crop to the anatomical ROI, excluding the surrounding scanner metadata.
    bounds=[max(0,int(2*seed[:,0].min())-70),min(first.shape[1],int(2*seed[:,0].max())+71),
            max(0,int(2*seed[:,1].min())-70),min(first.shape[0],int(2*seed[:,1].max())+71)]
    xmin,xmax,ymin,ymax=bounds
    rw,rh=xmax-xmin,ymax-ymin;factor=min(900/rw,650/rh);cw,ch=int(rw*factor),int(rh*factor)
    width,height=1920,960;output=args.output/(row['review_id']+'_review_1x.mp4')
    staged_output=Path(staging.name)/'review.mp4'
    writer=cv2.VideoWriter(str(staged_output),cv2.VideoWriter_fourcc(*'avc1'),fps,(width,height))
    if not writer.isOpened():raise RuntimeError('H264 writer unavailable')
    count=0;frame=first
    while True:
        canvas=np.full((height,width,3),24,np.uint8)
        cv2.putText(canvas,row['review_id']+' | native acquisition timeline | review candidates, not confirmed waves',
                    (25,38),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1,cv2.LINE_AA)
        cv2.putText(canvas,f'Time {count/fps:.2f} s / {summary["duration_s"]:.2f} s | playback 1x',
                    (25,75),cv2.FONT_HERSHEY_SIMPLEX,.7,(245,245,245),1,cv2.LINE_AA)
        crop=cv2.resize(frame[ymin:ymax,xmin:xmax],(cw,ch),interpolation=cv2.INTER_LINEAR)
        canvas[135:135+ch,25:25+cw]=crop
        canvas[120:120+h,965:965+w]=chart
        for wall in (0,1):
            x=965+int(xleft[wall]+count/fps/summary['duration_s']*(xright[wall]-xleft[wall]))
            lo,hi=ybounds[wall];cv2.line(canvas,(x,120+int(lo)),(x,120+int(hi)),(0,0,0),2)
        cv2.putText(canvas,'Left: original ultrasound ROI, no artificial wave animation.',(25,850),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1,cv2.LINE_AA)
        cv2.putText(canvas,'Right: green C2F / magenta F2C; dashed triangles = single phase; gray = missing.',(25,885),cv2.FONT_HERSHEY_SIMPLEX,.58,(240,240,240),1,cv2.LINE_AA)
        cv2.putText(canvas,'Please verify activity identity, direction and tracking artifacts. Candidate number is NOT wave count.',(25,920),cv2.FONT_HERSHEY_SIMPLEX,.58,(240,240,240),1,cv2.LINE_AA)
        writer.write(canvas)
        if count==min(100,len(tracks['time_s'])-1):
            from PIL import Image
            Image.fromarray(canvas[:,:,::-1]).save(folder/'review_preview.jpg')
        count+=1
        if count%500==0:print('RENDER_FRAME',row['review_id'],count,summary['frames'],flush=True)
        ok,frame=cap.read()
        if not ok:break
    cap.release();writer.release();plt.close(fig)
    check=cv2.VideoCapture(str(staged_output));ok,_=check.read();encoded=int(check.get(cv2.CAP_PROP_FRAME_COUNT));rate=check.get(cv2.CAP_PROP_FPS);check.release()
    if not ok or encoded!=summary['frames'] or abs(rate-fps)>.001:raise ValueError('Encoded frame count or timeline differs')
    shutil.copyfile(staged_output,output);staging.cleanup()
    print('EXPORTED',row['review_id'],encoded,flush=True)
    return dict(review_id=row['review_id'],video=str(output),frames=encoded,fps=rate,roi_native_xyxy=bounds)


def write_index(output):
    protocol=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    summaries=json.loads((output/'screening_summary.json').read_text(encoding='utf-8'))
    exports=json.loads((output/'exports.json').read_text(encoding='utf-8'))
    lookup={row['review_id']:row for row in summaries}
    cards=[]
    for exported in exports:
        rid=exported['review_id'];row=lookup[rid]
        supports=' / '.join(f'{100*x:.1f}%' for x in row['pair_valid_fraction'])
        suggestions=[]
        for event in row['suggested_structures']:
            arrival=event['arrival_time_s']
            wall='前壁' if event['wall']==0 else '后壁'
            suggestions.append(f"<li>{html.escape(event['event_id'])}：{wall}，{event['classification']}，"
                               f"到达约{min(arrival):.1f}–{max(arrival):.1f}秒；覆盖路径{event['spatial_span']*100:.1f}%</li>")
        cards.append(f'''<section><h2>{rid.replace('case_', '病例 ')} · {row['duration_s']:.1f}秒</h2>
<p>前/后壁有效点对样本比例：{supports}。下列方向均待人工复核。</p>
<video controls preload="metadata" poster="{rid}/review_preview.jpg" src="{rid}_review_1x.mp4"></video>
<ul>{''.join(suggestions)}</ul><p><a href="{rid}_review_1x.mp4">单独打开视频</a> ·
<a href="{rid}/spacetime.png">打开径向图</a> · <a href="{rid}/candidates.json">完整候选记录</a></p></section>''')
    table=[]
    for row in summaries:
        if row['status']=='screened':
            support=' / '.join(f'{x*100:.1f}%' for x in row['pair_valid_fraction'])
            table.append(f"<tr><td>{row['review_id']}</td><td>{row['duration_s']:.1f}</td><td>{support}</td>"
                         f"<td>{row['directional_structures_per_wall']}</td><td>已筛查，尚未人工确认波</td></tr>")
        else:table.append(f"<tr><td>{row['review_id']}</td><td colspan=4>{html.escape(row.get('reason','处理失败'))}</td></tr>")
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>多病例蠕动候选人工复核</title><style>body{font:17px/1.65 system-ui,sans-serif;max-width:1350px;margin:30px auto;padding:0 20px;background:#f3f5f7;color:#172331}section{background:white;padding:22px;margin:26px 0;border-radius:10px}video{width:100%;background:#111}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ccd4dd;padding:8px}a{color:#0757ab}.notice{padding:16px;background:#fff0cf}</style>
<h1>多病例蠕动候选人工复核</h1>
<p class="notice">这些是代码提示的候选，不是已确认的波，也不是最终波次数。请先看左侧原始视频，再对照右侧径向图；灰色为测量缺失。C2F＝宫颈→宫底，F2C＝宫底→宫颈；虚线三角为单相结构，可能与其他候选属于同一次活动。</p>
'''
    page+=f"<p>在历史{protocol['manifest_rows']}例批次与最终入组名单的交集中，找到{protocol['eligible_count']}例具备本次输入条件、时长至少60秒的检查；按编号预先等距抽取{len(protocol['selected'])}例，统一算法、2秒测量跨度及同一噪声参照。导出{len(exports)}例候选较清楚的视频供复核，不能用这批展示样本估计总体检出率。全程未按妊娠结局筛选或调参。</p>"
    page+='<p><a href="review_sheet.csv">下载/打开人工复核表</a> · <a href="screening_summary.json">完整筛查记录</a></p>'
    page+=''.join(cards)+'<h2>全部预选病例的筛查记录</h2><p>“方向结构数”包括可能重复描述同一事件的阶段，不是波次数。未检出时也不直接解释为无蠕动。</p><table><tr><th>病例</th><th>时长/秒</th><th>有效支持 前/后</th><th>方向结构数 前/后</th><th>状态</th></tr>'+''.join(table)+'</table></html>'
    (output/'index.html').write_text(page,encoding='utf-8')
    lines=['# 明日人工复核包','', '打开 index.html，可直接播放原始画面与径向图同步视频。','',
           '请填写 review_sheet.csv：可见活动（是/否/不确定）、是否同一次活动、实际方向、是否可能是抖动/追踪伪影。',
           '表中wall：0=前壁，1=后壁；C2F=宫颈到宫底，F2C=宫底到宫颈。',
           'same_event_group：认为属于同一次传播的不同壁/不同阶段，填写相同编号；不要直接按候选编号计次。',
           '候选方向和编号是提示，不是确认结果；请记录视频里看到但代码没有标出的活动。',
           '预选病例均来自历史入组记录，时长至少60秒；固定参数，先等距抽样再计算，没有按新结果或妊娠结局挑选测试病例。',
           '导出视频是筛查后挑出的展示样本，不能代表整个人群活动比例。全部预选病例保留在screening_summary.json。','']
    for row in exports:lines.append(f"- {row['review_id']}: {Path(row['video']).name}")
    (output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cohort','manifest','cache-manifest','annotations','legacy-root','noise-reference','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--limit',type=int,default=12);parser.add_argument('--exports',type=int,default=6)
    parser.add_argument('--exclude-review-ids',nargs='*',default=[],help='Previously used case-number prefixes to reserve outside this batch')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--opencv-bin',type=Path)
    args=parser.parse_args()
    if args.limit<1 or args.exports<0:parser.error('--limit must be positive and --exports nonnegative')
    protocol=prepare(args)
    if args.prepare_only:return
    dll=None
    if args.opencv_bin:
        import os
        dll=os.add_dll_directory(str(args.opencv_bin))
    import cv2
    if cv2.cuda.getCudaEnabledDeviceCount()<1:raise RuntimeError('Frozen A0 requires CUDA OpenCV')
    summaries=[]
    for row in protocol['selected']:
        try:summary=process(row,args,protocol,cv2)
        except (ValueError,OSError,cv2.error) as error:
            summary=dict(review_id=row['review_id'],case_id=row['case_id'],status='processing_failed',reason=str(error))
            save(args.output/row['review_id']/'failure.json',summary);print('FAILED',row['review_id'],str(error),flush=True)
        summaries.append(summary);save(args.output/'screening_summary.json',summaries)
    successful=[r for r in summaries if r['status']=='screened']
    ranked=sorted(successful,key=lambda r:r['review_rank_score'],reverse=True)
    chosen=ranked[:args.exports];lookup={r['review_id']:r for r in protocol['selected']}
    exported=[render(lookup[r['review_id']],r,args,cv2) for r in chosen]
    save(args.output/'exports.json',exported)
    with (args.output/'review_sheet.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['review_id','candidate_id','wall','suggested_direction','evidence_kind','suggested_arrival_s',
                'visible_activity_yes_no_uncertain','same_event_group','actual_direction','tracking_artifact','notes']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for summary in chosen:
            for event in summary['suggested_structures']:
                writer.writerow(dict(review_id=summary['review_id'],candidate_id=event['event_id'],wall=event['wall'],
                    suggested_direction=event['classification'],evidence_kind=event['evidence_kind'],
                    suggested_arrival_s=';'.join(f'{t:.2f}' for t in event['arrival_time_s'])))
    write_index(args.output)
    print('COMPLETE',len(successful),'screened',len(exported),'review videos',flush=True)


if __name__=='__main__':main()
