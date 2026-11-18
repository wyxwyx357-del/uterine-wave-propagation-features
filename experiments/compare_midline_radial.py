"""Review existing radial maps against tracked middle/adjacent band bending.

No new tracking, segmentation, threshold search, event detection or patient
feature aggregation. Curvature is the existing B definition at the same native
frame interval as A. Patient videos and review files stay local.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from uterine_wave.deformation import rate_B,interval_support
from uterine_wave.amplitude import band_limited_signal
from uterine_wave.io import jsonable


def load(path):
    with np.load(path,allow_pickle=False) as z:return dict(z)


def save(path,value):
    path.write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def band_maps(tracks,grid,radial):
    """Existing B rate, continuous support and physiological-band display."""
    t=np.asarray(tracks['time_s']);hop=int(radial['hop']);offset=len(grid['A_points'])
    centers=(t[hop:]+t[:-hop])/2
    if not np.allclose(centers,radial['interval_center_s'],atol=1e-8,rtol=0):
        raise ValueError('Radial and band measurement centers differ')
    points=np.asarray(tracks['corrected']);ids=np.asarray(grid['B_curve_ids'])
    if points.shape!=(len(t),offset+len(grid['B_points']),2):
        raise ValueError('Expected original [A,B] tracking order')
    b,_=rate_B(points[:,offset:],grid,t,hop)
    support=interval_support(tracks,hop)&np.isfinite(points[hop:]).all(2)&np.isfinite(points[:-hop]).all(2)
    valid=np.stack([np.stack([support[:,offset+curve[j-2:j+3]].all(1)
                             for j in range(2,len(curve)-2)],axis=1)for curve in ids],axis=1)
    valid&=np.isfinite(b);b=np.where(valid,b,np.nan)
    filtered=np.full_like(b,np.nan);minimum=max(3,int(np.ceil(12/np.median(np.diff(centers)))))
    for band,col in np.ndindex(b.shape[1:]):
        runs=np.flatnonzero(np.diff(np.r_[False,valid[:,band,col],False])).reshape(-1,2)
        for begin,end in runs:
            if end-begin>=minimum:
                filtered[begin:end,band,col]=band_limited_signal(b[begin:end,band,col],centers[begin:end])
    return dict(B_rate=b,B_band_rate=filtered,valid_mask=valid,
                band_valid_mask=valid&np.isfinite(filtered),interval_center_s=centers,
                interval_dt_s=t[hop:]-t[:-hop],s=np.asarray(radial['s'])[2:-2],
                fractions=np.array([.25,.5,.75]),event_count=None,
                units='(analysis_pixel * second)^-1')


def figures(radial,previous,bands,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def draw(name,panels):
        fig,axes=plt.subplots(len(panels),1,figsize=(9.5,2.25*len(panels)),sharex=True,constrained_layout=True)
        for ax,(values,space,title,unit,limit) in zip(np.atleast_1d(axes),panels):
            image=ax.pcolormesh(radial['interval_center_s'],space,np.ma.masked_invalid(values.T),
                shading='nearest',cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax.set_facecolor('.65');ax.set_ylim(1,0);ax.set_ylabel('Cervix -> fundus s');ax.set_title(title,fontsize=10)
            ax.set_xlim(float(radial['interval_start_s'][0]),float(radial['interval_end_s'][-1]))
            fig.colorbar(image,ax=ax,label=unit,fraction=.025)
        axes[-1].set_xlabel('Acquisition time (s)')
        fig.canvas.draw();height=int(fig.canvas.get_width_height()[1])
        metadata=dict(width=fig.canvas.get_width_height()[0],height=height,
            xlim=list(axes[0].get_xlim()),boxes=[dict(x0=ax.get_window_extent().x0,x1=ax.get_window_extent().x1,
            y0=height-ax.get_window_extent().y1,y1=height-ax.get_window_extent().y0)for ax in axes])
        fig.savefig(output/name,dpi=fig.dpi);plt.close(fig)
        return metadata
    def scale(values):
        finite=abs(values[np.isfinite(values)])
        return max(float(np.quantile(finite,.97)),1e-12)if len(finite)else 1.
    metadata={}
    for mode in ('band','raw'):
        a=previous['filtered_A']if mode=='band'else np.where(radial['valid_mask'],radial['A'],np.nan)
        b=bands['B_band_rate']if mode=='band'else bands['B_rate']
        alimit=scale(a);blimit=scale(b)
        metadata[mode]=draw(f'comparison_{mode}.png',[
            (a[:,0],radial['s'],'Anterior radial stretch rate A','1/s',alimit),
            (a[:,1],radial['s'],'Posterior radial stretch rate A','1/s',alimit),
            (b[:,1],bands['s'],'Tracked middle band: curvature change rate B','1/(px*s)',blimit)])
        metadata['near_'+mode]=draw(f'adjacent_{mode}.png',[
            (b[:,0],bands['s'],'Band nearer anterior side (initial fraction 0.25)','1/(px*s)',blimit),
            (b[:,2],bands['s'],'Band nearer posterior side (initial fraction 0.75)','1/(px*s)',blimit)])
    return metadata


STYLE="""body{font:16px/1.6 system-ui,sans-serif;max-width:1800px;margin:22px auto;padding:0 18px;color:#172331;background:#f4f6f8}section{background:white;padding:20px;margin:25px 0;border-radius:10px}.row{display:grid;grid-template-columns:48% 52%;gap:10px}.viewport{position:relative;width:100%;overflow:hidden;background:#111}.stage{position:absolute}.stage video{display:block;width:100%}.track{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.plot{position:relative}.plot img{display:block;width:100%}.cursor{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.controls{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0;align-items:center}.seek{width:100%}button{padding:6px 12px}a{color:#0757ab}.notice{padding:14px;background:#fff1ce}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}@media(max-width:1000px){.row{grid-template-columns:1fr}}"""

JAVASCRIPT=r"""
document.querySelectorAll('section[data-case]').forEach(card=>{
 const id=card.dataset.case,d=window.shapeReviewData[id],v=card.querySelector('video'),stage=card.querySelector('.stage'),view=card.querySelector('.viewport'),overlay=card.querySelector('.track');
 const toggle=card.querySelector('.points'),radials=card.querySelector('.radials'),band=card.querySelector('.bandpass'),seek=card.querySelector('.seek'),clock=card.querySelector('.clock');let roi=false;
 const plots=[card.querySelector('.mainplot'),card.querySelector('.nearplot')];
 function layout(){if(!v.videoWidth)return;const w=view.clientWidth,b=roi?d.roi:[0,v.videoWidth,0,v.videoHeight],scale=w/(b[1]-b[0]);view.style.height=(b[3]-b[2])*scale+'px';stage.style.width=v.videoWidth*scale+'px';stage.style.left=-b[0]*scale+'px';stage.style.top=-b[2]*scale+'px';overlay.width=v.videoWidth;overlay.height=v.videoHeight;draw();}
 function draw(){const i=Math.max(0,Math.min(d.frames.length-1,Math.round(v.currentTime*d.fps))),ctx=overlay.getContext('2d');ctx.clearRect(0,0,overlay.width,overlay.height);
  if(toggle.checked){d.band_ids.forEach((ids,k)=>{ctx.strokeStyle=['#28c5ff','#ffe330','#ee66ff'][k];ctx.lineWidth=2;let previous=null;ids.forEach(j=>{const p=d.frames[i][j];if(!p){previous=null;return;}if(previous){ctx.beginPath();ctx.moveTo(...previous);ctx.lineTo(...p);ctx.stroke();}ctx.beginPath();ctx.arc(...p,2.5,0,2*Math.PI);ctx.fillStyle=ctx.strokeStyle;ctx.fill();previous=p;});});}
  if(radials.checked){d.pair_ids.forEach(pair=>{const a=d.frames[i][pair[0]],b=d.frames[i][pair[1]];if(a&&b){ctx.strokeStyle='rgba(255,255,255,.7)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}[[a,'#44ee66'],[b,'#ff9b24']].forEach(([p,color])=>{if(p){ctx.beginPath();ctx.arc(...p,2.3,0,2*Math.PI);ctx.fillStyle=color;ctx.fill();}});});}
  const mode=band.checked?'band':'raw';plots.forEach((plot,j)=>{const meta=d.plots[(j?'near_':'')+mode],canvas=plot.querySelector('canvas');canvas.width=meta.width;canvas.height=meta.height;const c=canvas.getContext('2d');c.strokeStyle='black';c.lineWidth=2;meta.boxes.forEach(box=>{const x=box.x0+(v.currentTime-meta.xlim[0])/(meta.xlim[1]-meta.xlim[0])*(box.x1-box.x0);if(x>=box.x0&&x<=box.x1){c.beginPath();c.moveTo(x,box.y0);c.lineTo(x,box.y1);c.stroke();}});});
  clock.textContent=v.currentTime.toFixed(2)+' / '+d.duration.toFixed(2)+' 秒';seek.value=v.currentTime;}
 card.querySelector('.play').onclick=()=>{document.querySelectorAll('video').forEach(other=>{if(other!==v)other.pause();});if(v.paused)v.play();else v.pause();};
 card.querySelector('.zoom').onclick=()=>{roi=!roi;layout();};seek.max=d.duration;seek.oninput=()=>{v.currentTime=Number(seek.value);};toggle.onchange=draw;radials.onchange=draw;
 band.onchange=()=>{plots.forEach((p,j)=>p.querySelector('img').src=id+'/'+(j?'adjacent_':'comparison_')+(band.checked?'band':'raw')+'.png');draw();};
 function ready(){card.dataset.videoReady='true';layout();}v.addEventListener('loadedmetadata',ready);if(v.readyState>=1)ready();v.addEventListener('timeupdate',draw);v.addEventListener('seeked',draw);new ResizeObserver(layout).observe(view);
 if(v.requestVideoFrameCallback){const tick=()=>{draw();v.requestVideoFrameCallback(tick);};v.requestVideoFrameCallback(tick);}
 card.querySelectorAll('img').forEach(image=>image.addEventListener('load',draw));
});
"""


def build(args):
    import os
    dll=os.add_dll_directory(str(args.opencv_bin))if args.opencv_bin else None
    import cv2
    args.output.mkdir(parents=True,exist_ok=True)
    exports=json.loads((args.source/'exports.json').read_text(encoding='utf-8'))
    protocol=json.loads((args.source/'protocol.json').read_text(encoding='utf-8'))
    previous={r['review_id']:r for r in protocol['selected']};cards=[];summary=[]
    for row in exports:
        rid=row['review_id'];source=args.source/rid;dest=args.output/rid;dest.mkdir(exist_ok=True)
        tracks=load(source/'tracks.npz');grid=load(source/'grid.npz');radial=load(source/'maps.npz');old=load(source/'candidate_maps.npz')
        bands=band_maps(tracks,grid,radial)
        np.testing.assert_array_equal(bands['interval_center_s'],old['interval_center_s'])
        # Existing A stays byte-for-byte numerically identical; no re-estimation.
        np.savez_compressed(dest/'comparison_maps.npz',radial_A=radial['A'],radial_band_A=old['filtered_A'],
            radial_valid=radial['valid_mask'],radial_s=radial['s'],**{k:v for k,v in bands.items()if k not in ('event_count','units')})
        plots=figures(radial,old,bands,dest)
        original=Path(previous[rid]['video'])
        if not original.exists():original=ROOT.parent/original
        video=dest/'source_1x.mp4';shutil.copyfile(original,video)
        if digest(video)!=digest(original):raise ValueError('Review video copy differs from source')
        with tempfile.TemporaryDirectory(prefix='shape_review_decode_')as td:
            staged=Path(td)/'source.mp4';shutil.copyfile(video,staged);cap=cv2.VideoCapture(str(staged))
            fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));ok,first=cap.read();cap.release()
        if not ok or count!=len(tracks['time_s'])or abs(fps-row['fps'])>.001:
            raise ValueError('Original video and saved tracks differ')
        from PIL import Image
        Image.fromarray(first[:,:,::-1]).save(dest/'first_frame.jpg')
        positions=tracks['positions']*2
        positions=np.where(tracks['target_valid'][...,None],positions,np.nan)
        frames=[]
        for frame in positions:
            frames.append([[round(float(p[0]),1),round(float(p[1]),1)]if np.isfinite(p).all()else None for p in frame])
        data=dict(fps=fps,duration=count/fps,frames=frames,band_ids=(grid['B_curve_ids']+len(grid['A_points'])).tolist(),
                  pair_ids=grid['A_pair_ids'].reshape(-1,2).tolist(),roi=row['roi_native_xyxy'],plots=plots)
        (dest/'display_data.js').write_text('window.shapeReviewData=window.shapeReviewData||{};window.shapeReviewData['+json.dumps(rid)+']='+json.dumps(data,separators=(',',':'),allow_nan=False)+';',encoding='utf-8')
        report=dict(review_id=rid,duration_s=count/fps,frames=count,fps=fps,
            actual_interval_s=float(np.median(bands['interval_dt_s'])),
            band_valid_fraction=bands['valid_mask'].mean((0,2)).tolist(),
            source_tracks_sha256=digest(source/'tracks.npz'),source_grid_sha256=digest(source/'grid.npz'),
            source_radial_maps_sha256=digest(source/'maps.npz'),source_video_sha256=digest(video),
            source_definition='A0-tracked initial fractions 0.25/0.5/0.75; not framewise segmented boundaries',
            event_count=None)
        save(dest/'summary.json',report);summary.append(report)
        cards.append(f'''<section data-case="{html.escape(rid)}"><h2>{rid.replace('case_','病例 ')} · {count/fps:.1f}秒</h2>
<div class="controls"><button class="play">播放/暂停</button><button class="zoom">全景 / 局部放大</button>
<label><input type="checkbox" class="points">显示中线和相邻窄带</label><label><input type="checkbox" class="radials">显示径向点对</label>
<label><input type="checkbox" class="bandpass" checked>0.5–5次/分频带显示</label><span class="clock">0秒</span></div>
<input class="seek" type="range" min="0" step="0.01" value="0">
<div class="row"><div><div class="viewport"><div class="stage"><video controls preload="metadata" poster="{rid}/first_frame.jpg" src="{rid}/source_1x.mp4"></video><canvas class="track"></canvas></div></div>
<p>完整原始画面，可切换局部放大。测量位置开关：黄线＝初始中线带，蓝/紫＝两侧窄带；径向点对为绿点inner、橙点outer。缺失不连接。</p></div>
<div class="plot mainplot"><img src="{rid}/comparison_band.png"><canvas class="cursor"></canvas></div></div>
<details><summary>展开两侧窄带对照</summary><div class="plot nearplot"><img src="{rid}/adjacent_band.png"><canvas class="cursor"></canvas></div></details>
<p><a href="{rid}/comparison_band.png">单独打开对照图</a> · <a href="{rid}/source_1x.mp4">单独播放原视频</a> · <a href="{rid}/summary.json">测量说明</a></p></section>''')
        print('READY',rid,'band support',report['band_valid_fraction'],flush=True)
    scripts=''.join(f'<script src="{r["review_id"]}/display_data.js"></script>'for r in summary)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>径向与中线同步对照</title><style>{STYLE}</style>
<h1>原始视频 · 径向伸缩 · 中线形状变化</h1><p class="notice">这次只做信号对照，不新增波检测或次数。右侧上两行是原有前/后壁径向图，第三行是中线带弯曲变化；红蓝表示各自量的正负，不能直接当作传播方向或比较两种幅度大小。请看同一活动是否在相邻位置依次出现。</p>
<p>同一批6例、同一A0轨迹、同一约2秒跨度；没有重新分割或调参。中线带是初始中间位置的追踪曲线，尚未证明等于每一帧的真实内膜中线。可切换原始/频带显示，观察滤波影响。</p>
<p><a href="review_sheet.csv">人工对照表</a>：填写原视频活动时间、哪张图更贴近、方向是否清楚、可能的伪影。看不清请填“不确定”。</p>
{''.join(cards)}{scripts}<script src="review.js"></script></html>'''
    (args.output/'index.html').write_text(page,encoding='utf-8');(args.output/'review.js').write_text(JAVASCRIPT,encoding='utf-8')
    import csv
    with (args.output/'review_sheet.csv').open('w',encoding='utf-8-sig',newline='')as stream:
        writer=csv.writer(stream);writer.writerow(['review_id','activity_time_s','visible_activity','radial_matches','middle_band_matches','adjacent_band_matches','direction','same_activity_group','artifact_or_uncertain','notes'])
        for row in summary:writer.writerow([row['review_id']]+['']*9)
    save(args.output/'summary.json',summary)
    (args.output/'README.md').write_text('打开index.html。先看原视频，再对照径向和中线图；形状图不是新波次数或妊娠特征。测量位置可开关，全景可切换局部放大。\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='Previous cohort review folder')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--opencv-bin',type=Path)
    build(parser.parse_args())


if __name__=='__main__':main()
