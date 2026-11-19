"""Pair recorded clinical cpm with fixed .6 proposals, without fabricating rates.

Reads only the prior whitelisted clinical export, fixed salience decisions and
existing maps/video index. Patient outputs remain local. No threshold tuning.
"""
import argparse
import csv
from html import escape
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def comparison_row(reference, salience):
    matches=reference['matches'];record=matches[0] if len(matches)==1 else {}
    p,n=record.get('positive_value'),record.get('negative_value')
    numeric=p is not None and n is not None
    walls=salience['walls']
    counts=[w['metrics']['after_proposals'] for w in walls]
    directional=[w['metrics']['after_directional'] for w in walls]
    return dict(case=reference['review_id'],match_status=reference['match_status'],
        excel_row=record.get('excel_row'),clinical_check_id=record.get('clinical_check_id'),
        video_datetime=record.get('video_datetime'),clinical_datetime=record.get('clinical_datetime'),
        clinical_positive_raw=record.get('positive_raw'),clinical_negative_raw=record.get('negative_raw'),
        clinical_positive_cpm=p,clinical_negative_cpm=n,
        clinical_column_sum_cpm=p+n if numeric else None,
        clinical_has_recorded_activity=(p+n>0) if numeric else None,
        actual_video_duration_s=reference.get('actual_video_duration_s'),
        anterior_salient_proposals=counts[0],posterior_salient_proposals=counts[1],
        both_wall_proposal_rows=sum(counts),
        anterior_directional_proposals=directional[0],posterior_directional_proposals=directional[1],
        both_wall_directional_rows=sum(directional),
        has_salient_proposal=any(counts),has_directional_salient_proposal=any(directional),
        observation_period_correspondence='user_unconfirmed',positive_anatomical_direction='unconfirmed',
        candidate_counts_unit='rows per saved video, not cpm or unique physiological events',
        algorithm_event_count=None,algorithm_event_rate_cpm=None,clinical_accuracy=None)


def run(args):
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    refs_path=args.reference/'review_case_matches.json'
    refs=read(refs_path);source_summary=read(args.reference/'summary.json')
    if source_summary['clinical_record_unit']!='per_minute':raise ValueError('Reference unit must be confirmed cpm')
    aliases={r['review_id']:r['case_alias'] for r in read(args.video_index)['sources']}
    sources={str(refs_path):digest(refs_path),str(args.video_index):digest(args.video_index)}
    rows=[];details=[]
    def relative(path):return os.path.relpath(path.resolve(),args.output).replace(os.sep,'/')
    for ref in refs:
        name=ref['review_id'];path=args.salience/(name+'.json');original_path=args.ownership/(name+'.json')
        sources[str(path)]=digest(path);sources[str(original_path)]=digest(original_path)
        selected=read(path);original=read(original_path)
        if selected['source']!=name or original['case']!=name:raise ValueError('Case identities differ')
        row=comparison_row(ref,selected)
        maps_path=args.maps/name/'maps.npz';sources[str(maps_path)]=digest(maps_path)
        with np.load(maps_path,allow_pickle=False) as z:
            row['pair_valid_fraction_by_wall']=z['valid_mask'].mean((0,2)).tolist()
            row['actual_measurement_span_s']=float(np.median(z['interval_dt_s']))
        video=args.video_index.parent/aliases[name]/'source_1x.mp4'
        if not video.exists():raise ValueError('Existing original video missing: '+name)
        row['original_video_relative_path']=relative(video)
        row['maps_relative_path']=relative(args.salience/(name+'.png'))
        clips=[]
        for w in selected['walls']:
            wall=w['wall'];ownership=original['walls'][wall]['reconciliation']
            proposals={e['event_proposal_id']:e for e in ownership['event_proposals']}
            for score in w['scores']:
                if score['decision']!='salient_support':continue
                event=proposals[score['event_proposal_id']]
                entity=ownership['source_evidence'][event['canonical_source_entity_id']]
                phases=[p for c in score['retained_sections'] for p in entity['profile'][str(c)]]
                start=max(0.,min(p['start_s'] for p in phases));end=max(p['end_s'] for p in phases)
                clips.append(dict(wall=wall,event_proposal_id=event['event_proposal_id'],
                    retained_sections=score['retained_sections'],original_direction=event['direction'],
                    evidence_start_s=start,evidence_end_s=end,
                    ownership_status=event['ownership_status'],event_count=None))
        row['candidate_replay_intervals']=clips;rows.append(row)
        def fmt(value):return '未记录' if value is None or value=='' else escape(str(value))
        header=(f"<h2>{escape(name)}：临床 {fmt(row['clinical_positive_raw'])} / {fmt(row['clinical_negative_raw'])} 次/分；"
                f"全段显著候选 {row['anterior_salient_proposals']} / {row['posterior_salient_proposals']} 条</h2>")
        note=(f"<p>表格第 {fmt(row['excel_row'])} 行；视频 {row['actual_video_duration_s']:.2f} 秒。临床记录时间 {fmt(row['clinical_datetime'])}；"
              f"视频首帧 {fmt(row['video_datetime'])}。时间戳不能证明临床计数针对该片段。</p>")
        panel=(header+note+'<video controls preload="none" src="'+escape(relative(video))+'"></video>'+
               '<p><a href="'+escape(row['maps_relative_path'])+'">筛选前后时空图</a></p>')
        for clip in clips:
            link=relative(video)+f"#t={clip['evidence_start_s']:.3f},{clip['evidence_end_s']:.3f}"
            panel+=('<details><summary>'+('前壁' if clip['wall']==0 else '后壁')+' '+escape(clip['event_proposal_id'])+
                f" · {clip['evidence_start_s']:.1f}–{clip['evidence_end_s']:.1f}s · 原候选方向 "+escape(str(clip['original_direction']))+
                '</summary><p>这是候选证据范围，不是已确认波的持续时间；两壁可能表示同一活动。</p><video controls preload="none" src="'+escape(link)+'"></video></details>')
        details.append('<section id="'+escape(name)+'">'+panel+'</section>')
    def table(flag):
        return [dict(clinical_recorded_activity=clinical,algorithm_proposal_present=algorithm,
                     cases=sum(r['clinical_has_recorded_activity']==clinical and r[flag]==algorithm for r in rows))
                for clinical in (False,True) for algorithm in (False,True)]
    summary=dict(matched_cases=len(rows),same_saved_video_observation_confirmed=False,
        clinical_record_unit='per_minute',positive_direction_confirmed=False,
        clinical_both_zero=sum(r['clinical_column_sum_cpm']==0 for r in rows),
        clinical_nonzero=sum(r['clinical_has_recorded_activity'] is True for r in rows),
        descriptive_activity_presence_table=table('has_salient_proposal'),
        descriptive_directional_presence_table=table('has_directional_salient_proposal'),
        scope='same-examination exploratory comparison, not a video-level gold standard',
        algorithm_event_count=None,algorithm_event_rate_cpm=None,ICC=None,clinical_accuracy=None)
    save(args.output/'cases.json',rows);save(args.output/'summary.json',summary)
    columns=['case','excel_row','video_datetime','clinical_datetime','clinical_positive_cpm','clinical_negative_cpm',
        'clinical_column_sum_cpm','actual_video_duration_s','anterior_salient_proposals','posterior_salient_proposals',
        'anterior_directional_proposals','posterior_directional_proposals','algorithm_event_rate_cpm','observation_period_correspondence']
    with (args.output/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    cells=[]
    for row in rows:
        cells.append('<tr><td><a href="#'+escape(row['case'])+'">'+escape(row['case'])+'</a></td><td>'+str(row['excel_row'])+'</td><td>'+
            fmt(row['clinical_positive_cpm'])+'</td><td>'+fmt(row['clinical_negative_cpm'])+'</td><td>'+fmt(row['clinical_column_sum_cpm'])+
            f"</td><td>{row['actual_video_duration_s']:.1f}</td><td>{row['anterior_salient_proposals']} / {row['posterior_salient_proposals']}"+
            f"</td><td>{row['anterior_directional_proposals']} / {row['posterior_directional_proposals']}</td></tr>")
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>临床记录与显著候选对照</title><style>body{font:17px/1.65 system-ui;max-width:1250px;margin:25px auto;padding:15px;color:#203040}td,th{padding:.5rem;border-bottom:1px solid #ccc}section{border-top:2px solid #ccc;margin-top:3rem}video{width:100%;max-height:600px;background:#111}details{padding:.5rem;background:#f4f6f9;margin:.5rem 0}.note{background:#fff2d9;padding:1rem}a{color:#135f9b}</style>
<h1>临床记录与0.6显著活动候选对照</h1><p class="note">临床表为次/分；程序列是整段视频内的候选条数，尚不是波频率。用户不确定临床计数是否针对保存视频，正反解剖含义也未确认。本页只作同次检查参考，不算准确率或ICC，不用临床数字调阈值。</p>
<p>先看数字，需要时点病例展开原视频和显著候选证据时段；无需逐条补医学标签。两壁候选可能重复表示同一活动，不能直接相加计波。正反两列合计也不自动等于去重后的独立生理事件频率。</p>
<p><a href="comparison.csv">下载对照CSV</a> · <a href="cases.json">完整记录</a> · <a href="summary.json">摘要</a></p>
<table><tr><th>病例</th><th>Excel行</th><th>临床正向<br>次/分</th><th>临床反向<br>次/分</th><th>两列合计<br>次/分</th><th>视频秒</th><th>显著候选<br>前壁/后壁条数</th><th>其中方向候选<br>前壁/后壁条数</th></tr>'''+''.join(cells)+'</table>'+''.join(details)+'</html>'
    (args.output/'index.html').write_text(page,encoding='utf-8')
    if any(digest(Path(p))!=sha for p,sha in sources.items()):raise RuntimeError('Inputs changed during comparison')
    save(args.output/'protocol.json',dict(source_files=sources,source_workbook_sha256=source_summary['source_sha256'],
        allowed_body_columns=source_summary['body_columns_read'],workbook_modified=False,
        source_algorithm='fixed .6 normalized_activity_trial',thresholds_modified=False,
        outcomes_used=False,clinical_observation_scope='user_unconfirmed',formal_accuracy_computed=False))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--salience',type=Path,default=ROOT/'outputs/normalized_activity_trial_20261119_v1')
    p.add_argument('--ownership',type=Path,default=ROOT/'outputs/global_ownership_real_20261119_v1')
    p.add_argument('--maps',type=Path,default=ROOT/'outputs/cohort_wave_review_20261118_v1')
    p.add_argument('--video-index',type=Path,default=ROOT/'outputs/blind_wave_review_20261118_v1/review_key_private.json')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
