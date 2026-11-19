"""Patient-level salience ledger and clinical comparison, no parameter fitting."""
import argparse
import csv
from html import escape
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.test_activity_strength_trial import read,save,digest
from experiments.patient_activity_aggregation import patient_summary


def run(args):
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    references=read(args.clinical/'cases.json');lookup={r['case']:r for r in references}
    sources={str(args.clinical/'cases.json'):digest(args.clinical/'cases.json')}
    rows=[];details=[];all_merges=0;all_input=0
    for path in sorted(args.salience.glob('case_*.json')):
        name=path.stem;original_path=args.ownership/path.name;mp=args.maps/name/'maps.npz'
        for p in (path,original_path,mp):sources[str(p)]=digest(p)
        with np.load(mp,allow_pickle=False) as z:maps=dict(z)
        result=patient_summary(read(path),read(original_path),maps)
        ref=lookup[name];result.update(case=name,clinical_positive_cpm=ref['clinical_positive_cpm'],
            clinical_negative_cpm=ref['clinical_negative_cpm'],clinical_column_sum_cpm=ref['clinical_column_sum_cpm'],
            clinical_observation_correspondence='user_unconfirmed',positive_clinical_direction='unconfirmed',
            clinical_excel_row=ref['excel_row'],actual_video_duration_s=ref['actual_video_duration_s'])
        save(args.output/(name+'.json'),result);all_merges+=result['cross_wall_pairs_merged'];all_input+=result['input_wall_proposal_rows']
        row={k:result[k] for k in ('case','clinical_positive_cpm','clinical_negative_cpm','clinical_column_sum_cpm',
            'actual_video_duration_s','input_wall_proposal_rows','patient_candidate_rows','directional_candidate_rows',
            'local_or_unresolved_rows','cross_wall_pairs_merged','unresolved_cross_wall_rows','candidate_activity_rate_cpm',
            'candidate_propagation_rate_cpm','conditional_identity_count_range','conditional_candidate_rate_range_cpm')}
        row['effective_local_observation_s']=result['exposure']['seconds'];rows.append(row)
        for direction in ('C2F','F2C'):
            bucket=result['directional_summaries'][direction]
            for i,label in enumerate(('min','max')):
                row[direction+'_count_'+label]=bucket['conditional_identity_count_range'][i]
                row[direction+'_cpm_'+label]=bucket['candidate_rate_range_cpm'][i]
        row['other_candidate_rows']=result['non_directional_summary']['patient_candidate_rows']
        video=(args.clinical/ref['original_video_relative_path']).resolve()
        if not video.exists():raise ValueError('Original video missing')
        relative=os.path.relpath(video,args.output).replace(os.sep,'/')
        def f(x):return '—' if x is None else f'{x:.2f}'
        def interval(values,integer=False):
            a,b=values
            if a is None:return '—'
            show=lambda x:str(int(x)) if integer else f(x)
            return show(a) if a==b else show(a)+'–'+show(b)
        panel=('<section id="'+escape(name)+'"><h2>'+escape(name)+'</h2><p>程序C2F候选率 '+interval(result['directional_summaries']['C2F']['candidate_rate_range_cpm'])+
            ' 次/分；F2C候选率 '+interval(result['directional_summaries']['F2C']['candidate_rate_range_cpm'])+
            ' 次/分；非单向/方向未定候选 '+str(row['other_candidate_rows'])+' 条，不并入单向传播频率。'+
            ' 局部可观察时长 '+f(result['exposure']['seconds'])+' 秒。</p><video controls preload="none" src="'+escape(relative)+'"></video>')
        display_activities=[e for d in ('C2F','F2C') for e in result['directional_summaries'][d]['activities']]+result['non_directional_summary']['activities']
        for activity in display_activities:
            start=activity['evidence_start_s'];end=activity['evidence_end_s']
            sources_label='、'.join(('前壁' if w==0 else '后壁') for w in activity['walls'])
            kind=activity['direction'] or '局部/方向未定'
            panel+='<details><summary>'+escape(activity['patient_candidate_id'])+' · '+escape(kind)+f' · {start:.1f}–{end:.1f}s</summary>'
            panel+='<p>来源：'+sources_label+'；记录状态：'+escape(activity['ownership_status'])+'。这是一个患者级候选身份，不是两份壁计数。</p>'
            panel+='<video controls preload="none" src="'+escape(relative+f'#t={start:.3f},{end:.3f}')+'"></video></details>'
        if not result['activities']:panel+='<p>当前0.6规则下无显著候选；不据此断言整个子宫无活动。</p>'
        panel+='<p><a href="'+name+'.json">统一活动清单与跨壁对应证据</a></p></section>';details.append(panel)
        print(name,row,flush=True)
    if len(rows)!=12:raise ValueError('Expected original 12 cases')
    csv_columns=['case','clinical_positive_cpm','clinical_negative_cpm','C2F_count_min','C2F_count_max','C2F_cpm_min','C2F_cpm_max',
        'F2C_count_min','F2C_count_max','F2C_cpm_min','F2C_cpm_max','other_candidate_rows','effective_local_observation_s']
    with (args.output/'patient_comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=csv_columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    summary=dict(cases=rows,input_wall_rows=all_input,patient_candidate_rows=sum(r['patient_candidate_rows'] for r in rows),
        conditional_total_count_range=[sum(r['conditional_identity_count_range'][i] for r in rows) for i in (0,1)],
        merged_cross_wall_pairs=all_merges,unresolved_cross_wall_rows=sum(r['unresolved_cross_wall_rows'] for r in rows),
        primary_reporting='direction_first',direction_count_ranges={d:[sum(r[d+'_count_'+k] for r in rows) for k in ('min','max')] for d in ('C2F','F2C')},
        non_directional_candidate_rows=sum(r['other_candidate_rows'] for r in rows),
        interpretation='C2F/F2C reported separately; other activities excluded from directional rates; walls are measurement provenance',
        formal_event_count=None,formal_event_frequency_cpm=None,clinical_accuracy=None,ICC=None)
    save(args.output/'summary.json',summary)
    cells=''
    for r in rows:
        cells+=('<tr><td><a href="#'+r['case']+'">'+r['case']+'</a></td><td>'+f(r['clinical_positive_cpm'])+' / '+f(r['clinical_negative_cpm'])+
            '</td><td>'+interval([r['C2F_cpm_min'],r['C2F_cpm_max']])+'</td><td>'+interval([r['F2C_cpm_min'],r['F2C_cpm_max']])+
            '</td><td>'+str(r['other_candidate_rows'])+
            '</td><td>'+f(r['effective_local_observation_s'])+'</td><td>'+str(r['unresolved_cross_wall_rows'])+'</td></tr>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>患者级统一活动汇总</title>
<style>body{font:17px/1.65 system-ui;max-width:1280px;margin:25px auto;padding:15px;color:#203040}td,th{padding:.5rem;border-bottom:1px solid #ccc}section{border-top:2px solid #ccc;margin-top:3rem}video{width:100%;max-height:580px;background:#111}details{padding:.6rem;background:#f2f5f8;margin:.5rem 0}.note{padding:1rem;background:#fff1d7}</style>
<h1>按传播方向汇总显著活动</h1><p>先按宫颈→宫底C2F、宫底→宫颈F2C分流，再在同方向内根据时空证据识别是否为同一次活动。方向相同不等于同一次；方向不同不合并。前后壁仅作为来源和对应线索。局部、复杂及方向未定活动独立列出，不加入单向传播频率。</p>
<p class="note">方向沿用原候选的宫颈—宫底到达顺序判断，不由壁标签或径向A正负决定；显著筛选子段尚未独立重新拟合方向。两列为待验证的传播候选率。未定同方向对应显示条件范围，不是统计置信区间。分母是至少一壁有3个相邻有效位置的测量区间并集，安静时段计入，两壁时间不相加。临床是否针对同一视频不确定，正反解剖含义未确认，暂不将正反列强配C2F/F2C，不计算ICC或准确率。</p>
<p><a href="patient_comparison.csv">患者级对照CSV</a> · <a href="summary.json">汇总与来源</a></p>
<table><tr><th>病例</th><th>临床正/反<br>次/分，方向未对应</th><th>C2F候选<br>次/分</th><th>F2C候选<br>次/分</th><th>非单向/方向未定<br>条数，不并入频率</th><th>局部可观察<br>秒</th><th>未定跨壁<br>归属条数</th></tr>'''+cells+'</table>'+''.join(details)+'</html>'
    (args.output/'index.html').write_text(page,encoding='utf-8')
    if any(digest(Path(p))!=sha for p,sha in sources.items()):raise RuntimeError('Inputs changed')
    save(args.output/'protocol.json',dict(source_files=sources,cutoff=.6,threshold_changed=False,
        cross_wall_tolerance='existing measurement span + one native sample',
        identity='mutually unique, >=3 matching native positions, phase order/peaks/arrival/direction agree; independent continuity on each wall',
        primary_reporting='direction_first',non_directional_excluded_from_directional_rates=True,
        no_cross_wall_transitive_chains=True,no_outcomes_read=True,no_clinical_fitting=True,
        code_sha256={n:digest(ROOT/n) for n in ('experiments/patient_activity_aggregation.py','experiments/summarize_patient_activity.py')},
        formal_event_frequency_cpm=None,clinical_accuracy=None))
    print('FINAL',summary['input_wall_rows'],summary['patient_candidate_rows'],all_merges,summary['unresolved_cross_wall_rows'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag,folder in [('salience','normalized_activity_trial_20261119_v1'),('ownership','global_ownership_real_20261119_v1'),
                        ('maps','cohort_wave_review_20261118_v1'),('clinical','clinical_salience_comparison_20261119_v1')]:
        p.add_argument('--'+flag,type=Path,default=ROOT/'outputs'/folder)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
