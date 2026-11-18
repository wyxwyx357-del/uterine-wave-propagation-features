"""Apply validated ownership rules to the existing 12 development cases.

Proposal review only; no outcome reading, tracking rerun or formal wave count.
"""
import argparse
from collections import Counter
import csv
import hashlib
from html import escape
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from uterine_wave.amplitude import band_limited_signal
from uterine_wave.io import jsonable
from experiments.candidate_split_verifier import prepare_parent,decide_split
from experiments.global_event_ownership import reconcile


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def export_missing_observations(proposals):
    """Only observation arrays encode unsupported samples as JSON null.

    Do not blanket-convert nonfinite estimates or change the computational
    signal; valid masks and the absent samples remain explicit for review.
    """
    for proposal in proposals:
        local=proposal['local_observations']
        for key in ('raw_A','smoothed_A'):
            array=np.asarray(local[key],float)
            exported=array.astype(object);exported[~np.isfinite(array)]=None
            local[key]=exported.tolist()


def figure(path,maps,walls,color_limit,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=maps['interval_center_s'];s=maps['s'];a=maps['A'];valid=maps['valid_mask']
    fig,axes=plt.subplots(2,4,figsize=(18,8),sharex=True,sharey=True)
    limit=color_limit
    def curves(ax,sections,arrival,color):
        ids=np.asarray(sections,int);arrival=np.asarray(arrival,float)
        for block in np.split(np.arange(len(ids)),np.flatnonzero(np.diff(ids)!=1)+1):
            if len(block):ax.plot(arrival[block],s[ids[block]],color=color,lw=1.3,marker='.',ms=2)
    for wall,record in enumerate(walls):
        raw=np.where(valid[:,wall],a[:,wall],np.nan);periodic=np.full_like(raw,np.nan)
        for c in range(len(s)):
            for begin,end in np.flatnonzero(np.diff(np.r_[False,valid[:,wall,c]&np.isfinite(raw[:,c]),False])).reshape(-1,2):
                if end-begin>=max(3,int(np.ceil(12/np.median(np.diff(t))))):
                    periodic[begin:end,c]=band_limited_signal(raw[begin:end,c],t[begin:end])
        for column,values in ((0,raw),(1,periodic),(3,periodic)):
            color_source=axes[wall,column].pcolormesh(t,s,values.T,shading='nearest',cmap='RdBu_r',vmin=-limit,vmax=limit,rasterized=True)
        for parent in record['parent_report']['events']:
            curves(axes[wall,1],parent['sections'],parent['arrival_time_s'],'#777777')
        proposals=record['local_split_proposals']
        selected=next((p for p in proposals if p['split_status']=='split_two'),
                      next((p for p in proposals if p['split_status']=='split_ambiguous'),proposals[0] if proposals else None))
        subtitle='no parent'
        if selected:
            local=selected['local_observations'];ids=np.asarray(local['sections'],int)
            axes[wall,2].pcolormesh(local['time_s'],s[ids],np.asarray(local['smoothed_A'],float).T,
                shading='nearest',cmap='RdBu_r',vmin=-limit,vmax=limit,rasterized=True)
            for bound in (selected['parent']['start_s'],selected['parent']['end_s']):
                axes[wall,2].axvline(bound,color='black',lw=.6,ls='--')
            subtitle=selected['parent_candidate_id']+' / '+selected['split_status']
        for event in record['reconciliation']['event_proposals']:
            color='#923b99' if event['ownership_status'].startswith('ambiguous') else '#e58b13' if event['canonical_source_entity_id'].startswith('child:') else '#21763c'
            curves(axes[wall,3],event['supporting_sections'],event['arrival_time_s'],color)
        for col,heading in enumerate(('Raw radial A','Periodic + original marks','Local auxiliary: '+subtitle,'Periodic + reconciled marks')):
            axes[wall,col].set_title(heading,fontsize=9);axes[wall,col].set_facecolor('#d5d5d5')
            axes[wall,col].set_xlim(float(t[0]),float(t[-1]));axes[wall,col].set_ylim(0,1)
            axes[wall,col].set_xlabel('Time (s)')
        axes[wall,0].set_ylabel(('Anterior' if wall==0 else 'Posterior')+' / cervix 0 -> fundus 1')
    fig.suptitle(title+' / ownership proposals, not validated wave counts',fontsize=12)
    fig.text(.02,.01,'Same A (s^-1) color limits in all panels/cases. Gray = missing/unprocessed. Green = existing; orange = new child; purple = ownership ambiguity. Local panel shows one parent only.',fontsize=8)
    fig.subplots_adjust(left=.06,right=.94,bottom=.1,top=.89,hspace=.32,wspace=.06)
    fig.colorbar(color_source,cax=fig.add_axes([.95,.16,.01,.67]),label='A (s^-1)')
    fig.savefig(path,dpi=140);plt.close(fig)


def run(args):
    validation=read(args.validation/'summary.json')
    fingerprints=read(args.validation/'protocol.json')
    if not validation['passed']:
        raise ValueError('Fixed controls have not met proposal-review acceptance')
    for name,sha in fingerprints['code_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Ownership code changed after control validation')
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=read(args.source/'protocol.json');cases=[r['review_id'] for r in protocol['selected']]
    if len(cases)!=12:raise ValueError('Expected the same 12 prespecified development cases')
    # One pooled display scale, shared across all cases and panels. This changes
    # visualization only; it is not an amplitude threshold or detector input.
    display_values=[]
    for case in cases:
        with np.load(args.source/case/'maps.npz',allow_pickle=False) as z:
            display_values.append(abs(z['A'][z['valid_mask']&np.isfinite(z['A'])]))
    color_limit=float(np.percentile(np.concatenate(display_values),98))
    floor=protocol['noise_floor_rate'];sources={};summaries=[];events=[];actions=[]
    for case in cases:
        maps_path=args.source/case/'maps.npz';parents_path=args.parents/(case+'.json')
        sources[str(maps_path)]=digest(maps_path);sources[str(parents_path)]=digest(parents_path)
        with np.load(maps_path,allow_pickle=False) as z:maps=dict(z)
        old=read(parents_path);t=maps['interval_center_s'];s=maps['s'];span=float(np.median(maps['interval_dt_s']))
        walls=[]
        for wall in old['walls']:
            w=wall['wall'];report=wall['partial_evidence']
            local=[decide_split(prepare_parent(maps['A'][:,w],maps['valid_mask'][:,w],t,s,p,
                noise_floor=floor,measurement_span_s=span),minimum_adjacent_support=3,include_signals=True)
                for p in report['events']]
            result=reconcile(report,local,t,maps['valid_mask'][:,w],measurement_span_s=span)
            export_missing_observations(local)
            walls.append(dict(wall=w,parent_report=report,local_split_proposals=local,reconciliation=result))
            for e in result['event_proposals']:
                events.append(dict(case=case,wall=w,event_proposal_id=e['event_proposal_id'],
                    ownership_status=e['ownership_status'],canonical_source=e['canonical_source_entity_id'],
                    source_parents=json.dumps(e['source_parent_ids']),from_split=e['from_split'],
                    sections=json.dumps(e['supporting_sections']),arrival_time_s=json.dumps(e['arrival_time_s']),direction=e['direction'],
                    phase_provenance=json.dumps(e['phase_ridge_provenance']),conflicts=json.dumps(e['ownership_conflicts']),
                    event_count=None,event_frequency=None,reviewed=False))
            global_actions={a['parent_id']:a for a in result['split_actions']}
            for p in local:
                a=global_actions.get(p['parent_candidate_id'])
                actions.append(dict(case=case,wall=w,parent_candidate_id=p['parent_candidate_id'],
                    local_status=p['split_status'],global_status=a['status'] if a else 'preserve_original',
                    final_event_ids=json.dumps(a['final_event_ids'] if a else []),
                    reason=a['reason'] if a else p['reason'],event_count=None,event_frequency=None))
        save(args.output/(case+'.json'),dict(case=case,walls=walls,event_count=None,event_frequency=None,
            status='development_case_ownership_proposals_require_review'))
        figure(args.output/(case+'.png'),maps,walls,color_limit,case)
        local_counts=Counter(p['split_status'] for w in walls for p in w['local_split_proposals'])
        global_counts=Counter(a['status'] for w in walls for a in w['reconciliation']['split_actions'])
        row=dict(case=case,original_parent_groups=sum(len(w['parent_report']['proposed_events']) for w in walls),
            final_proposal_rows=sum(len(w['reconciliation']['event_proposals']) for w in walls),
            local_status_counts=dict(local_counts),global_action_counts=dict(global_counts),
            ambiguous_final_rows=sum(e['ownership_status'].startswith('ambiguous') for w in walls for e in w['reconciliation']['event_proposals']),
            event_count=None,event_frequency=None)
        summaries.append(row);print(json.dumps(row),flush=True)
    for name,rows in [('event_proposals.csv',events),('ownership_actions.csv',actions)]:
        with (args.output/name).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if any(digest(Path(p))!=sha for p,sha in sources.items()):raise RuntimeError('Real input files changed')
    save(args.output/'summary.json',dict(cases=summaries,event_count=None,event_frequency=None,
        clinical_accuracy=None,strict_control_confirmation_nonregression=validation['strict_confirmation_nonregression'],
        status='proposal review only; unresolved cases retained, no algorithm freeze'))
    save(args.output/'protocol.json',dict(source_files=sources,
        control_validation_sha256=digest(args.validation/'summary.json'),
        display_limit_s_inv=color_limit,display_rule='Pooled absolute raw-A P98; one linear color scale for all cases/panels, display only',
        code_sha256={name:digest(ROOT/name) for name in ('experiments/global_event_ownership.py',
            'experiments/candidate_split_verifier.py','experiments/apply_global_ownership_real.py')},
        no_outcome_read=True,feature_definitions_unchanged=True,event_count=None,event_frequency=None))
    review_link=ROOT/'outputs'/'blind_wave_review_20261118_v1'/'index.html'
    review_key=review_link.parent/'review_key_private.json'
    aliases={r['review_id']:r['case_alias'] for r in read(review_key)['sources']} if review_key.exists() else {}
    def video_link(case,start=0.,end=None):
        alias=aliases.get(case)
        if alias is None:return ''
        video=review_link.parent/alias/'source_1x.mp4'
        relative=os.path.relpath(video,args.output).replace(os.sep,'/')
        return relative+f'#t={start:.3f}'+(f',{end:.3f}' if end is not None else '')
    review=f'<p><a href="{escape(os.path.relpath(review_link,args.output).replace(os.sep,"/"))}">Existing original-video review</a></p>' if review_link.exists() else ''
    table=''.join('<tr><td>'+escape(r['case'])+'</td><td><a href="'+r['case']+'.png">Maps</a> / <a href="'+r['case']+'.json">Evidence</a></td><td>'+escape(str(r['global_action_counts']))+'</td></tr>' for r in summaries)
    focus=''
    for r in summaries:
        if not r['global_action_counts'].get('accept_split',0):continue
        record=read(args.output/(r['case']+'.json'))
        for w in record['walls']:
            proposals={p['parent_candidate_id']:p for p in w['local_split_proposals']}
            for action in w['reconciliation']['split_actions']:
                if action['status']!='accept_split':continue
                parent=proposals[action['parent_id']]['parent'];start=parent['start_s'];end=parent['end_s']
                url=video_link(r['case'],start,end)
                focus+='<section><p>'+escape(r['case'])+' / wall '+str(w['wall'])+' / '+escape(action['parent_id'])+f' / {start:.2f}–{end:.2f}s</p>'
                if url:focus+='<video controls preload="metadata" style="max-width:650px;width:100%" src="'+escape(url)+'"></video>'
                focus+='<p><a href="'+r['case']+'.png">同步时空图</a> / <a href="'+r['case']+'.json">完整证据</a></p></section>'
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Ownership proposal review</title><style>body{font:16px system-ui;margin:2rem;max-width:1200px}td,th{padding:.7rem;border-bottom:1px solid #ddd}a{margin-right:.5rem}</style><h1>事件归属提案复核</h1><p>所有结果均未人工确认。提案行数不是波次数；正式次数、频率为空。先检查同一活动是否重复、相邻活动是否分开、弱活动是否保留。紫色表示归属歧义。每例JSON保留全部父候选、局部证据和来源。</p><p>优先复核发生新增拆分的位置；下方沿用原一倍速视频，未重新编码：</p>'+focus+review+'<p><a href="event_proposals.csv">事件提案表</a> / <a href="ownership_actions.csv">父候选处理表</a></p><table><tr><th>病例</th><th>对照与来源</th><th>归属动作</th></tr>'+table+'</table>',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--parents',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
