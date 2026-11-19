"""Fixed .6 relative salience trial: unchanged 87 conditions and 12 cases."""
import argparse
from collections import Counter
from html import escape
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.test_activity_strength_trial import read,save,digest
from experiments.activity_strength import filtered_runs
from experiments.normalized_activity import normalize_magnitude,score_salience
from experiments.audit_wave_evidence import control_arrays
from experiments.test_double_event_signals import double_fixture,raw_truth,one_to_one
from experiments.test_weak_wave_signals import radial_fixture
from experiments.test_global_event_ownership import truth_groups


def assess_matches(matches,expected,span,dt):
    supported=[]
    for match in matches:
        ids=np.asarray(sorted(p['section'] for p in match.get('measurements',[]) if abs(p['error_s'])<=span+dt),int)
        longest=max((len(b) for b in np.split(ids,np.flatnonzero(np.diff(ids)!=1)+1)),default=0)
        supported.append(bool(longest>=3 and match.get('direction')==expected))
    return supported


def evaluate_case(name,a,v,t,s,walls,truth,expected,span,parent=None,local=None):
    filtered=filtered_runs(a,v,t);normalized,metadata=normalize_magnitude(filtered,v)
    results=[]
    for wall in walls:
        w=wall['wall'];result=wall['reconciliation']
        scores=[score_salience(e,result['source_evidence'],normalized[:,w],v[:,w],t)
                for e in result['event_proposals']]
        selected={r['event_proposal_id']:set(r['retained_sections']) for r in scores if r['retained_sections']}
        metric=dict(before_proposals=len(scores),after_proposals=len(selected),
            before_directional=sum(e['direction'] in ('C2F','F2C') for e in result['event_proposals']),
            after_directional=sum(e['event_proposal_id'] in selected and e['direction'] in ('C2F','F2C') for e in result['event_proposals']),
            decisions=dict(Counter(r['decision'] for r in scores)))
        if truth is not None:
            report=parent['methods']['periodic_frozen']['walls'][w]['report']
            proposals=local['walls'][w]['versions']['3']['proposals']
            _,mapping=truth_groups(report,proposals,truth)
            groups=[dict(mapping[e['canonical_source_entity_id']],group_id=e['event_proposal_id']) for e in result['event_proposals']]
            reduced=[]
            for group in groups:
                if group['group_id'] not in selected:continue
                # Preserve event identity; evaluate only the selected native supports.
                assignments={i:[p for p in points if p['section'] in selected[group['group_id']]]
                             for i,points in group['assignments'].items()}
                reduced.append(dict(group,assignments=assignments))
            before=one_to_one(groups,truth['uniform_observable_count'],len(s))
            after=one_to_one(reduced,truth['uniform_observable_count'],len(s))
            for b,original in zip(before,wall['evaluation']['after_matches']):
                if b['matched']!=original['matched'] or b['native_section_fraction']!=original['native_section_fraction']:
                    raise ValueError('Frozen baseline identity evaluation differs')
            metric.update(before_matches=before,after_matches=after,
                before_supported_truth=assess_matches(before,expected,span,float(np.median(np.diff(t)))),
                after_supported_truth=assess_matches(after,expected,span,float(np.median(np.diff(t)))))
        results.append(dict(wall=w,metrics=metric,scores=scores))
    return dict(source=name,normalization=metadata,walls=results,event_count=None,event_frequency=None),filtered,normalized


def aggregate(cases):
    rows=[w['metrics'] for c in cases for w in c['walls']]
    metrics={k:sum(r[k] for r in rows) for k in ('before_proposals','after_proposals','before_directional','after_directional')}
    pairs=[(b,a) for row in rows for b,a in zip(row.get('before_matches',[]),row.get('after_matches',[]))]
    metrics.update(truth_identities=len(pairs),before_assigned=sum(b['matched'] for b,a in pairs),
        after_assigned=sum(a['matched'] for b,a in pairs),native_support_decreased=sum(
            a['native_section_fraction']<b['native_section_fraction'] for b,a in pairs))
    metrics['by_identity']={str(i):dict(before_supported=sum(len(r.get('before_supported_truth',[]))>i and r['before_supported_truth'][i] for r in rows),
        after_supported=sum(len(r.get('after_supported_truth',[]))>i and r['after_supported_truth'][i] for r in rows)) for i in (0,1)}
    return metrics


def plot_case(path,maps,filtered,normalized,ownership,result,limit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=maps['interval_center_s'];s=maps['s'];fig,axes=plt.subplots(2,2,figsize=(14,7),sharex=True,sharey=True)
    for wall in (0,1):
        axes[wall,0].pcolormesh(t,s,filtered[:,wall].T,shading='nearest',cmap='RdBu_r',vmin=-limit,vmax=limit)
        im=axes[wall,1].pcolormesh(t,s,normalized[:,wall].T,shading='nearest',cmap='magma',vmin=0,vmax=1)
        scores={r['event_proposal_id']:r for r in result['walls'][wall]['scores']}
        for event in ownership['walls'][wall]['reconciliation']['event_proposals']:
            ids=np.asarray(event['supporting_sections']);arrival=np.asarray(event['arrival_time_s'])
            for block in np.split(np.arange(len(ids)),np.flatnonzero(np.diff(ids)!=1)+1):
                if len(block):axes[wall,0].plot(arrival[block],s[ids[block]],'.-',color='#555',lw=.8,ms=2)
            keep=set(scores[event['event_proposal_id']]['retained_sections'])
            for block in scores[event['event_proposal_id']]['native_contiguous_blocks']:
                ix=np.flatnonzero(np.isin(ids,block))
                if len(ix):axes[wall,1].plot(arrival[ix],s[ids[ix]],'.-',color='#2aff63',lw=1.3,ms=3)
        for col in (0,1):
            axes[wall,col].set_facecolor('#bbb');axes[wall,col].set_ylim(0,1);axes[wall,col].set_xlabel('Acquisition time (s)')
        axes[wall,0].set_ylabel(('Anterior' if wall==0 else 'Posterior')+' / cervix -> fundus s')
        axes[wall,0].set_title('Unchanged filtered A + original proposals')
        axes[wall,1].set_title('Normalized |A|; green = >=3 adjacent salient sections')
    fig.suptitle(path.stem+' | fixed .6 salience, not validated wave counts')
    fig.subplots_adjust(left=.07,right=.91,bottom=.12,top=.87,hspace=.3)
    fig.colorbar(im,cax=fig.add_axes([.93,.17,.015,.65]),label='Relative magnitude (whole video / both walls)')
    fig.text(.07,.025,'Salient: normalized value > .6. Physical amplitudes unchanged; both walls use the same min/max. '
             'Original direction labels retained; no event recount.',fontsize=8)
    fig.savefig(path,dpi=140);plt.close(fig)


def run(args):
    args.output.mkdir(parents=True,exist_ok=True);sources={}
    def checked(path):sources[str(path)]=digest(path);return read(path)
    names=['uterine_wave/'+n+'.py' for n in ('tracking','_lk','deformation','amplitude','spectrum','radial_events')]
    names+=['experiments/'+n+'.py' for n in ('partial_phase_events','candidate_split_verifier','global_event_ownership')]
    hashes={n:digest(ROOT/n) for n in names}
    protocol=dict(status='sandbox_wang_inspired_not_curvature_replication',cutoff=.6,
        normalization='(|filtered A|-min |filtered A|)/(max |filtered A|-min |filtered A|)',
        normalization_scope='complete video, both walls, all valid filtered native samples; one min/max pair',
        paper_scope_unspecified=True,normalization_scope_prespecified=True,threshold_sweep=False,
        eligibility='three adjacent native salient sections inside existing canonical phase footprint',
        minimum_sections_source='existing numerical propagation fit support, not new clinical threshold',
        direction_and_ownership_unchanged=True,formal_amplitude_ER_unchanged=True,
        inputs='same 87 controls and 12 development cases; no invalid journal geometry',
        frozen_code_sha256=hashes,event_count=None,event_frequency=None,clinical_accuracy=None)
    save(args.output/'protocol.json',protocol)
    cache={}
    for path in sorted(args.controls.glob('texture*_rigid.npz')):
        sources[str(path)]=digest(path);cache[path.stem.split('_')[0]]=control_arrays(path)
    paths=sorted(args.ownership.glob('texture*.json'))
    if len(paths)!=87 or len(cache)!=3:raise ValueError('Expected frozen 87 conditions / three textures')
    cases=[]
    for path in paths:
        original=checked(path);parent=checked(args.parents/path.name);local=checked(args.splits/path.name)
        name=original['source'];texture=parent['texture'];family=original['family'];truth=None
        a,v,t,s,span=cache[texture]
        if family=='double':
            clean,_=double_fixture(t,s,span,parent['expected_direction'],parent['gap_s'],parent['second_amplitude'])
            truth=raw_truth(clean,t);a=a+clean[:,None]
        elif family=='single_or_negative':
            kind,amp=name[len(texture+'_single_'):].rsplit('_',1)
            clean,_,_=radial_fixture(t,s,span,kind,float(amp));a=a+clean[:,None]
            if parent['expected_direction']:truth=raw_truth(clean,t)
        elif family=='image_control':
            cp=args.controls/(name+'.npz');sources[str(cp)]=digest(cp);a,v,t,s,span=control_arrays(cp)
        else:raise ValueError('Unknown frozen family')
        result,_,_=evaluate_case(name,a,v,t,s,original['walls'],truth,parent['expected_direction'],span,parent,local)
        result.update(family=family,gap_s=parent['gap_s'],second_amplitude=parent['second_amplitude'],expected_direction=parent['expected_direction'])
        cases.append(result);save(args.output/(name+'.json'),result)
        print(name,aggregate([result]),flush=True)
    summary=dict(controls=aggregate(cases),single_levels=[dict(amplitude=amp,metrics=aggregate([
        c for c in cases if c['family']=='single_or_negative' and c['expected_direction'] and c['source'].endswith(f'{amp:.2f}')]))
        for amp in (.04,.02,.01)],double_conditions=[dict(gap_s=gap,second_amplitude=amp,metrics=aggregate([
            c for c in cases if c['family']=='double' and c['gap_s']==gap and c['second_amplitude']==amp]))
            for gap in (5.,10.,15.,20.) for amp in (.04,.01)],
        nonpropagating=aggregate([c for c in cases if c['family']=='single_or_negative' and not c['expected_direction']]),
        image_controls=aggregate([c for c in cases if c['family']=='image_control']),real_cases=[],
        event_count=None,event_frequency=None,clinical_accuracy=None)
    real=[];plots=[];magnitudes=[]
    real_paths=sorted(args.real_ownership.glob('case_*.json'))
    if len(real_paths)!=12:raise ValueError('Expected the same 12 cases')
    for path in real_paths:
        original=checked(path);mp=args.real_maps/path.stem/'maps.npz';sources[str(mp)]=digest(mp)
        with np.load(mp,allow_pickle=False) as z:maps=dict(z)
        t=maps['interval_center_s'];v=maps['valid_mask'];a=maps['A'];s=maps['s']
        result,filtered,normalized=evaluate_case(path.stem,a,v,t,s,original['walls'],None,None,float(np.median(maps['interval_dt_s'])))
        save(args.output/(path.stem+'.json'),result)
        np.savez_compressed(args.output/(path.stem+'_salience.npz'),normalized=normalized,
            salient_mask=normalized>.6,valid_filtered_mask=np.isfinite(normalized),interval_center_s=t,s=s,
            positive_A_salient_samples=((filtered>0)&(normalized>.6)).sum((1,2)),
            negative_A_salient_samples=((filtered<0)&(normalized>.6)).sum((1,2)))
        real.append(result);summary['real_cases'].append(dict(case=path.stem,normalization=result['normalization'],**aggregate([result])))
        plots.append((path.stem,maps,filtered,normalized,original,result));magnitudes.extend(abs(filtered[np.isfinite(filtered)]).tolist())
        print(path.stem,aggregate([result]),flush=True)
    summary['real_total']=aggregate(real)
    limit=float(np.percentile(magnitudes,98))
    for name,maps,f,n,o,result in plots:plot_case(args.output/(name+'.png'),maps,f,n,o,result,limit)
    if any(digest(Path(p))!=sha for p,sha in sources.items()):raise RuntimeError('Inputs changed')
    if {n:digest(ROOT/n) for n in names}!=hashes:raise RuntimeError('Frozen algorithm changed')
    protocol.update(source_files=sources,code_sha256={n:digest(ROOT/n) for n in
        ('experiments/normalized_activity.py','experiments/test_normalized_activity_trial.py')})
    save(args.output/'protocol.json',protocol);save(args.output/'summary.json',summary)
    table=''.join('<tr><td>'+escape(r['case'])+'</td><td>'+str(r['before_proposals'])+'</td><td>'+str(r['after_proposals'])+
        '</td><td>'+str(r['before_directional'])+'</td><td>'+str(r['after_directional'])+'</td><td><a href="'+escape(r['case'])+'.png">Maps</a></td></tr>' for r in summary['real_cases'])
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Fixed .6 relative salience</title>'
        '<style>body{font:17px system-ui;margin:2rem}td,th{padding:.6rem;border-bottom:1px solid #ddd}</style>'
        '<h1>Normalized radial activity: fixed 0.6</h1><p>Wang-inspired radial adaptation. One min/max over both walls/full video; no RMS or rigid-noise threshold. '
        'Proposal rows are NOT waves; sign of radial A is NOT propagation direction. Physical amplitude and ER unchanged.</p>'
        '<p><a href="summary.json">Summary</a> / <a href="protocol.json">Protocol</a></p><table><tr><th>Case</th><th>Before</th>'
        '<th>After</th><th>Directional before</th><th>Directional after</th><th>Evidence</th></tr>'+table+'</table>',encoding='utf-8')
    print('FINAL',summary['real_total'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag,path in [('controls','p3_outer_ideal_prior_20261117_v2'),('ownership','global_ownership_trial_20261119_v1'),
        ('parents','event_input_smoothing_20261118_v1'),('splits','candidate_split_trial_20261119_v1'),
        ('real-ownership','global_ownership_real_20261119_v1'),('real-maps','cohort_wave_review_20261118_v1')]:
        p.add_argument('--'+flag,type=Path,default=ROOT/'outputs'/path)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
