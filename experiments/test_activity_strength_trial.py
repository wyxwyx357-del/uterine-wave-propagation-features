"""One fixed whole-activity intensity trial; reuse 87 controls and 12 cases."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.activity_strength import RigidReference, filtered_runs, score_activity
from experiments.audit_wave_evidence import control_arrays
from experiments.test_double_event_signals import double_fixture, raw_truth, one_to_one
from experiments.test_weak_wave_signals import radial_fixture
from experiments.test_global_event_ownership import truth_groups
from uterine_wave.io import jsonable


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, obj): path.write_text(json.dumps(jsonable(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def selection(reconciliation, filtered, valid, t, reference):
    scores = [score_activity(e, reconciliation['source_evidence'], filtered, valid, t, reference)
              for e in reconciliation['event_proposals']]
    keep = {e['event_proposal_id'] for e in scores if e['decision']=='above_rigid_reference'}
    pending = {e['event_proposal_id'] for e in scores if e['decision'].startswith('unscored_')}
    events = reconciliation['event_proposals']
    return scores, keep, dict(before_proposal_rows=len(events),prominent_proposal_rows=len(keep),
        before_directional=sum(e['direction'] in ('C2F','F2C') for e in events),
        prominent_directional=sum(e['event_proposal_id'] in keep and e['direction'] in ('C2F','F2C') for e in events),
        unscored_proposal_rows=len(pending),
        not_below_reference_rows=len(keep|pending),
        not_below_reference_directional=sum(e['event_proposal_id'] in keep|pending and e['direction'] in ('C2F','F2C') for e in events),
        decisions=dict(Counter(e['decision'] for e in scores)),
        prominent_with_ownership_ambiguity=sum(e['event_proposal_id'] in keep and
            e['ownership_status'].startswith('ambiguous') for e in events),event_count=None,event_frequency=None)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {}
    def checked(path):
        sources[str(path)] = digest(path)
        return read(path)
    frozen_names = ['uterine_wave/'+n+'.py' for n in ('tracking','_lk','deformation','amplitude','spectrum','radial_events')]
    frozen_names += ['experiments/'+n+'.py' for n in ('partial_phase_events','candidate_split_verifier','global_event_ownership')]
    hashes = {p:digest(ROOT/p) for p in frozen_names}
    protocol = dict(status='sandbox',score='median per-section RMS across canonical signed-phase footprint',
        reference='maximum over matched native time/spatial translations, complete rigid windows only',
        threshold_ratio=1.,threshold_semantics='engineering prominence above matched calibration residual maximum',
        validation='leave-one-texture-out; real cases use all three calibration textures',
        same_frozen_periodic_filter=True,no_parameter_sweep=True,formal_features_unchanged=True,
        no_patient_outcomes=True,invalid_published_video_trial_excluded=True,
        frozen_code_sha256=hashes,event_count=None,event_frequency=None,clinical_accuracy=None)
    save(args.output/'protocol.json',protocol)
    banks = []; cache = {}
    for path in sorted(args.controls.glob('texture*_rigid.npz')):
        sources[str(path)] = digest(path)
        a,v,t,s,span = control_arrays(path)
        texture = path.stem.split('_')[0];cache[texture]=(a,v,t,s,span)
        banks.append((texture, filtered_runs(a,v,t), t))
    if len(banks)!=3: raise ValueError('Expected existing three rigid textures')
    references = {name:RigidReference([b for b in banks if b[0]!=name]) for name,_,_ in banks}
    all_reference = RigidReference(banks)
    paths = sorted(args.ownership.glob('texture*.json'))
    if len(paths)!=87:raise ValueError('Expected original 87 controls')
    rows=[]
    for path in paths:
        case=checked(path);old=checked(args.parents/path.name);local=checked(args.splits/path.name)
        name=case['source'];texture=old['texture'];family=case['family']
        a,v,t,s,span=cache[texture];truth=None
        if family=='double':
            clean,_=double_fixture(t,s,span,old['expected_direction'],old['gap_s'],old['second_amplitude'])
            truth=raw_truth(clean,t);a=a+clean[:,None]
        elif family=='single_or_negative':
            kind,amp=name[len(texture+'_single_'):].rsplit('_',1)
            clean,_,_=radial_fixture(t,s,span,kind,float(amp));a=a+clean[:,None]
            if old['expected_direction']:truth=raw_truth(clean,t)
        elif family=='image_control':
            cp=args.controls/(name+'.npz');sources[str(cp)]=digest(cp)
            a,v,t,s,span=control_arrays(cp)
        else:raise ValueError('Unknown existing fixture family')
        filtered=filtered_runs(a,v,t);walls=[]
        for wall in case['walls']:
            w=wall['wall'];result=wall['reconciliation']
            scores,keep,metrics=selection(result,filtered[:,w],v[:,w],t,references[texture])
            if truth is not None:
                report=old['methods']['periodic_frozen']['walls'][w]['report']
                proposals=local['walls'][w]['versions']['3']['proposals']
                _,mapping=truth_groups(report,proposals,truth)
                groups=[dict(mapping[e['canonical_source_entity_id']],group_id=e['event_proposal_id'])
                        for e in result['event_proposals']]
                before=one_to_one(groups,truth['uniform_observable_count'],len(s))
                after=one_to_one([g for g in groups if g['group_id'] in keep],truth['uniform_observable_count'],len(s))
                unknown={e['event_proposal_id'] for e in scores if e['decision'].startswith('unscored_')}
                unresolved_preserved=one_to_one([g for g in groups if g['group_id'] in keep|unknown],truth['uniform_observable_count'],len(s))
                # Existing ownership evaluation must reproduce before testing a new selector.
                for b,recorded in zip(before,wall['evaluation']['after_matches']):
                    if b['matched']!=recorded['matched'] or b['native_section_fraction']!=recorded['native_section_fraction']:
                        raise ValueError('Existing ownership match failed to reproduce')
                metrics.update(before_matches=before,after_matches=after,unscored_preserved_matches=unresolved_preserved)
            walls.append(dict(wall=w,metrics=metrics,scores=scores))
        row=dict(source=name,family=family,expected_direction=old['expected_direction'],
            known_nonpropagating=old['known_nonpropagating'],gap_s=old['gap_s'],
            second_amplitude=old['second_amplitude'],walls=walls,event_count=None,event_frequency=None)
        save(args.output/(name+'.json'),row);rows.append(row)
        print(name,[(w['metrics']['before_directional'],w['metrics']['prominent_directional']) for w in walls],flush=True)

    def aggregate(selected):
        metrics=[w['metrics'] for r in selected for w in r['walls']]
        result={key:sum(m[key] for m in metrics) for key in ('before_proposal_rows','prominent_proposal_rows',
            'before_directional','prominent_directional','prominent_with_ownership_ambiguity',
            'unscored_proposal_rows','not_below_reference_rows','not_below_reference_directional')}
        result['decisions']=dict(sum((Counter(m['decisions']) for m in metrics),Counter()))
        paired=[(b,a) for m in metrics for b,a in zip(m.get('before_matches',[]),m.get('after_matches',[]))]
        result.update(observable_truth_events=len(paired),before_assigned=sum(b['matched'] for b,a in paired),
            after_assigned=sum(a['matched'] for b,a in paired),not_promoted_to_prominent=sum(b['matched'] and not a['matched'] for b,a in paired),
            prominent_support_decreased=sum(a['native_section_fraction']<b['native_section_fraction'] for b,a in paired))
        result['by_identity']={str(i):dict(before_assigned=sum(len(m.get('before_matches',[]))>i and m['before_matches'][i]['matched'] for m in metrics),
            after_assigned=sum(len(m.get('after_matches',[]))>i and m['after_matches'][i]['matched'] for m in metrics)) for i in (0,1)}
        corrected=[(b,a) for m in metrics for b,a in zip(m.get('before_matches',[]),m.get('unscored_preserved_matches',[]))]
        result.update(assigned_if_unscored_preserved=sum(a['matched'] for b,a in corrected),
            removed_known_assignments_by_low_strength=sum(b['matched'] and not a['matched'] for b,a in corrected),
            support_decreased_by_low_strength=sum(a['native_section_fraction']<b['native_section_fraction'] for b,a in corrected))
        return result
    summary=dict(controls=aggregate(rows),
        single_levels=[dict(amplitude=amp,metrics=aggregate([r for r in rows if r['family']=='single_or_negative' and
            r['expected_direction'] and r['source'].endswith(f'{amp:.2f}')])) for amp in (.04,.02,.01)],
        double_conditions=[dict(gap_s=gap,second_amplitude=amp,metrics=aggregate([r for r in rows if
            r['family']=='double' and r['gap_s']==gap and r['second_amplitude']==amp]))
            for gap in (5.,10.,15.,20.) for amp in (.04,.01)],
        nonpropagating_signals=aggregate([r for r in rows if r['family']=='single_or_negative' and r['known_nonpropagating']]),
        image_controls=aggregate([r for r in rows if r['family']=='image_control']),
        rigid_only=aggregate([r for r in rows if '_rigid_only_' in r['source'] or r['source'].endswith('_rigid')]),
        real_cases=[],event_count=None,event_frequency=None,clinical_accuracy=None)

    real_paths=sorted(args.real_ownership.glob('case_*.json'))
    if len(real_paths)!=12:raise ValueError('Expected existing 12 real development cases')
    for path in real_paths:
        case=checked(path);mp=args.real_maps/path.stem/'maps.npz';sources[str(mp)]=digest(mp)
        with np.load(mp,allow_pickle=False) as z:maps=dict(z)
        t=maps['interval_center_s'];v=maps['valid_mask'];filtered=filtered_runs(maps['A'],v,t);walls=[]
        for wall in case['walls']:
            w=wall['wall'];scores,keep,metrics=selection(wall['reconciliation'],filtered[:,w],v[:,w],t,all_reference)
            walls.append(dict(wall=w,scores=scores,metrics=metrics))
        row=dict(source=path.stem,walls=walls,event_count=None,event_frequency=None,
            interpretation='prominent proposals, not validated wave counts; no outcomes or clinical frequency used')
        save(args.output/(path.stem+'.json'),row)
        summary['real_cases'].append(dict(case=path.stem,**aggregate([row])))
        print(path.stem,[(w['metrics']['before_directional'],w['metrics']['prominent_directional']) for w in walls],flush=True)
    summary['real_total']=aggregate([dict(walls=[dict(metrics=r)]) for r in summary['real_cases']])
    save(args.output/'summary.json',summary)
    if any(digest(Path(p))!=sha for p,sha in sources.items()):raise RuntimeError('Saved trial inputs changed')
    if {p:digest(ROOT/p) for p in frozen_names}!=hashes:raise RuntimeError('Frozen algorithms changed')
    protocol['source_files']=sources
    protocol['trial_code_sha256']={p:digest(ROOT/p) for p in ('experiments/activity_strength.py','experiments/test_activity_strength_trial.py')}
    save(args.output/'protocol.json',protocol)
    print('SUMMARY',json.dumps({k:v for k,v in summary.items() if k!='real_cases'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controls',type=Path,default=ROOT/'outputs/p3_outer_ideal_prior_20261117_v2')
    p.add_argument('--ownership',type=Path,default=ROOT/'outputs/global_ownership_trial_20261119_v1')
    p.add_argument('--parents',type=Path,default=ROOT/'outputs/event_input_smoothing_20261118_v1')
    p.add_argument('--splits',type=Path,default=ROOT/'outputs/candidate_split_trial_20261119_v1')
    p.add_argument('--real-ownership',type=Path,default=ROOT/'outputs/global_ownership_real_20261119_v1')
    p.add_argument('--real-maps',type=Path,default=ROOT/'outputs/cohort_wave_review_20261118_v1')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
