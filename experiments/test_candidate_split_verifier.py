"""Run ONLY the existing 87 conditions for parent-constrained split proposals."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from uterine_wave.io import jsonable
from experiments.audit_wave_evidence import control_arrays
from experiments.test_double_event_signals import double_fixture,raw_truth,evaluate,describe_piece,one_to_one
from experiments.test_weak_wave_signals import radial_fixture
from experiments.candidate_split_verifier import prepare_parent,decide_split


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def _child_groups(proposal,ridges,truth):
    groups=[]
    for child in (proposal['child_1'],proposal['child_2']):
        described=describe_piece(child,ridges,truth)
        groups.append(dict(group_id=child['event_id'],assignments=described['assignments'],
                           classification=child['classification']))
    return groups


def assess(report,prepared,proposals,truth):
    output=dict(status_counts=dict(Counter(p['split_status'] for p in proposals)),
        parent_count=len(proposals),split_count=sum(p['split_status']=='split_two' for p in proposals),
        directional_children=sum(c['propagation_supported'] for p in proposals if p['split_status']=='split_two'
                                 for c in (p['child_1'],p['child_2'])),
        split_details=[],before_matches=None,after_matches=None,
        formal_event_count=None,event_frequency=None)
    if truth is None:
        return output
    before=evaluate(report,truth)
    original_ridges={r['ridge_id']:r for r in report['phase_ridges']}
    original_events={e['event_id']:e for e in report['events']}
    described={f['fragment_id']:f for f in before['fragments']}
    after_groups=[];split_by_id={p['parent_candidate_id']:p for p in proposals if p['split_status']=='split_two'}
    for group in report['proposed_events']:
        combined={}
        for fid in group['fragment_ids']:
            if fid in split_by_id:
                continue
            for identity,points in described[fid]['assignments'].items():
                combined.setdefault(identity,[]).extend(points)
        if combined:
            after_groups.append(dict(group_id='original_'+group['proposal_id'],assignments=combined,
                                     classification=next(e['classification'] for e in report['events']
                                         if e['event_id']==group['fragment_ids'][0])))
    number=truth['uniform_observable_count']
    for p in proposals:
        if p['split_status']!='split_two':
            continue
        local={r['ridge_id']:r for r in prepared[p['parent_candidate_id']]['local_phase_ridges']}
        groups=_child_groups(p,local,truth)
        after_groups.extend(groups)
        matched=one_to_one(groups,number,len(truth['columns'])) if number is not None else []
        supports_distinct_two=(number==2 and sum(m['matched'] for m in matched)==2 and
            all(len(g['assignments'])==1 for g in groups) and
            all(len({x['section'] for x in m.get('measurements',[])})>=p['minimum_adjacent_support'] for m in matched))
        original=describe_piece(original_events[p['parent_candidate_id']],original_ridges,truth)
        output['split_details'].append(dict(parent_candidate_id=p['parent_candidate_id'],
            supports_two_distinct_raw_truth_events=supports_distinct_two,
            child_truth_matches=matched,child_sections=[p['child_1']['sections'],p['child_2']['sections']],
            original_parent_assigned_truth_ids=sorted(original['assignments']),
            original_parent_already_one_raw_identity=len(original['assignments'])==1))
    output.update(before_matches=before['matches'],
        after_matches=one_to_one(after_groups,number,len(truth['columns'])) if number is not None else None,
        before_supported_group_counts=before['multiple_group_support'],
        after_supported_group_counts=[sum(bool(g['assignments'].get(i)) for g in after_groups) for i in range(number or 0)],
        true_split_parents=sum(p['supports_two_distinct_raw_truth_events'] for p in output['split_details']),
        false_split_count=sum(not p['supports_two_distinct_raw_truth_events'] for p in output['split_details']),
        false_split_fraction=sum(not p['supports_two_distinct_raw_truth_events'] for p in output['split_details'])/len(proposals)
            if proposals else None)
    return output


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=read(args.source/'protocol.json')
    # Freeze the parent producer rather than changing it to obtain a larger
    # parent. Source data, old parent JSON and core functions remain unchanged.
    for name in ('experiments/partial_phase_events.py','uterine_wave/radial_events.py','uterine_wave/amplitude.py'):
        if digest(ROOT/name)!=protocol['code_sha256'][name]:
            raise ValueError('Frozen parent producer differs from the baseline trial')
    paths=sorted(args.source.glob('texture*.json'))
    if len(paths)!=87:
        raise ValueError('Expected exactly the existing 87 condition records')
    calibration=read(args.noise_reference)
    sources={};cached={};summaries=[]
    for path in paths:
        sources[str(path)]=digest(path)
        old=read(path);texture=old['texture'];family=old['family']
        control_path=args.controls_root/(texture+'_rigid.npz')
        if family=='image_control':
            control_path=args.controls_root/(old['source']+'.npz')
        if str(control_path) not in cached:
            if digest(control_path)!=protocol['source_files'][str(control_path)]:
                raise ValueError('Frozen source cache changed')
            sources[str(control_path)]=digest(control_path)
            cached[str(control_path)]=control_arrays(control_path)
        noise,valid,t,s,span=cached[str(control_path)]
        truth=None
        if family=='double':
            clean,_=double_fixture(t,s,span,old['expected_direction'],old['gap_s'],old['second_amplitude'])
            signal=noise+clean[:,None];truth=raw_truth(clean,t)
        elif family=='single_or_negative':
            kind,amplitude=old['source'][len(texture+'_single_'):].rsplit('_',1)
            clean,_,_=radial_fixture(t,s,span,kind,float(amplitude))
            signal=noise+clean[:,None]
            if old['expected_direction']:
                truth=raw_truth(clean,t)
        else:
            signal=noise
        floor=calibration['leave_one_texture_out_thresholds'][texture[7:]]
        record=dict(source=old['source'],family=family,known_nonpropagating=old['known_nonpropagating'],
            expected_direction=old['expected_direction'],gap_s=old['gap_s'],second_amplitude=old['second_amplitude'],
            raw_observable_count=truth['uniform_observable_count'] if truth else None,walls=[])
        for wall in old['methods']['periodic_frozen']['walls']:
            w=wall['wall'];parent_report=wall['report']
            prepared={p['event_id']:prepare_parent(signal[:,w],valid[:,w],t,s,p,
                noise_floor=floor,measurement_span_s=span) for p in parent_report['events']}
            versions={}
            for minimum in (3,4,5):
                proposals=[decide_split(prepared[p['event_id']],minimum_adjacent_support=minimum,
                                       include_signals=minimum==3) for p in parent_report['events']]
                if [p['parent'] for p in proposals]!=parent_report['events']:
                    raise RuntimeError('A parent candidate was mutated or omitted')
                evaluation=assess(parent_report,prepared,proposals,truth)
                # Record the maximum observable truth support available INSIDE
                # any existing parent, without passing truth to the verifier.
                coverage=[]
                if truth and truth['uniform_observable_count']==2:
                    for p in parent_report['events']:
                        ids=[c for c in p['sections'] if all(p['start_s']<=event['arrival_s']<=p['end_s']
                                                            for event in truth['columns'][c])]
                        runs=np.split(np.asarray(ids,int),np.flatnonzero(np.diff(ids)!=1)+1)
                        coverage.append(max((len(x) for x in runs),default=0))
                evaluation['parent_truth_joint_support_upper_bound']=max(coverage,default=0) if truth else None
                versions[str(minimum)]=dict(proposals=proposals,evaluation=evaluation)
            record['walls'].append(dict(wall=w,versions=versions))
        save(args.output/(old['source']+'.json'),record)
        # Compact aggregate records retain wall-level metrics without cloning
        # all local curves in the summary.
        summaries.append(dict(source=record['source'],family=family,gap_s=record['gap_s'],
            second_amplitude=record['second_amplitude'],known_nonpropagating=record['known_nonpropagating'],
            raw_observable_count=record['raw_observable_count'],expected_direction=record['expected_direction'],
            walls=[dict(wall=w['wall'],evaluations={n:v['evaluation'] for n,v in w['versions'].items()}) for w in record['walls']]))
        print(old['source'],flush=True)
    if any(digest(Path(p))!=sha for p,sha in sources.items()):
        raise RuntimeError('Source records or control data changed')
    def aggregate(rows,minimum):
        items=[w['evaluations'][str(minimum)] for r in rows for w in r['walls']]
        known=[e for e in items if e.get('false_split_count') is not None]
        statuses=Counter()
        for e in items:statuses.update(e['status_counts'])
        best_children=[]
        for e in items:
            successful=[d for d in e['split_details'] if d['supports_two_distinct_raw_truth_events']]
            if successful:
                best=max(successful,key=lambda d:min(m['native_section_fraction'] for m in d['child_truth_matches']))
                best_children.extend(best['child_truth_matches'])
        matches=[m for e in items for m in e['after_matches'] or []]
        return dict(wall_conditions=len(items),parent_count=sum(e['parent_count'] for e in items),
            status_counts=dict(statuses),
            walls_with_true_two_child_split=sum(any(d['supports_two_distinct_raw_truth_events']
                for d in e['split_details']) for e in items),
            walls_with_multiple_true_split_parents=sum(sum(d['supports_two_distinct_raw_truth_events']
                for d in e['split_details'])>1 for e in items),
            false_split_count=sum(e['false_split_count'] for e in known) if known else None,
            false_split_fraction=sum(e['false_split_count'] for e in known)/sum(e['parent_count'] for e in known)
                if known and sum(e['parent_count'] for e in known) else None,
            negative_added_splits=sum(w['evaluations'][str(minimum)]['split_count'] for r in rows
                if r['known_nonpropagating'] for w in r['walls']) if any(r['known_nonpropagating'] for r in rows) else None,
            negative_parent_count=sum(w['evaluations'][str(minimum)]['parent_count'] for r in rows
                if r['known_nonpropagating'] for w in r['walls']) if any(r['known_nonpropagating'] for r in rows) else None,
            negative_directional_children=sum(w['evaluations'][str(minimum)]['directional_children'] for r in rows
                if r['known_nonpropagating'] for w in r['walls']) if any(r['known_nonpropagating'] for r in rows) else None,
            child_minimum_native_section_fraction=min((m['native_section_fraction'] for m in best_children),default=None),
            child_arrival_RMSE_median_s=float(np.median([m['arrival_RMSE_s'] for m in best_children])) if best_children else None,
            child_arrival_RMSE_max_s=max((m['arrival_RMSE_s'] for m in best_children),default=None),
            after_truth_1_assigned=sum(bool(e['after_matches'] and e['after_matches'][0]['matched']) for e in items),
            after_truth_2_assigned=sum(bool(e['after_matches'] and len(e['after_matches'])>1 and e['after_matches'][1]['matched']) for e in items),
            after_truth_1_assignment_fraction=sum(bool(e['after_matches'] and e['after_matches'][0]['matched']) for e in items)/len(items) if items else None,
            after_truth_2_assignment_fraction=sum(bool(e['after_matches'] and len(e['after_matches'])>1 and e['after_matches'][1]['matched']) for e in items)/len(items)
                if items and any(e['after_matches'] and len(e['after_matches'])>1 for e in items) else None,
            before_minimum_native_section_fraction=min((m['native_section_fraction'] for e in items for m in e['before_matches'] or []),default=None),
            split_parents_originally_one_raw_identity=sum(d['original_parent_already_one_raw_identity'] for e in items for d in e['split_details']),
            after_truth_RMSE_by_identity={str(i):dict(
                median_s=float(np.median([e['after_matches'][i]['arrival_RMSE_s'] for e in items
                    if e['after_matches'] and len(e['after_matches'])>i and e['after_matches'][i]['matched']]))
                    if any(e['after_matches'] and len(e['after_matches'])>i and e['after_matches'][i]['matched'] for e in items) else None,
                minimum_native_section_fraction=min((e['after_matches'][i]['native_section_fraction'] for e in items
                    if e['after_matches'] and len(e['after_matches'])>i),default=None)) for i in (0,1)},
            after_unmatched_truths=sum(not m['matched'] for m in matches),
            after_minimum_native_section_fraction=min((m['native_section_fraction'] for m in matches),default=None),
            before_duplicate_truths=sum(sum(n>1 for n in e.get('before_supported_group_counts',[])) for e in items),
            after_duplicate_truths=sum(sum(n>1 for n in e.get('after_supported_group_counts',[])) for e in items),
            parent_joint_support_upper_bound_minimum=min((e['parent_truth_joint_support_upper_bound'] for e in items
                if e['parent_truth_joint_support_upper_bound'] is not None),default=None))
    summary=dict(parameters=dict(adjacent_support=[3,4,5],smoothing='Existing actual measurement span only',
                                 context='One measurement span; child arrival inside original parent'),
        double_conditions=[dict(gap_s=g,second_amplitude=a,versions={str(n):aggregate([
            r for r in summaries if r['family']=='double' and r['gap_s']==g and r['second_amplitude']==a],n) for n in (3,4,5)})
            for g in (5.,10.,15.,20.) for a in (.04,.01)],
        single_levels=[dict(amplitude=a,versions={str(n):aggregate([r for r in summaries
            if r['family']=='single_or_negative' and r['expected_direction'] and r['source'].endswith(f'{a:.2f}')],n)
            for n in (3,4,5)}) for a in (.04,.02,.01)],
        signal_negatives={str(n):aggregate([r for r in summaries if r['family']=='single_or_negative' and r['known_nonpropagating']],n) for n in (3,4,5)},
        image_controls={str(n):aggregate([r for r in summaries if r['family']=='image_control'],n) for n in (3,4,5)},
        event_count=None,event_frequency=None,clinical_accuracy=None,real_cases_run=False)
    save(args.output/'summary.json',summary);save(args.output/'all_conditions.json',summaries)
    save(args.output/'protocol.json',dict(source_files=sources,baseline_commit='5f9e3e945b2e124dc29eec2c5be415aa7e97d55f',
        code_sha256={name:digest(ROOT/name) for name in ('experiments/candidate_split_verifier.py',
            'experiments/test_candidate_split_verifier.py','experiments/event_input_smoothing.py',
            'experiments/partial_phase_events.py','uterine_wave/radial_events.py')},
        reference_sha256={str(p):digest(p) for p in (args.noise_reference,args.source/'protocol.json')},
        parents='Exact cached periodic+partial events; no union, re-detection, new parent or scope expansion',
        auxiliary='Parent-buffered native sections only, existing noise floor, no global auxiliary detector',
        identity='Two positive-to-negative complete phase sequences with adjacent shared support and fixed temporal rank',
        propagation='Unchanged ridge, phase-edge, pair measurement and restricted pair assignment functions',
        formal_features_changed=False,event_count=None,event_frequency=None))
    print('SUMMARY_SAVED',len(summaries),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--controls-root',type=Path,required=True)
    p.add_argument('--noise-reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
