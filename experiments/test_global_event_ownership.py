"""Fixed 87-case ownership validation; truth is evaluation-only."""
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
from experiments.global_event_ownership import reconcile


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def truth_groups(report,proposals,truth):
    before=evaluate(report,truth)
    groups={'base:'+g['group_id']:g for g in before['groups']}
    for p in proposals:
        if p['split_status']!='split_two':continue
        for child in (p['child_1'],p['child_2']):
            ridges={rid:dict(sign=child['supporting_phase_evidence'][0]['phases'][i]['sign'],
                sections=child['sections'],peak_time_s=[r['phases'][i]['peak_s'] for r in child['supporting_phase_evidence']])
                for i,rid in enumerate(child['phase_ridge_ids'])}
            evidence=describe_piece(child,ridges,truth)
            groups['child:'+child['event_id']]=dict(group_id=child['event_id'],
                assignments=evidence['assignments'],classification=child['classification'])
    return before,groups


def assess(report,proposals,result,truth,local_evaluation,negative):
    statuses=Counter(e['ownership_status'] for e in result['event_proposals'])
    out=dict(final_proposal_rows=len(result['event_proposals']),status_counts=dict(statuses),
        actions=dict(Counter(a['status'] for a in result['split_actions'])),
        ambiguous_ownership=len(result['ambiguous_ownership']),
        local_true_split_wall=any(d['supports_two_distinct_raw_truth_events'] for d in local_evaluation['split_details']),
        accepted_true_two_split_wall=False,local_duplicate_truths=None,global_duplicate_truths=None,
        original_duplicate_truths=None,before_matches=None,after_matches=None,
        negative_new_identities=sum(e['canonical_source_entity_id'].startswith('child:') for e in result['event_proposals']) if negative else None,
        negative_new_directional_identities=sum(e['canonical_source_entity_id'].startswith('child:') and e['direction'] in ('C2F','F2C')
                                               for e in result['event_proposals']) if negative else None)
    if truth is None:return out
    before,mapping=truth_groups(report,proposals,truth)
    final_groups={e['event_proposal_id']:dict(mapping[e['canonical_source_entity_id']],group_id=e['event_proposal_id'])
                  for e in result['event_proposals']}
    number=truth['uniform_observable_count'];n=len(truth['columns'])
    tolerance=result['matching_rules']['timing_tolerance_s']
    def timely(g,identity):
        ids=sorted({p['section'] for p in g['assignments'].get(identity,[]) if abs(p['error_s'])<=tolerance})
        blocks=np.split(np.asarray(ids,int),np.flatnonzero(np.diff(ids)!=1)+1)
        return max((len(b) for b in blocks),default=0)>=3
    # Reconstruct the previous LOCAL replacement for a fair duplicate test.
    # Merely belonging to a long raw phase is not enough to call two candidate
    # markers the same event: use the already fixed timing scale as well.
    splitting={p['parent_candidate_id'] for p in proposals if p['split_status']=='split_two'}
    base_by_id={g['group_id']:g for g in before['groups']}
    descriptions={p['fragment_id']:p for p in before['fragments']}
    local_groups=[]
    for group in report['proposed_events']:
        points={}
        for fid in group['fragment_ids']:
            if fid in splitting:continue
            for i,values in descriptions[fid]['assignments'].items():points.setdefault(i,[]).extend(values)
        if points:local_groups.append(dict(assignments=points))
    local_groups.extend(g for k,g in mapping.items() if k.startswith('child:'))
    def timely_excess(groups):
        return sum(max(0,sum(timely(g,i) for g in groups)-1) for i in range(number))
    after=one_to_one(list(final_groups.values()),number,n)
    original_counts=before['multiple_group_support']
    final_counts=[sum(bool(g['assignments'].get(i)) for g in final_groups.values()) for i in range(number)]
    accepted_details=[]
    for action in result['split_actions']:
        if action['status']=='ambiguous':continue
        ids=action['final_event_ids']
        if len(ids)!=2 or None in ids or ids[0]==ids[1]:
            raise RuntimeError('An accepted split must reference two distinct final identities')
        groups=[final_groups[eid] for eid in ids]
        matches=one_to_one(groups,number,n)
        good=(number==2 and all(m['matched'] for m in matches) and
              all(len(g['assignments'])==1 for g in groups) and
              all(len({p['section'] for p in m.get('measurements',[])})>=3 for m in matches))
        accepted_details.append(dict(parent_id=action['parent_id'],true_two=good,matches=matches))
    local_success={d['parent_candidate_id'] for d in local_evaluation['split_details'] if d['supports_two_distinct_raw_truth_events']}
    actions={a['parent_id']:a for a in result['split_actions']}
    unresolved_local=any(actions[p]['status']=='ambiguous' and all(c in result['source_evidence'] for c in actions[p]['children'])
                         for p in local_success)
    false_collapses=sum(a['status']!='ambiguous' and len(set(a['final_event_ids']))!=2 for a in result['split_actions'])
    out.update(before_matches=before['matches'],after_matches=after,
        local_duplicate_truths=sum(x>1 for x in local_evaluation.get('after_supported_group_counts',[])),
        global_duplicate_truths=sum(x>1 for x in final_counts),original_duplicate_truths=sum(x>1 for x in original_counts),
        local_excess_representations=sum(max(0,x-1) for x in local_evaluation.get('after_supported_group_counts',[])),
        global_excess_representations=sum(max(0,x-1) for x in final_counts),
        original_excess_representations=sum(max(0,x-1) for x in original_counts),
        accepted_true_two_split_wall=any(d['true_two'] for d in accepted_details),
        unresolved_local_true_pair_preserved=unresolved_local,
        false_collapses=false_collapses,
        original_timely_excess=timely_excess(before['groups']),
        local_timely_excess=timely_excess(local_groups),global_timely_excess=timely_excess(list(final_groups.values())),
        false_added_split_transactions=sum(not d['true_two'] for d in accepted_details) if number==1 else None,
        accepted_details=accepted_details)
    return out


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    source_protocol=read(args.splits/'protocol.json')
    for name in ('experiments/candidate_split_verifier.py','experiments/partial_phase_events.py','uterine_wave/radial_events.py'):
        if digest(ROOT/name)!=source_protocol['code_sha256'][name]:
            raise ValueError('Frozen split or parent implementation differs from baseline')
    paths=sorted(args.splits.glob('texture*.json'))
    if len(paths)!=87:raise ValueError('Expected exactly the frozen 87 conditions')
    sources={};cache={};rows=[]
    for path in paths:
        sources[str(path)]=digest(path);local=read(path)
        parent_path=args.parents/(local['source']+'.json');sources[str(parent_path)]=digest(parent_path)
        if sources[str(parent_path)]!=source_protocol['source_files'][str(parent_path)]:
            raise ValueError('Frozen parent evidence changed')
        old=read(parent_path);texture=old['texture'];family=old['family']
        control=args.controls_root/(old['source']+'.npz' if family=='image_control' else texture+'_rigid.npz')
        if str(control) not in cache:
            if digest(control)!=source_protocol['source_files'][str(control)]:raise ValueError('Frozen cache changed')
            sources[str(control)]=digest(control);cache[str(control)]=control_arrays(control)
        noise,valid,t,s,span=cache[str(control)];truth=None
        if family=='double':
            clean,_=double_fixture(t,s,span,old['expected_direction'],old['gap_s'],old['second_amplitude']);truth=raw_truth(clean,t)
        elif family=='single_or_negative' and old['expected_direction']:
            kind,amplitude=old['source'][len(texture+'_single_'):].rsplit('_',1)
            clean,_,_=radial_fixture(t,s,span,kind,float(amplitude));truth=raw_truth(clean,t)
        row=dict(source=old['source'],family=family,gap_s=old['gap_s'],second_amplitude=old['second_amplitude'],
            expected_direction=old['expected_direction'],known_nonpropagating=old['known_nonpropagating'],walls=[])
        for wall in local['walls']:
            w=wall['wall'];report=old['methods']['periodic_frozen']['walls'][w]['report']
            proposals=wall['versions']['3']['proposals']
            result=reconcile(report,proposals,t,valid[:,w],measurement_span_s=span)
            if result['event_count'] is not None or result['event_frequency'] is not None:
                raise RuntimeError('Formal counts must remain null')
            assessment=assess(report,proposals,result,truth,wall['versions']['3']['evaluation'],old['known_nonpropagating'])
            row['walls'].append(dict(wall=w,reconciliation=result,evaluation=assessment))
        save(args.output/(old['source']+'.json'),row)
        rows.append(dict(row,walls=[dict(wall=w['wall'],evaluation=w['evaluation']) for w in row['walls']]))
        print(old['source'],flush=True)
    def aggregate(selected):
        items=[w['evaluation'] for r in selected for w in r['walls']]
        known=[x for x in items if x['after_matches'] is not None]
        def support(which,i):
            return min((x[which][i]['native_section_fraction'] for x in known if len(x[which])>i),default=None)
        return dict(wall_conditions=len(items),
            local_true_two_split_walls=sum(x['local_true_split_wall'] for x in items),
            global_true_two_split_walls=sum(x['accepted_true_two_split_wall'] for x in items),
            unresolved_local_true_pair_walls=sum(x.get('unresolved_local_true_pair_preserved',False) and not x['accepted_true_two_split_wall'] for x in items),
            false_collapses=sum(x.get('false_collapses',0) for x in items),
            original_timely_excess=sum(x.get('original_timely_excess',0) for x in items),
            local_timely_excess=sum(x.get('local_timely_excess',0) for x in items),
            global_timely_excess=sum(x.get('global_timely_excess',0) for x in items),
            local_duplicate_truths=sum(x['local_duplicate_truths'] or 0 for x in known),
            global_duplicate_truths=sum(x['global_duplicate_truths'] or 0 for x in known),
            original_duplicate_truths=sum(x['original_duplicate_truths'] or 0 for x in known),
            local_excess_representations=sum(x['local_excess_representations'] for x in known),
            global_excess_representations=sum(x['global_excess_representations'] for x in known),
            original_excess_representations=sum(x['original_excess_representations'] for x in known),
            before_minimum_support_by_identity={str(i):support('before_matches',i) for i in (0,1)},
            after_minimum_support_by_identity={str(i):support('after_matches',i) for i in (0,1)},
            after_assigned_by_identity={str(i):sum(len(x['after_matches'])>i and x['after_matches'][i]['matched'] for x in known) for i in (0,1)},
            after_arrival_RMSE_median_by_identity={str(i):float(np.median([x['after_matches'][i]['arrival_RMSE_s'] for x in known
                if len(x['after_matches'])>i and x['after_matches'][i]['matched']]))
                if any(len(x['after_matches'])>i and x['after_matches'][i]['matched'] for x in known) else None for i in (0,1)},
            negative_new_identities=sum(x['negative_new_identities'] or 0 for x in items),
            negative_new_directional_identities=sum(x['negative_new_directional_identities'] or 0 for x in items),
            false_added_single_splits=sum(x.get('false_added_split_transactions') or 0 for x in items),
            action_counts=dict(sum((Counter(x['actions']) for x in items),Counter())),
            ambiguous_ownership=sum(x['ambiguous_ownership'] for x in items))
    summary=dict(double_conditions=[dict(gap_s=g,second_amplitude=a,metrics=aggregate([
        r for r in rows if r['family']=='double' and r['gap_s']==g and r['second_amplitude']==a]))
        for g in (5.,10.,15.,20.) for a in (.04,.01)],
        single_levels=[dict(amplitude=a,metrics=aggregate([r for r in rows if r['family']=='single_or_negative'
            and r['expected_direction'] and r['source'].endswith(f'{a:.2f}')])) for a in (.04,.02,.01)],
        signal_negatives=aggregate([r for r in rows if r['family']=='single_or_negative' and r['known_nonpropagating']]),
        image_controls=aggregate([r for r in rows if r['family']=='image_control']),event_count=None,event_frequency=None)
    by={(r['gap_s'],r['second_amplitude']):r['metrics'] for r in summary['double_conditions']}
    ten=by[10.,.04];fifteen=by[15.,.04];weak=by[20.,.01]
    checks=dict(ten_true_pairs_not_collapsed_or_discarded=ten['false_collapses']==0 and
            ten['global_true_two_split_walls']+ten['unresolved_local_true_pair_walls']>=ten['local_true_two_split_walls'] and
            ten['global_true_two_split_walls']>0,
        fifteen_duplicates_reduced=fifteen['global_timely_excess']<fifteen['local_timely_excess'] and
            fifteen['global_timely_excess']<=fifteen['original_timely_excess'],
        weak_second_support_preserved=weak['after_minimum_support_by_identity']['1']>=weak['before_minimum_support_by_identity']['1'],
        no_added_single_splits=all(r['metrics']['false_added_single_splits']==0 for r in summary['single_levels']),
        no_added_negative_direction=sum(summary[k]['negative_new_directional_identities'] for k in ('signal_negatives','image_controls'))==0)
    summary.update(acceptance_checks=checks,passed=all(checks.values()),real_cases_run=False,
        strict_confirmation_nonregression=ten['global_true_two_split_walls']>=ten['local_true_two_split_walls'],
        interpretation='Proposal-review acceptance allows preserved ambiguity; strict confirmed recovery is reported separately')
    if any(digest(Path(p))!=sha for p,sha in sources.items()):raise RuntimeError('Frozen inputs changed')
    save(args.output/'summary.json',summary);save(args.output/'all_conditions.json',rows)
    save(args.output/'protocol.json',dict(source_files=sources,baseline_commit='bcef4d290ba35cd32179b432ca099227d9bd1c4c',
        code_sha256={p:digest(ROOT/p) for p in ('experiments/global_event_ownership.py','experiments/test_global_event_ownership.py')},
        matching='Exact phase provenance first, then complete local phase/arrival structure, then unique partial ownership',
        tolerance='Existing measurement span + one native sample; phase-interval membership also required',
        no_parameter_sweep=True,split_support=3,formal_features_unchanged=True,event_count=None,event_frequency=None))
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',type=Path,required=True)
    p.add_argument('--parents',type=Path,required=True)
    p.add_argument('--controls-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
