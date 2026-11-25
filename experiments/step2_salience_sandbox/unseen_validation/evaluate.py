"""Locked evaluation fields; never used for source selection or model tuning."""
from collections import Counter
from ..mechanism_2x2.metrics import outcome,phase_diagnostics,aggregate
from ..diagnostic.provenance import phase_kind

MERGE_DEFINITION=(
    'Per final propagation prediction in a two-truth coexist sample: evidence bounds contain BOTH physical truth centers, '
    'prediction shares >=3 native sections with each truth, and the frozen maximum-cardinality matcher has FN>=1. '
    'Count qualifying predictions, not wall nodes. Uses existing evidence bounds, section overlap and matcher identities; '
    'no time-radius/fit/amplitude threshold is added. This is a conservative absorption pattern, not proof of causation. '
    'Cross-truth phase pairing and unassigned phase labels remain separate existing diagnostics.')

def single_phase_flow(record,evaluation,row):
    shared=record['shared'];formed=set();single_canonical=set()
    detector=sum(e['evidence_kind']=='single_phase' for w in shared['walls'] for e in w['parent_report']['events'])
    for wall in shared['walls']:
        for e in wall['reconciliation']['event_proposals']:
            source=wall['reconciliation']['source_evidence'][e['canonical_source_entity_id']]
            if phase_kind(source)=='single_phase':
                formed.add((wall['wall'],e['event_proposal_id']));single_canonical.add((wall['wall'],e['canonical_source_entity_id']))
    selected={f"wall{wall}/{proposal}" for wall,proposal in formed}
    nodes={n['node_id'] for n in evaluation['nodes']}
    prop_nodes={e['node_id'] for a in evaluation['result']['activities'] for e in a['block_evidence'] if e['propagation_supported']}
    unmatched=set(row['match']['unmatched_prediction_ids'])
    final=[p for p in row['predictions'] if p['phase_origin']=='single_phase']
    mixed=[p for p in row['predictions'] if p['phase_origin']=='mixed']
    return dict(detector_single_phase_events=detector,formed=len(formed),passed_MinMax_06=len(selected&nodes),
        propagation_supported=len(selected&prop_nodes),final_propagation_activities=len(final),
        final_unmatched=sum(p['prediction_id'] in unmatched for p in final),
        final_mixed_propagation_activities=len(mixed),final_mixed_unmatched=sum(p['prediction_id'] in unmatched for p in mixed),
        single_phase_node_members_in_final_unmatched=sum(sum(tuple(key) in single_canonical for key in p['stable_identity'])
            for p in row['predictions'] if p['prediction_id'] in unmatched),
        truth_containing_final_unmatched=sum(p['prediction_id'] in unmatched for p in final) if record['truth'] else 0,
        truth_bearing=bool(record['truth']),case_id=record['case_id'],texture_id=record['source']['texture_id'])

def merge_errors(record,row):
    if record['scenario']['family']!='coexist':return []
    truth=record['truth'];assert len(truth)==2
    if row['FN']==0:return []
    errors=[]
    for p in row['predictions']:
        if all(p['evidence_start_s']<=t['truth_center_s']<=p['evidence_end_s'] and
               len(set(p['sections'])&set(t['truth_sections']))>=3 for t in truth):
            errors.append(dict(case_id=record['case_id'],prediction_id=p['prediction_id'],stable_identity=p['stable_identity'],
                evidence_start_s=p['evidence_start_s'],evidence_end_s=p['evidence_end_s'],
                truth_ids=[t['truth_event_id'] for t in truth],matcher_unmatched_truth_ids=row['match']['unmatched_truth_ids']))
    return errors

def evaluate_saved(payload,model):
    record=payload['record'];ev=payload['evaluation']
    row=outcome(record,ev['result'],ev['nodes'],'S1')
    diagnostics=phase_diagnostics(record,model,payload['pairing_logs'])
    return dict(row=row,phase=diagnostics,flow=single_phase_flow(record,ev,row),merge_errors=merge_errors(record,row))

def coexist_summary(results):
    output={}
    for model,values in results.items():
        output[model]={}
        for key,ratio in [('pooled',None),('2',2),('3',3),('5',5)]:
            selected=[v for v in values if v['row']['scenario']['family']=='coexist' and (ratio is None or v['row']['scenario']['ratio']==ratio)]
            summary=aggregate([v['row'] for v in selected])
            cross=[x for v in selected for x in v['phase']['cross_truth']]
            merge=[x for v in selected for x in v['merge_errors']]
            summary.update(cross_truth_pairing_error=sum(x['cross_truth_pairing_error'] for x in cross),
                possible_cross_truth_pairing=sum(x['possible_cross_truth'] for x in cross),
                unassigned_phase_pairs=sum(x['unassigned_phase'] for x in cross),merge_two_truth_error=len(merge),merge_records=merge)
            output[model][key]=summary
    return output

def flow_summary(values):
    fields=('detector_single_phase_events','formed','passed_MinMax_06','propagation_supported','final_propagation_activities',
            'final_unmatched','truth_containing_final_unmatched','final_mixed_propagation_activities','final_mixed_unmatched',
            'single_phase_node_members_in_final_unmatched')
    return {label:{field:sum(v['flow'][field] for v in values if label=='all_raw_samples' or v['flow']['truth_bearing']) for field in fields}
            for label in ('all_raw_samples','truth_bearing_samples')}

def lost_gained(results):
    before={(v['row']['case_id'],t['truth_id']):t for v in results['M0'] for t in v['row']['truth_rows']}
    after={(v['row']['case_id'],t['truth_id']):t for v in results['M1'] for t in v['row']['truth_rows']}
    assert before.keys()==after.keys()
    output={label:dict(lost=[],gained=[],denominator=sum(t['amplitude_class']==label for t in before.values()))
            for label in ('clear','medium','structured_weak')}
    for key,a in before.items():
        b=after[key];label=a['amplitude_class']
        if a['matched'] and not b['matched']:output[label]['lost'].append(dict(case_id=key[0],truth_id=key[1]))
        if not a['matched'] and b['matched']:output[label]['gained'].append(dict(case_id=key[0],truth_id=key[1]))
    return output
