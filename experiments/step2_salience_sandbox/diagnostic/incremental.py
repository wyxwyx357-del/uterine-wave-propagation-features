"""Compare immutable canonical-node sets, with composite changes explicit."""
from collections import Counter
from .attribution_rules import overlap

def tags(rows):
    return {field:dict(Counter(p[field] for p in rows)) for field in ('primary_class','phase_origin','split_origin','wall_status')}

def compare_case(record,all_by_point,name):
    base=all_by_point['baseline'];mad=all_by_point[name]
    bp={p['stable_source_identity']:p for p in base if p['propagation_supported']}
    mp={p['stable_source_identity']:p for p in mad if p['propagation_supported']}
    bu={k:p for k,p in bp.items() if not p['matched_TP']};mu={k:p for k,p in mp.items() if not p['matched_TP']}
    all_base={p['stable_source_identity']:p for p in base}
    selected_base={tuple(k) for p in base for k in p['source_node_keys']}
    selected_mad={tuple(k) for p in mad for k in p['source_node_keys']}
    new=[]
    for key,p in mu.items():
        if key in bu:continue
        keys={tuple(k) for k in p['source_node_keys']}
        supporting=[]
        for q in base:
            common=keys&{tuple(k) for k in q['source_node_keys']}
            if common:
                seconds,iou=overlap(p['evidence_start_s'],p['evidence_end_s'],q['evidence_start_s'],q['evidence_end_s'])
                supporting.append(dict(baseline_identity=q['stable_source_identity'],baseline_propagates=q['propagation_supported'],
                    baseline_matched=q['matched_TP'],node_key_overlap=sorted(common),time_overlap_seconds=seconds,
                    time_IoU=iou,section_overlap=sorted(set(p['sections'])&set(q['sections']))))
        broken=[q for q in base if keys < {tuple(k) for k in q['source_node_keys']}]
        details=[]
        for q in broken:
            ids=set(q['source_nodes']);ev=record['points'][name]['evaluation']
            baseline_ev=record['points']['baseline']['evaluation']
            direct=[e for e in ev['result']['identity_comparisons'] if {e['node_a'],e['node_b']}==ids]
            neighbors=[e for e in ev['result']['identity_comparisons'] if len({e['node_a'],e['node_b']}&ids)==1 and e['relation'] in ('shared_identity_supported','unresolved')]
            changed=[n['node_id'] for n in ev['nodes'] if n['node_id'] in ids and any(b['node_id']==n['node_id'] and b['sections']!=n['sections'] for b in baseline_ev['nodes'])]
            baseline_ids={n['node_id'] for n in baseline_ev['nodes']}
            new_neighbors=[e for e in neighbors if not {e['node_a'],e['node_b']}<=baseline_ids]
            details.append(dict(baseline_composite=q['stable_source_identity'],direct_pair_relation=direct,
                changed_retained_section_nodes=changed,blocking_neighbors=neighbors,new_blocking_neighbors=new_neighbors,
                reason='new_neighbors_block_strict_pair' if direct and direct[0]['relation']=='shared_identity_supported' and new_neighbors else
                       'retained_support_changed_direct_identity' if direct and direct[0]['relation']!='shared_identity_supported' and changed else
                       'member_not_selected' if not direct else 'existing_neighbors_or_grouping_context'))
        if key in bp:kind='same_propagation_prediction_assignment_changed'
        elif key in all_base:kind='existing_selected_activity_became_propagation_supported'
        elif broken:kind='baseline_composite_broken'
        elif keys.isdisjoint(selected_base):kind='newly_admitted_shared_upstream_candidate'
        else:kind='canonical_nodes_regrouped'
        new.append(dict(event_key=p['event_key'],case_id=record['case_id'],point=name,stable_source_identity=key,
            primary_class=p['primary_class'],phase_origin=p['phase_origin'],split_origin=p['split_origin'],wall_status=p['wall_status'],
            source_type=kind,baseline_correspondences=supporting,merge_break_diagnostics=details,
            entirely_new_selected_node_keys=sorted(keys-selected_base),all_present_in_frozen_upstream=True))
    return dict(case_id=record['case_id'],point=name,all_propagation_sets=dict(baseline_only=len(bp.keys()-mp.keys()),
        MAD_only=len(mp.keys()-bp.keys()),both=len(bp.keys()&mp.keys())),
        unmatched_sets=dict(baseline_only=len(bu.keys()-mu.keys()),MAD_only=len(mu.keys()-bu.keys()),both=len(bu.keys()&mu.keys())),
        unmatched_net_change=len(mu)-len(bu),new_unmatched=new,
        removed_unmatched=[dict(stable_source_identity=k,now_matched=k in mp and mp[k]['matched_TP'],
            still_selected_nonpropagation=k not in mp and any(q['stable_source_identity']==k for q in mad)) for k in bu.keys()-mu.keys()],
        source_type_counts=dict(Counter(p['source_type'] for p in new)),new_tags=tags(new))

def summarize(cases):
    new=[p for c in cases for p in c['new_unmatched']]
    return dict(cases=cases,all_propagation_sets={k:sum(c['all_propagation_sets'][k] for c in cases) for k in ('baseline_only','MAD_only','both')},
        unmatched_sets={k:sum(c['unmatched_sets'][k] for c in cases) for k in ('baseline_only','MAD_only','both')},
        unmatched_net_change=sum(c['unmatched_net_change'] for c in cases),new_tags=tags(new),
        source_type_counts=dict(Counter(p['source_type'] for p in new)),
        merge_break_reason_counts=dict(Counter(d['reason'] for p in new for d in p['merge_break_diagnostics'])),
        interpretation='MAD-only UNMATCHED set includes matched-to-unmatched assignment changes; these are separated from new propagation detections. Composite identity uses canonical node sets, never activity IDs.')
