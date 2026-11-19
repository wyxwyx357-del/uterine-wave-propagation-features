"""Sandbox: preserve candidate provenance and conditional cross-wall ranges.

No new detection, phase pairing, direction gate or numerical threshold. A
paired-phase candidate is still not a confirmed physiological event. The
conditional range covers only recorded cross-wall matching alternatives;
missing shared support cannot be turned into a match or a global lower bound.
"""
from copy import deepcopy

from experiments.activity_counting import count_activities
from experiments.patient_activity_aggregation import candidate_nodes


def provenance_nodes(salience, ownership):
    nodes = candidate_nodes(salience, ownership)
    for node in nodes:
        rec = ownership['walls'][node['wall']]['reconciliation']
        event = next(e for e in rec['event_proposals']
                     if e['event_proposal_id'] == node['source_proposal_id'])
        entity = rec['source_evidence'][event['canonical_source_entity_id']]
        node.update(
            evidence_kind=entity['canonical_measurement'].get('evidence_kind', 'unknown'),
            provenance=dict(
                canonical_source=event['canonical_source_entity_id'],
                source_kind=entity['source_kind'],
                source_entities=event['source_entity_ids'],
                source_parents=event['source_parent_ids'],
                phase_ridges=deepcopy(event['phase_ridge_provenance']),
                source_evidence_kinds=[m.get('evidence_kind', 'unknown')
                                       for m in event['preserved_original_measurements']]))
    return nodes


def conditional_range(result, nodes, seconds):
    """Reuse the historical maximum bipartite matching conditional range.

    Definite unique pairs have already merged in count_activities. Remaining
    recorded alternatives are hypothetical merges, not accepted identities.
    A one-to-many edge cannot make the lower endpoint subtract every edge.
    """
    lookup = {n['node_id']: n for n in nodes}
    counted = [a for a in result['activities'] if a['propagation_supported']]
    singletons = {a['source_nodes'][0] for a in counted if len(a['source_nodes']) == 1}
    graph = {n: [] for n in singletons if lookup[n]['wall'] == 0}
    alternatives, unassessed = [], []
    for edge in result['identity_comparisons']:
        a, b = edge['node_a'], edge['node_b']
        if a not in singletons or b not in singletons:
            continue
        if edge['relation'] in ('shared_identity_supported', 'unresolved'):
            if lookup[a]['wall'] == 1:
                a, b = b, a
            graph[a].append(b)
            alternatives.append(edge)
        elif not edge['shared_sections']:
            # Footprint overlap is descriptive only: no new identity edge.
            left = [p for ps in lookup[a]['profile'].values() for p in ps]
            right = [p for ps in lookup[b]['profile'].values() for p in ps]
            overlap = max(min(p['start_s'] for p in left), min(p['start_s'] for p in right)) <= min(
                max(p['end_s'] for p in left), max(p['end_s'] for p in right))
            unassessed.append(dict(node_a=a, node_b=b, temporal_footprints_overlap=overlap,
                reason='no shared native sections; identity not established either way'))
    owner = {}

    def augment(a, seen):
        for b in sorted(graph[a]):
            if b in seen:
                continue
            seen.add(b)
            if b not in owner or augment(owner[b], seen):
                owner[b] = a
                return True
        return False

    possible = sum(augment(a, set()) for a in sorted(graph))
    high = len(counted)
    low = high - possible
    return dict(
        conditional_candidate_identity_count_range=[low, high],
        conditional_candidate_rate_range_cpm=([low*60/seconds, high*60/seconds]
                                              if seconds > 0 else [None, None]),
        hypothetical_cross_wall_matches=[dict(node_a=a, node_b=b) for b, a in sorted(owner.items())],
        recorded_identity_alternatives=alternatives,
        no_shared_section_pairs=unassessed,
        identity_remains_unresolved=bool(alternatives or any(
            p['temporal_footprints_overlap'] for p in unassessed)),
        range_is_global_identity_bound=False,
        range_semantics=('conditional on recorded cross-wall alternatives only; not a confidence '
                         'interval, true-wave bound or permission to merge; complementary '
                         'sections and same-wall ambiguity are not resolved'))


def trial_count(nodes, s, span, dt, seconds):
    baseline = count_activities(nodes, s, span, dt, seconds)
    eligible = [n for n in nodes if n['evidence_kind'] == 'paired_phase']
    deferred = [deepcopy(n) for n in nodes if n['evidence_kind'] != 'paired_phase']
    primary = count_activities(eligible, s, span, dt, seconds)
    kinds = {n['node_id']: n['evidence_kind'] for n in nodes}
    for activity in primary['activities']:
        activity['source_evidence_kinds'] = [kinds[n] for n in activity['source_nodes']]
        activity['count_status'] = 'primary_candidate' if activity['propagation_supported'] else 'other_evidence'
    # Keep single-phase evidence intact, including its original classifier QC.
    for node in deferred:
        node['count_status'] = 'unresolved_phase_identity'
        node['old_propagation_supported'] = any(
            e['node_id'] == node['node_id'] and e['propagation_supported']
            for a in baseline['activities'] for e in a['block_evidence'])
    return dict(
        baseline=baseline, primary=primary,
        primary_candidate_row_count=primary['significant_propagation_activity_count'],
        deferred_evidence=deferred,
        provenance_nodes=deepcopy(nodes),
        identity_range=conditional_range(primary, eligible, seconds),
        event_count=None, event_frequency=None,
        formal_results_changed=False, new_thresholds=False,
        interpretation='paired-phase significant propagation candidate density; not true wave count')

