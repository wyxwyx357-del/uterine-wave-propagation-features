"""Step-1 v2 count implementation.

This preserves the old strict-pair/single activity ledger for auditability,
then adds a transparent propagation-count range over unresolved/one-to-many
cross-wall relationships. No threshold, filter, propagation classifier, or
physiological definition is changed.
"""
from itertools import combinations
import numpy as np
from analysis.mse_pregnancy_association.code.direction_identity_trial import compare_identity_without_direction
from analysis.mse_pregnancy_association.code.measurement.radial_events import _describe_arrival, _arrival_evidence, RadialWaveConfig


def retained_phase_fits(node,block,s,dt):
    """Fit signed phase peaks only where each section has one unambiguous peak."""
    fits=[];availability=[]
    for sign in (1,-1):
        peaks=[[p['peak_s'] for p in node['profile'][int(c)] if p['sign']==sign]
               for c in block]
        complete=all(len(values)==1 for values in peaks)
        availability.append(dict(sign=sign,complete=complete,
                                 peak_counts=[len(values) for values in peaks]))
        if complete:
            fits.append(dict(_arrival_evidence(np.asarray(s)[block],
                             np.array([values[0] for values in peaks]),dt),sign=sign))
    return fits,availability


def _node_evidence(node, s, dt):
    evidence=[]
    ids=np.asarray(sorted(node['sections']),int)
    for block in np.split(ids,np.flatnonzero(np.diff(ids)!=1)+1):
        if len(block)<3:
            continue
        arrival=np.array([node['arrival'][int(c)] for c in block])
        phase_fits,availability=retained_phase_fits(node,block,s,dt)
        measurement=_describe_arrival(block,arrival,np.asarray(s),dt,phase_fits,RadialWaveConfig())
        evidence.append(dict(node_id=node['node_id'],sections=block.tolist(),measurement=measurement,
            phase_fit_availability=availability,
            propagation_supported=measurement['classification'] in ('C2F','F2C')))
    return evidence


def _maximum_bipartite_matching_count(left_nodes, adjacency):
    """Deterministic maximum cardinality matching; returns number of pairs."""
    owner={}
    def augment(left, seen):
        for right in sorted(adjacency.get(left, ())):
            if right in seen:
                continue
            seen.add(right)
            if right not in owner or augment(owner[right], seen):
                owner[right]=left
                return True
        return False
    return sum(augment(left,set()) for left in sorted(left_nodes))


def count_activities(nodes,s,span,dt,observation_seconds):
    edges=[compare_identity_without_direction(a,b,span,dt)
           for a,b in combinations(nodes,2) if a['wall']!=b['wall']]
    choices={n['node_id']:[] for n in nodes};uncertain={n['node_id']:[] for n in nodes}
    for edge in edges:
        target=choices if edge['relation']=='shared_identity_supported' else uncertain if edge['relation']=='unresolved' else None
        if target is not None:
            a,b=edge['node_a'],edge['node_b'];target[a].append(b);target[b].append(a)

    # Preserve v1 strict one-to-one/no-uncertainty grouping exactly.
    pairs={tuple(sorted((a,bs[0]))) for a,bs in choices.items() if len(bs)==1 and choices[bs[0]]==[a]
           and not uncertain[a] and not uncertain[bs[0]]}
    lookup={n['node_id']:n for n in nodes};used=set();groups=[]
    for a,b in sorted(pairs):
        groups.append([lookup[a],lookup[b]]);used.update((a,b))
    groups.extend([[n] for n in nodes if n['node_id'] not in used])

    # Compute propagation support once per original wall node.  This lets the
    # uncertainty graph exclude non-propagation nodes, preventing them from
    # consuming a matching slot that matters to the propagation count bound.
    node_evidence={node['node_id']:_node_evidence(node,s,dt) for node in nodes}
    node_propagates={node_id:any(row['propagation_supported'] for row in evidence)
                     for node_id,evidence in node_evidence.items()}

    activities=[]
    for i,group in enumerate(groups,1):
        evidence=[row for node in group for row in node_evidence[node['node_id']]]
        members=[n['node_id'] for n in group]
        phases=[p for n in group for ps in n['profile'].values() for p in ps]
        activities.append(dict(activity_id=f'activity_{i:03d}',source_nodes=members,
            walls=sorted({n['wall'] for n in group}),
            propagation_supported=any(node_propagates[node_id] for node_id in members),
            block_evidence=evidence,direction_used_for_counting=False,
            direction_conflict_QC=len({n['direction'] for n in group if n['direction']})>1,
            unresolved_related_nodes=sorted({k for a in members for k in choices[a]+uncertain[a] if k not in members}),
            evidence_start_s=min(p['start_s'] for p in phases),evidence_end_s=max(p['end_s'] for p in phases)))

    # ----- Legacy v1 count: strict pairs + every unmerged propagation node -----
    legacy_count=sum(a['propagation_supported'] for a in activities)

    # Strict merged pair contributes one propagation activity if either member
    # has supported propagation under the unchanged classifier.
    confirmed_pair_members={member for pair in pairs for member in pair}
    confirmed_propagation_pairs=sum(
        any(node_propagates[node_id] for node_id in pair) for pair in pairs
    )

    # Only unmerged propagation-supported nodes enter the ambiguity graph.
    unmerged_prop={node_id for node_id,prop in node_propagates.items()
                   if prop and node_id not in confirmed_pair_members}

    relevant_relations={'shared_identity_supported','unresolved'}
    adjacency={node_id:[] for node_id in unmerged_prop if lookup[node_id]['wall']==0}
    ambiguous_nodes=set()
    for edge in edges:
        if edge['relation'] not in relevant_relations:
            continue
        a,b=edge['node_a'],edge['node_b']
        if a not in unmerged_prop or b not in unmerged_prop:
            continue
        if lookup[a]['wall']==1:
            a,b=b,a
        adjacency.setdefault(a,[]).append(b)
        ambiguous_nodes.update((a,b))

    max_matching=_maximum_bipartite_matching_count(adjacency.keys(),adjacency)
    unmerged_prop_count=len(unmerged_prop)
    min_count=int(confirmed_propagation_pairs + unmerged_prop_count - max_matching)
    max_count=int(confirmed_propagation_pairs + unmerged_prop_count)
    clean_unmerged=sum(node_id not in ambiguous_nodes for node_id in unmerged_prop)
    resolved_count=int(confirmed_propagation_pairs + clean_unmerged)

    if max_count != legacy_count:
        raise RuntimeError('v2 upper count must equal the frozen v1 strict-pair/single count')

    if observation_seconds>0:
        frequency_range=[min_count*60/observation_seconds,max_count*60/observation_seconds]
        legacy_frequency=legacy_count*60/observation_seconds
        exact_frequency=(frequency_range[0] if min_count==max_count else None)
    else:
        frequency_range=[None,None];legacy_frequency=None;exact_frequency=None

    return dict(
        activities=activities,
        identity_comparisons=edges,
        # Historical fields retained unchanged for backward audit compatibility.
        significant_propagation_activity_count=legacy_count,
        significant_propagation_activity_frequency_cpm=legacy_frequency,
        significant_propagation_activity_count_v1_legacy=legacy_count,
        legacy_significant_propagation_activity_count=legacy_count,
        significant_propagation_activity_frequency_cpm_v1_legacy=legacy_frequency,
        # New v2 transparent identity-uncertainty outputs.
        significant_propagation_activity_count_range=[min_count,max_count],
        significant_propagation_activity_count_min=min_count,
        significant_propagation_activity_count_max=max_count,
        significant_propagation_activity_count_v2_exact=(min_count if min_count==max_count else None),
        significant_propagation_activity_count_exact=(min_count if min_count==max_count else None),
        significant_propagation_activity_frequency_cpm_range=frequency_range,
        significant_propagation_activity_frequency_cpm_min=frequency_range[0],
        significant_propagation_activity_frequency_cpm_max=frequency_range[1],
        significant_propagation_activity_frequency_cpm_v2_exact=exact_frequency,
        significant_propagation_activity_frequency_cpm_exact=exact_frequency,
        count_range_semantics="identity ambiguity range, not statistical confidence interval",
        resolved_propagation_activity_count=resolved_count,
        ambiguous_cross_wall_propagation_pairs=int(max_matching),
        ambiguous_cross_wall_propagation_node_count=int(len(ambiguous_nodes)),
        ambiguous_propagation_node_count=int(len(ambiguous_nodes)),
        ambiguous_propagation_fraction=(len(ambiguous_nodes)/sum(node_propagates.values()) if any(node_propagates.values()) else None),
        ambiguous_cross_wall_propagation_fraction=(len(ambiguous_nodes)/sum(node_propagates.values()) if any(node_propagates.values()) else None),
        other_significant_activity_count=len(activities)-legacy_count,
        effective_observation_seconds=observation_seconds,
        direction_used_for_counting=False,cutoff=.6,
        count_algorithm_version='activity_count_v2_logic_fix_only',
        reviewed=False,frozen_count_algorithm=False,
        interpretation=('algorithm-defined significant propagation activity; '
                        'identity uncertainty is represented by a count range; weak activity may be missed'))
