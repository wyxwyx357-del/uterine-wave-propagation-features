"""Revalidate retained-direction evidence before cross-wall direction reporting.

This is an experimental reconciliation layer.  It does not create candidates,
change the fixed salience cutoff, or replace formal event outputs.
"""
from copy import deepcopy
from itertools import combinations

import numpy as np

from analysis.mse_pregnancy_association.code.measurement.radial_events import RadialWaveConfig, _describe_arrival


DIRECTIONS = ('C2F', 'F2C')


def _longest_adjacent(ids):
    ids = np.asarray(sorted(set(ids)), int)
    return max((len(block) for block in np.split(
        ids, np.flatnonzero(np.diff(ids) != 1) + 1)), default=0)


def _adjacent_blocks(ids):
    ids = np.asarray(sorted(set(ids)), int)
    if not len(ids):
        return []
    return [block for block in np.split(
        ids, np.flatnonzero(np.diff(ids) != 1) + 1) if len(block)]


def _phase_subset_matches(left, right, tolerance_s):
    """Allow extra local phases while matching the smaller signed sequence."""
    smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
    smaller = sorted(smaller, key=lambda phase: phase['peak_s'])
    larger = sorted(larger, key=lambda phase: phase['peak_s'])
    first_available = 0
    matches = []
    for phase in smaller:
        choices = [(abs(phase['peak_s'] - larger[j]['peak_s']), j)
                   for j in range(first_available, len(larger))
                   if phase['sign'] == larger[j]['sign']]
        if not choices:
            return False, matches
        distance, index = min(choices)
        if distance > tolerance_s:
            return False, matches
        first_available = index + 1
        matches.append((phase['peak_s'], larger[index]['peak_s']))
    return bool(matches), matches


def compare_identity_without_direction(a, b, span_s, sample_dt_s):
    """Compare activity identity before using either wall's direction label."""
    if a['wall'] == b['wall']:
        raise ValueError('Identity comparison requires different walls')
    tolerance = float(span_s + sample_dt_s)
    shared = sorted(set(a['sections']) & set(b['sections']))
    matched, near, conflicts = [], [], []
    for section in shared:
        arrival_near = (a['arrival_marker'] == b['arrival_marker'] and
                        abs(a['arrival'][section] - b['arrival'][section]) <= tolerance)
        phases_match, _ = _phase_subset_matches(
            a['profile'][section], b['profile'][section], tolerance)
        if arrival_near or phases_match:
            near.append(section)
        if arrival_near and phases_match:
            matched.append(section)
        elif arrival_near or phases_match:
            conflicts.append(section)
    supported = _longest_adjacent(matched) >= 3
    relation = ('shared_identity_supported' if supported else
                'unresolved' if near else 'no_shared_identity_evidence')
    return dict(node_a=a['node_id'], node_b=b['node_id'], relation=relation,
                shared_sections=shared, matched_sections=matched,
                near_sections=near, conflict_sections=conflicts,
                original_directions=[a['direction'], b['direction']],
                original_direction_conflict=(a['direction'] in DIRECTIONS and
                                             b['direction'] in DIRECTIONS and
                                             a['direction'] != b['direction']),
                tolerance_s=tolerance,
                rule='>=3 adjacent arrival-and-phase matches; direction not used for identity')


def revalidate_retained_direction(node, s, sample_dt_s, config=RadialWaveConfig()):
    """Refit each contiguous retained block; never bridge a salience gap."""
    sections = np.asarray(sorted(node['sections']), int)
    blocks = _adjacent_blocks(sections)
    measurements = []
    for block in blocks:
        arrival = np.asarray([node['arrival'][int(section)] for section in block], float)
        measurement = _describe_arrival(
            block, arrival, np.asarray(s, float), float(sample_dt_s), [], config)
        measurements.append(dict(sections=block.tolist(), arrival_time_s=arrival.tolist(),
                                 measurement=measurement))
    block_directions = [item['measurement']['classification'] for item in measurements
                        if item['measurement']['classification'] in DIRECTIONS]
    refit = block_directions[0] if len(set(block_directions)) == 1 else None
    original = node['direction'] if node['direction'] in DIRECTIONS else None
    if original is None:
        status, supported = 'original_direction_unresolved', None
    elif any(direction != original for direction in block_directions):
        status, supported = 'retained_support_conflicts_with_original', None
    elif any(direction == original for direction in block_directions):
        status, supported = 'retained_support_agrees', original
    else:
        status, supported = 'retained_support_unresolved', None
    return dict(node_id=node['node_id'], original_direction=original,
                retained_refit_direction=refit, supported_direction=supported,
                direction_status=status, retained_sections=sections.tolist(),
                retained_contiguous_blocks=[block.tolist() for block in blocks],
                block_direction_evidence=measurements,
                direction_rule=('each contiguous retained block refit separately; '
                                'opposing block makes direction unresolved'))


def reconcile_identity_then_direction(nodes, s, span_s, sample_dt_s):
    """Form cross-wall identities first, then combine revalidated directions."""
    comparisons = [compare_identity_without_direction(a, b, span_s, sample_dt_s)
                   for a, b in combinations(nodes, 2) if a['wall'] != b['wall']]
    choices = {node['node_id']: [] for node in nodes}
    uncertain = {node['node_id']: [] for node in nodes}
    for edge in comparisons:
        target = choices if edge['relation'] == 'shared_identity_supported' else (
            uncertain if edge['relation'] == 'unresolved' else None)
        if target is not None:
            target[edge['node_a']].append(edge['node_b'])
            target[edge['node_b']].append(edge['node_a'])
    pairs = {tuple(sorted((node_id, matches[0])))
             for node_id, matches in choices.items()
             if len(matches) == 1 and choices[matches[0]] == [node_id]
             and not uncertain[node_id] and not uncertain[matches[0]]}
    lookup = {node['node_id']: node for node in nodes}
    used, groups = set(), []
    for left, right in sorted(pairs):
        groups.append([lookup[left], lookup[right]])
        used.update((left, right))
    groups.extend([[node] for node in nodes if node['node_id'] not in used])
    groups.sort(key=lambda group: (min(np.median(list(node['arrival'].values()))
                                       for node in group), group[0]['node_id']))

    activities = []
    for index, group in enumerate(groups, 1):
        evidence = [revalidate_retained_direction(node, s, sample_dt_s) for node in group]
        supported = {item['supported_direction'] for item in evidence
                     if item['supported_direction'] in DIRECTIONS}
        original = {node['direction'] for node in group if node['direction'] in DIRECTIONS}
        if len(original) > 1:
            direction, direction_status = None, 'cross_wall_direction_conflict'
        elif len(supported) == 1:
            direction = next(iter(supported))
            direction_status = ('cross_wall_direction_agreement' if len(group) == 2 and
                                len([item for item in evidence if item['supported_direction']]) == 2
                                else 'single_wall_direction_support')
        elif len(supported) > 1:
            direction, direction_status = None, 'cross_wall_direction_conflict'
        else:
            direction, direction_status = None, 'direction_unresolved_after_retained_refit'
        members = [node['node_id'] for node in group]
        related = sorted({other for member in members
                          for other in uncertain[member] + choices[member]
                          if other not in members})
        phases = [phase for node in group for values in node['profile'].values()
                  for phase in values]
        activity_id=f'activity_{index:03d}'
        activities.append(dict(activity_id=activity_id,patient_candidate_id=activity_id,
            source_nodes=members, walls=sorted({node['wall'] for node in group}),
            cross_wall_identity=len(group) == 2, direction=direction,
            direction_status=direction_status, direction_evidence=evidence,
            classification=('propagation' if direction else 'direction_unresolved_activity'),
            ownership_status=('unresolved_cross_wall' if related else
                              'shared_identity' if len(group) == 2 else 'single_wall'),
            unresolved_related_nodes=related,
            evidence_start_s=min(phase['start_s'] for phase in phases),
            evidence_end_s=max(phase['end_s'] for phase in phases),
            source_measurements=deepcopy(group), reviewed=False))
    return dict(activities=activities, identity_comparisons=comparisons,
                input_candidate_rows=len(nodes), activity_rows=len(activities),
                cross_wall_identity_pairs=len(pairs),
                direction_counts={direction: sum(item['direction'] == direction
                                                  for item in activities)
                                  for direction in DIRECTIONS},
                direction_unresolved_rows=sum(item['direction'] is None for item in activities),
                formal_event_count=None, formal_event_frequency_cpm=None,
                experimental=True)
