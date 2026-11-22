"""Sandbox: retain observable boundary peaks and propose fragment ownership.

Default detector and formal event interfaces are unchanged. Proposed groups
are not reviewed waves, and their number is not physiological frequency.
"""
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from uterine_wave import radial_events as baseline


def observed_lobes(y, valid, t, threshold):
    filtered = np.full_like(y, np.nan)
    columns = [[] for _ in range(y.shape[1])]
    unresolved = []
    minimum = max(3, int(np.ceil(12/np.median(np.diff(t)))))
    for col in range(y.shape[1]):
        for run_id, (begin, end) in enumerate(baseline._runs(valid[:, col])):
            if end-begin < minimum:
                continue
            signal = baseline.band_limited_signal(y[begin:end, col], t[begin:end])
            filtered[begin:end, col] = signal
            edges = np.r_[0, np.flatnonzero(np.diff(signal >= 0))+1, len(signal)]
            for sequence, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
                peak = int(a+np.argmax(abs(signal[a:b])))
                amplitude = float(abs(signal[peak]))
                left, right = bool(a == 0), bool(b == len(signal))
                if amplitude <= max(float(threshold[col]), 1e-12):
                    continue
                if (left or right) and peak in (a, b-1):
                    unresolved.append(dict(column=col, run_id=run_id,
                        observed_start_s=float(t[begin+a]), observed_end_s=float(t[begin+b-1]),
                        peak_observed=False, left_censored=left, right_censored=right))
                    continue
                columns[col].append(dict(column=col, run_id=run_id, sequence=sequence,
                    sign=1 if signal[peak] > 0 else -1, peak_index=int(begin+peak),
                    peak_time_s=float(t[begin+peak]), amplitude=amplitude,
                    first=int(begin+a), last=int(begin+b-1),
                    start_s=float(t[begin+a]), end_s=float(t[begin+b-1]),
                    left_censored=left, right_censored=right, peak_observed=True))
    return filtered, columns, unresolved


def propose_groups(report):
    """Unique disjoint complementary ownership proposals, no automatic review.

    Only a whole single-phase candidate can attach, never silently assign a
    subset and lose its remaining support. Opposite temporal roles, ambiguous
    destinations and overlapping local phase nodes remain separate evidence.
    """
    events = report['events']
    paired = [e for e in events if e['evidence_kind'] == 'paired_phase']
    mapping = {e['event_id']:e['event_id'] for e in events}
    attachments, unresolved = {}, []
    used_sections = {e['event_id']:set(e['sections']) for e in paired}
    for event in events:
        if event['evidence_kind'] != 'single_phase':
            continue
        ridge = event['phase_ridge_ids'][0]
        choices = []
        for edge in report['alternative_phase_pairs']:
            if edge['selected'] or ridge not in edge['phase_ridge_ids']:
                continue
            if edge['sections'] != event['sections']:
                continue
            other = next(k for k in edge['phase_ridge_ids'] if k != ridge)
            for parent in paired:
                if (other in parent['phase_ridge_ids'] and
                    not set(edge['sections'])&set(parent['sections']) and
                    edge['leading_polarity'] == parent['leading_polarity'] and
                    edge['classification'] == parent['classification'] == event['classification'] and
                    event['classification'] in ('C2F', 'F2C')):
                    choices.append((parent, edge))
        if len(choices) != 1:
            if choices:
                unresolved.append(dict(candidate_id=event['event_id'], reason='multiple_possible_parent_events',
                                       possible_parents=[p['event_id'] for p, _ in choices]))
            continue
        parent, edge = choices[0]
        if set(edge['sections'])&used_sections[parent['event_id']]:
            unresolved.append(dict(candidate_id=event['event_id'], reason='competing_spatial_ownership'))
            continue
        used_sections[parent['event_id']].update(edge['sections'])
        mapping[event['event_id']] = parent['event_id']
        attachments[event['event_id']] = deepcopy(edge)
    groups = {}
    for event in events:
        groups.setdefault(mapping[event['event_id']], []).append(event['event_id'])
    return groups, attachments, unresolved


def detect_partial(signal, valid, time_s, space, *, noise_floor):
    original_ridges = baseline._ridges
    captured, unresolved_peaks, boundary_nodes = {}, [], []
    def lobes(y, supported, t, floor):
        filtered, columns, unresolved = observed_lobes(y, supported, t, floor)
        unresolved_peaks.extend(unresolved)
        boundary_nodes.extend(n for column in columns for n in column
                              if n['left_censored'] or n['right_censored'])
        return filtered, columns
    def ridges(columns, t, s):
        result = original_ridges(columns, t, s)
        captured.update({r['id']:r for r in result})
        return result
    with patch.object(baseline, '_lobes', lobes), patch.object(baseline, '_ridges', ridges):
        report = baseline.detect_radial_events(signal, valid, time_s, space, noise_floor=noise_floor)
    groups, attachments, unresolved = propose_groups(report)
    fragments = []
    for event in report['events']:
        piece = attachments.get(event['event_id'], event)
        nodes = []
        for rid in piece['phase_ridge_ids']:
            nodes.extend(n for n in captured[rid]['nodes'] if n['column'] in piece['sections'])
        leading = piece['phase_ridge_ids'][0]
        trailing = piece['phase_ridge_ids'][-1]
        # _measure_pair writes phase IDs in temporal order; singles have one.
        left = any(n['left_censored'] for n in captured[leading]['nodes'] if n['column'] in piece['sections'])
        right = any(n['right_censored'] for n in captured[trailing]['nodes'] if n['column'] in piece['sections'])
        fragments.append(dict(fragment_id=event['event_id'], source_candidate_id=event['event_id'],
            observed_start_s=piece['start_s'], observed_end_s=piece['end_s'],
            structure_start_s=None if left else piece['start_s'],
            structure_end_s=None if right else piece['end_s'],
            boundary_status=dict(start_unknown=left, end_unknown=right),
            sections=piece['sections'], arrival_time_s=piece['arrival_time_s'],
            arrival_marker=piece['arrival_marker'], direction=piece['classification'],
            phase_ridge_ids=piece['phase_ridge_ids'],
            local_phase_support=[dict(section=n['column'], run_id=n['run_id'], sign=n['sign'],
                observed_start_s=n['start_s'], observed_end_s=n['end_s'], peak_s=n['peak_time_s'],
                start_unknown=n['left_censored'], end_unknown=n['right_censored']) for n in nodes],
            time_bounds_semantics='Observed filtered-phase structure, not complete physiological contraction duration',
            relation='complementary_spatial_piece_proposal' if event['event_id'] in attachments else 'original_candidate',
            reviewed=False))
    report.update(candidate_algorithm='experimental_observable_partial_phase_v1',
        fragments=fragments,
        proposed_events=[dict(proposal_id=key, fragment_ids=ids, identity_reviewed=False)
                         for key, ids in groups.items()],
        ownership_ambiguities=unresolved,
        unresolved_boundary_segments=unresolved_peaks,
        observed_boundary_nodes=[dict(column=n['column'], run_id=n['run_id'],
            peak_time_s=n['peak_time_s'], observed_start_s=n['start_s'], observed_end_s=n['end_s'],
            left_censored=n['left_censored'], right_censored=n['right_censored'])
            for n in boundary_nodes],
        provisional_direction_groups=sum(any(report_event['classification'] in ('C2F', 'F2C')
            for report_event in report['events'] if report_event['event_id'] in ids) for ids in groups.values()),
        formal_patient_output=dict(event_count=None, event_frequency_cpm=None,
                                   status='identity_and_detection_not_validated'))
    return report


def anchor_to_existing_candidates(original, observed):
    """Use retained boundary evidence to repair existing proposals only.

    All original candidates remain traceable. Novel/unmatched boundary paths
    remain additional support, never automatically new independent events.
    Exact phase-node identity is used; no numerical match threshold is tuned.
    """
    def tokens(report, event):
        ridges = {r['ridge_id']:r for r in report['phase_ridges']}
        return {(ridges[k]['sign'], col, time) for k in event['phase_ridge_ids']
                for col, time in zip(ridges[k]['sections'], ridges[k]['peak_time_s'])
                if col in event['sections']}
    observed_by_id = {e['event_id']:e for e in observed['events']}
    pieces_by_id = {p['fragment_id']:p for p in observed['fragments']}
    group_tokens = {}
    for group in observed['proposed_events']:
        combined = set()
        for fragment_id in group['fragment_ids']:
            source = observed_by_id[fragment_id]
            piece = pieces_by_id[fragment_id]
            combined.update(tokens(observed, dict(source, sections=piece['sections'],
                                                   phase_ridge_ids=piece['phase_ridge_ids'])))
        group_tokens[group['proposal_id']] = combined
    destinations, ambiguity = {}, []
    for event in original['events']:
        signature = tokens(original, event)
        choices = [key for key, support in group_tokens.items() if signature <= support]
        if len(choices) == 1:
            destinations[event['event_id']] = choices[0]
        else:
            destinations[event['event_id']] = None
            ambiguity.append(dict(candidate_id=event['event_id'], reason='no_unique_complete_phase_support',
                                  possible_groups=choices))
    conflicting = set()
    for key in group_tokens:
        directions = {event['classification'] for event in original['events']
                      if destinations[event['event_id']] == key
                      and event['classification'] in ('C2F', 'F2C')}
        if len(directions) > 1:
            conflicting.add(key)
            ambiguity.append(dict(observed_group=key, reason='opposing_original_directions'))
    groups = {}
    for event in original['events']:
        destination = destinations[event['event_id']]
        key = ('observed', destination) if destination is not None and destination not in conflicting else ('original', event['event_id'])
        groups.setdefault(key, []).append(event)
    output = []
    used_observed = set()
    observed_groups = {g['proposal_id']:g for g in observed['proposed_events']}
    for number, (key, events) in enumerate(groups.items(), 1):
        if key[0] == 'observed':
            used_observed.add(key[1])
        output.append(dict(proposal_id=f'activity_proposal_{number:03d}',
            source_candidate_ids=[e['event_id'] for e in events],
            observed_fragment_ids=observed_groups[key[1]]['fragment_ids'] if key[0] == 'observed' else [],
            observed_support_group=key[1] if key[0] == 'observed' else None,
            direction_candidate=any(e['classification'] in ('C2F', 'F2C') for e in events),
            ownership_status='proposed_shared_identity' if len(events) > 1 else 'single_existing_candidate',
            identity_reviewed=False))
    return dict(existing_candidate_groups=output,
        additional_support_groups=[g for g in observed['proposed_events'] if g['proposal_id'] not in used_observed],
        mapping_ambiguities=ambiguity,
        existing_direction_group_proposals=sum(g['direction_candidate'] for g in output),
        formal_event_count=None, formal_event_frequency_cpm=None,
        interpretation='Support-assisted ownership proposals only; no new independent waves or reviewed identities')
