"""Describe activity, waveform continuity and identity ambiguity of saved candidates.

These are review diagnostics, not new physiological features or a detector.
The surrounding signal may contain real waves: context contrast is not SNR and
must not be used as a threshold for rejecting periodic or weak activity.
"""
import numpy as np


def _runs(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)


def _rms(values):
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else None


def _correlation(a, b):
    if len(a) < 3:
        return None
    a, b = a-a.mean(), b-b.mean()
    scale = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(a@b/scale, -1, 1)) if scale > 1e-20 else None


def _summary(values):
    values = [v for v in values if v is not None]
    return dict(count=len(values), mean=float(np.mean(values)) if values else None,
                minimum=float(min(values)) if values else None)


def _phase_nodes(wall, y, valid, t):
    """Recover exact signed phase and finite run around each saved native peak."""
    cache = {}
    for ridge in wall['phase_ridges']:
        if len(ridge['sections']) != len(ridge['peak_time_s']):
            raise ValueError('Phase sections and peak times must match')
        nodes = {}
        for col, peak_s in zip(ridge['sections'], ridge['peak_time_s']):
            peak = int(np.searchsorted(t, peak_s))
            if peak == len(t) or abs(t[peak]-peak_s) > 1e-8 or not valid[peak, col]:
                raise ValueError('Saved phase peaks and filtered native maps differ')
            key = col, peak
            if key not in cache:
                signed = valid[:, col] & ((y[:, col] >= 0) == (ridge['sign'] > 0))
                phases = _runs(signed)
                run = _runs(valid[:, col])
                begin, end = phases[(phases[:, 0] <= peak) & (phases[:, 1] > peak)][0]
                first, last = run[(run[:, 0] <= peak) & (run[:, 1] > peak)][0]
                cache[key] = (int(begin), int(end), int(first), int(last))
            nodes[col] = cache[key]
        yield ridge['ridge_id'], nodes


def _event_evidence(event, nodes, y, valid, t):
    dt = float(np.median(np.diff(t)))
    ids = np.asarray(event['sections'], int)
    arrival = np.asarray(event['arrival_time_s'], float)
    if ids.shape != arrival.shape or not len(ids) or np.any(np.diff(ids) != 1):
        raise ValueError('Expected contiguous native candidate sections and matching arrivals')
    footprints = {}
    activity, context, per_section = [], [], []
    for col in ids:
        phases = [nodes[ridge][col] for ridge in event['phase_ridge_ids']]
        begin, end = min(p[0] for p in phases), max(p[1] for p in phases)
        first, last = phases[0][2:]
        if any(p[2:] != (first, last) for p in phases) or not valid[begin:end, col].all():
            raise ValueError('Candidate phases cross a missing interval')
        signal = y[begin:end, col]
        surroundings = np.r_[y[first:begin, col], y[end:last, col]]
        footprints[int(col)] = begin, end
        activity.extend(signal)
        context.extend(surroundings)
        ar, cr = _rms(signal), _rms(surroundings)
        per_section.append(dict(section=int(col), activity_rms=ar, context_rms=cr,
            activity_start_s=float(t[begin]), activity_end_s=float(t[end-1]),
            same_run_start_s=float(t[first]), same_run_end_s=float(t[last-1]),
            activity_duration_s=float((end-begin)*dt), context_duration_s=float(len(surroundings)*dt)))
    ar, cr = _rms(activity), _rms(context)
    pairs = []
    for j, (left, right) in enumerate(zip(ids[:-1], ids[1:])):
        a, b = footprints[int(left)], footprints[int(right)]
        delay_s = float(arrival[j+1]-arrival[j])
        shift = int(np.rint(delay_s/dt))
        start, stop = max(a[0], b[0]-shift), min(a[1], b[1]-shift)
        aligned = (_correlation(y[start:stop, left], y[start+shift:stop+shift, right])
                   if stop > start else None)
        start0, stop0 = max(a[0], b[0]), min(a[1], b[1])
        synchronous = (_correlation(y[start0:stop0, left], y[start0:stop0, right])
                       if stop0 > start0 else None)
        pairs.append(dict(left_section=int(left), right_section=int(right), delay_s=delay_s,
            native_shift_frames=shift, alignment_rounding_error_s=shift*dt-delay_s,
            aligned_correlation=aligned, zero_lag_correlation=synchronous,
            correlation_gain=aligned-synchronous if aligned is not None and synchronous is not None else None,
            aligned_samples=max(0, stop-start)))
    delta = np.diff(arrival)
    resolved = abs(delta) > dt
    slope = event['arrival_time_slope']
    reversed_steps = resolved & (delta*slope < 0)
    return dict(event_id=event['event_id'], classification=event['classification'],
        evidence_kind=event['evidence_kind'], quality_flags=list(event['quality_flags']),
        activity=dict(units='s^-1', footprint_rms=ar, same_run_context_rms=cr,
            activity_to_context_rms_ratio=ar/cr if ar is not None and cr is not None and cr > 1e-15 else None,
            footprint_samples=len(activity), context_samples=len(context),
            context_is_noise_reference=False, context_may_contain_other_waves=True,
            per_section=per_section),
        continuity=dict(adjacent_pairs=pairs,
            aligned_correlation=_summary([p['aligned_correlation'] for p in pairs]),
            zero_lag_correlation=_summary([p['zero_lag_correlation'] for p in pairs]),
            alignment_gain=_summary([p['correlation_gain'] for p in pairs]),
            resolved_adjacent_steps=int(resolved.sum()), reversed_adjacent_steps=int(reversed_steps.sum()),
            adjacent_order_agreement=float(np.mean(~reversed_steps[resolved])) if resolved.any() else None,
            phase_direction_agreement=[p['slope']*slope > 0 for p in event['phase_arrival_fits']]),
        confirmed_event=False)


def _relations(events, s):
    """Report possible overlap, never merge or equate overlap with identity."""
    rows = []
    for i, left in enumerate(events):
        for right in events[i+1:]:
            overlap = min(left['end_s'], right['end_s'])-max(left['start_s'], right['start_s'])
            same_wall = left['wall'] == right['wall']
            shared_phases = (sorted(set(left['phase_ridge_ids']) & set(right['phase_ridge_ids']))
                             if same_wall else [])
            if overlap <= 0 and not shared_phases:
                continue
            shared = sorted(set(left['sections']) & set(right['sections']))
            comparable = (left['arrival_marker'] == right['arrival_marker'] and
                          left['leading_polarity'] == right['leading_polarity'])
            if shared and comparable:
                a = dict(zip(left['sections'], left['arrival_time_s']))
                b = dict(zip(right['sections'], right['arrival_time_s']))
                lag = np.array([b[j]-a[j] for j in shared])
                mean_lag = float(lag.mean())
                spread = float(np.sqrt(np.mean((lag-mean_lag)**2)))
            else:
                mean_lag, spread = None, None
            rows.append(dict(left_id=left['event_id'], right_id=right['event_id'],
                same_wall=same_wall, temporal_overlap_s=float(max(overlap, 0)),
                shared_sections=shared, shared_s=[float(s[j]) for j in shared],
                shared_phase_ridge_ids=shared_phases,
                arrival_markers_comparable=comparable,
                mean_arrival_lag_s=mean_lag, lag_variation_rms_s=spread,
                same_direction=left['classification'] == right['classification'],
                same_event_confirmed=False))
    return rows


def analyze_candidate_evidence(candidates, arrays):
    """Consume unchanged candidate JSON and filtered_A/time/s/valid arrays."""
    y = np.asarray(arrays['filtered_A'], float)
    t = np.asarray(arrays['interval_center_s'], float)
    s = np.asarray(arrays['s'], float)
    valid = np.asarray(arrays['candidate_valid_mask'], bool) & np.isfinite(y)
    if (y.shape != (len(t), 2, len(s)) or valid.shape != y.shape or len(t) < 3 or
            not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or
            not np.isfinite(s).all() or np.any(np.diff(s) <= 0) or np.any((s < 0) | (s > 1)) or
            not np.allclose(np.diff(t), np.median(np.diff(t)), rtol=1e-5, atol=1e-9)):
        raise ValueError('Expected matching filtered radial maps and native uniform timestamps')
    evidence, all_events = [], []
    for wall in candidates['walls']:
        index = wall['wall']
        nodes = dict(_phase_nodes(wall, y[:, index], valid[:, index], t))
        for event in wall['events']:
            row = _event_evidence(event, nodes, y[:, index], valid[:, index], t)
            evidence.append(dict(wall=index, **row))
            all_events.append(dict(event, wall=index))
    relations = _relations(all_events, s)
    return dict(status='review_evidence_only_no_new_detector_or_threshold',
                events=evidence, relations=relations,
                exact_same_wall_phase_reuses=sum(bool(r['shared_phase_ridge_ids']) for r in relations),
                physiological_event_count=None, event_frequency_cpm=None,
                note='Context can contain real waves; correlation and overlap do not establish event identity.')
