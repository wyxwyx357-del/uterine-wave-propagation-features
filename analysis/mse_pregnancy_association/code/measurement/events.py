"""Measure reviewed propagation events on the existing radial maps.

Arrival annotations describe the same contraction/relaxation structure across
positions. They are not detected from individual colour bands or ER windows.
One event_id identifies one activity across walls; a reflection is a new event.
"""
import numpy as np


def measure_event(spacetime, event):
    """Fit arrival time = intercept + slope*s; speed = 1/slope.

    Required: event_id, wall (0/1), start_s, end_s, sections (zero based),
    arrival_time_s, reviewed. Three distinct adjacent sections are a numerical
    fitting requirement, not a published physiological minimum wave extent.
    Non-monotone arrival sequences remain complex, with no single speed.
    """
    t = np.asarray(spacetime['interval_center_s'], float)
    s = np.asarray(spacetime['s'], float)
    wall = event['wall']
    ids = np.asarray(event['sections'])
    arrival = np.asarray(event['arrival_time_s'], float)
    start, end = float(event['start_s']), float(event['end_s'])
    if (wall not in (0, 1) or ids.ndim != 1 or len(ids) < 3 or
            not np.issubdtype(ids.dtype, np.integer) or
            np.any(np.diff(ids) != 1) or ids[0] < 0 or ids[-1] >= len(s)):
        raise ValueError('Event needs wall 0/1 and at least three adjacent section indices')
    if (not event['event_id'] or arrival.shape != ids.shape or not np.isfinite(arrival).all()
            or not np.isfinite([start, end]).all() or not start < end
            or np.any(arrival < start) or np.any(arrival > end)):
        raise ValueError('Event arrivals must be finite and inside its time bounds')
    row = dict(event_id=str(event['event_id']), wall=int(wall),
               start_s=start, end_s=end, sections=ids.tolist(),
               s_start=float(s[ids[0]]), s_end=float(s[ids[-1]]),
               spatial_span=float(s[ids[-1]]-s[ids[0]]),
               arrival_time_s=arrival.tolist(), reviewed=bool(event['reviewed']),
               arrival_time_slope=None, velocity_paths_per_s=None,
               event_interval_strain_SD=None, direction=None)
    if not row['reviewed']:
        return dict(row, status='needs_review')
    select = (t >= start) & (t <= end)
    valid = np.asarray(spacetime['valid_mask'], bool)
    a = np.asarray(spacetime['A'], float)
    eps = np.asarray(spacetime['epsilon'], float)
    if (start < t[0] or end > t[-1] or select.sum() < 3 or
            not valid[select, wall][:, ids].all() or
            not np.isfinite(a[select, wall][:, ids]).all() or
            not np.isfinite(eps[select, wall][:, ids]).all()):
        return dict(row, status='unsupported_time_or_segment_gap')
    # Keep the original finite-interval meaning. A short-event SD is unfiltered:
    # narrow event windows cannot represent the full physiological frequency band.
    row['event_interval_strain_SD'] = float(np.std(eps[select, wall][:, ids], axis=0).mean())
    if row['event_interval_strain_SD'] <= 1e-12:
        return dict(row, direction='local_or_synchronous', status='no_resolved_propagation')
    slope, intercept = np.polyfit(s[ids], arrival, 1)
    residual = arrival-(intercept+slope*s[ids])
    total = float(np.sum((arrival-arrival.mean())**2))
    row.update(arrival_time_slope=float(slope),
               arrival_fit_R2=float(1-np.sum(residual**2)/total) if total > 1e-20 else None,
               arrival_fit_RMSE_s=float(np.sqrt(np.mean(residual**2))),
               arrival_time_intercept=float(intercept), amplitude_units='dimensionless',
               amplitude_definition='mean_per_pair_unfiltered_event_interval_strain_SD')
    tolerance = float(np.median(np.diff(t)))
    delta = np.diff(arrival)
    if np.ptp(arrival) <= tolerance:
        return dict(row, direction='local_or_synchronous', status='no_resolved_propagation')
    monotone = np.all(delta >= -tolerance) or np.all(delta <= tolerance)
    if not monotone:
        return dict(row, direction='complex', status='reviewed_complex_activity')
    if abs(slope)*(s[ids[-1]]-s[ids[0]]) <= tolerance:
        return dict(row, direction='local_or_synchronous', status='no_resolved_propagation')
    return dict(row, direction='C2F' if slope > 0 else 'F2C',
                velocity_paths_per_s=float(1/slope), status='reviewed_propagation')


def pair_wall_events(events, space):
    """Pair only explicitly reviewed shared event IDs; never sum both walls."""
    pairs = []
    for event_id in sorted({row['event_id'] for row in events}):
        rows = [row for row in events if row['event_id'] == event_id and
                row['status'] in ('reviewed_propagation', 'reviewed_complex_activity')]
        if len(rows) != 2 or {row['wall'] for row in rows} != {0, 1}:
            continue
        anterior, posterior = sorted(rows, key=lambda row: row['wall'])
        common = sorted(set(anterior['sections']) & set(posterior['sections']))
        if not common:
            continue
        # Only shared sampled positions; no spatial interpolation.
        ta = dict(zip(anterior['sections'], anterior['arrival_time_s']))
        tp = dict(zip(posterior['sections'], posterior['arrival_time_s']))
        lag = [tp[j]-ta[j] for j in common]
        pairs.append(dict(event_id=event_id, sections=common,
                          s=np.asarray(space)[common].tolist(),
                          directions_agree=anterior['direction'] == posterior['direction'],
                          posterior_minus_anterior_arrival_s=lag,
                          mean_arrival_difference_s=float(np.mean(lag))))
    return pairs
