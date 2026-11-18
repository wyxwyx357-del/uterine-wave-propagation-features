"""Leave-one-position-out checks of frozen phase connections; no classifier.

Arrival at a held-out location is predicted from the other phase arrivals,
under the same constant-speed approximation as the current velocity fit.
"""
import numpy as np
from uterine_wave.candidate_evidence import _phase_nodes


def _cosine(a, b):
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(a@b/norm, 0, 1)) if norm > 0 else None


def predict_phase(filtered, time_s, space, sections, nodes, peak_times):
    """Native phase templates; no optimal-shift search or time interpolation.

    Zero outside each signed phase defines a comparison template, not missing
    motion. Predictions are clipped to the held-out point's continuous run;
    clipped support is reported and never silently treated as complete.
    """
    y, t, s = np.asarray(filtered, float), np.asarray(time_s, float), np.asarray(space, float)
    columns, peaks = np.asarray(sections, int), np.asarray(peak_times, float)
    if (y.shape != (len(t), len(s)) or len(t) < 3 or np.any(np.diff(t) <= 0) or
            not np.allclose(np.diff(t), np.median(np.diff(t)), rtol=1e-5, atol=1e-9)):
        raise ValueError('Expected native-uniform time x space phase maps')
    if len(columns) < 3 or columns.shape != peaks.shape or np.any(np.diff(columns) != 1):
        raise ValueError('Expected at least three contiguous phase positions')
    dt = float(np.median(np.diff(t)))
    observations = []
    for held, col in enumerate(columns):
        keep = np.arange(len(columns)) != held
        slope, intercept = np.polyfit(s[columns[keep]], peaks[keep], 1)
        predicted_time = float(slope*s[col]+intercept)
        predicted_index = int(np.rint((predicted_time-t[0])/dt))
        actual_index = int(np.argmin(abs(t-peaks[held])))
        begin, end, first, last = nodes[int(col)]
        observed = np.zeros(last-first)
        values = abs(y[begin:end, col])
        norm = np.linalg.norm(values)
        if norm:
            observed[begin-first:end-first] = values/norm

        def template(center):
            result = np.zeros(last-first)
            retained, templates = [], 0
            for train_col, peak_s in zip(columns[keep], peaks[keep]):
                a, b, _, _ = nodes[int(train_col)]
                values = abs(y[a:b, train_col])
                if not np.isfinite(values).all():
                    raise ValueError('A phase template crosses missing data')
                norm = np.linalg.norm(values)
                if not norm:
                    continue
                train_peak = int(np.argmin(abs(t-peak_s)))
                offset = center-train_peak
                left, right = max(a+offset, first), min(b+offset, last)
                retained.append(max(0, right-left)/(b-a))
                if right > left:
                    result[left-first:right-first] += values[left-offset-a:right-offset-a]/norm
                    templates += 1
            return result, float(np.mean(retained)) if retained else None, templates

        predicted, support, count = template(predicted_index)
        aligned, aligned_support, _ = template(actual_index)
        in_run = bool(first <= predicted_index < last)
        observations.append(dict(section=int(col), actual_peak_s=float(peaks[held]),
            predicted_peak_s=predicted_time, arrival_error_s=predicted_time-float(peaks[held]),
            spatial_prediction='interpolation' if s[columns[keep]].min() <= s[col] <= s[columns[keep]].max() else 'extrapolation',
            rounding_error_s=float(t[0]+predicted_index*dt-predicted_time),
            prediction_inside_continuous_run=in_run,
            predicted_shape_cosine=_cosine(observed, predicted) if in_run else None,
            peak_aligned_shape_cosine=_cosine(observed, aligned),
            retained_prediction_support_fraction=support,
            retained_aligned_support_fraction=aligned_support,
            training_templates_used=count, training_positions=int(keep.sum())))
    errors = np.array([x['arrival_error_s'] for x in observations])
    cosines = [x['predicted_shape_cosine'] for x in observations if x['predicted_shape_cosine'] is not None]
    aligned = [x['peak_aligned_shape_cosine'] for x in observations if x['peak_aligned_shape_cosine'] is not None]
    extent = float(np.ptp(peaks))
    return dict(positions=observations, arrival_loo_RMSE_s=float(np.sqrt(np.mean(errors**2))),
                arrival_range_s=extent,
                arrival_loo_RMSE_over_range=float(np.sqrt(np.mean(errors**2))/max(extent, dt)),
                predicted_shape_cosine_mean=float(np.mean(cosines)) if cosines else None,
                peak_aligned_shape_cosine_mean=float(np.mean(aligned)) if aligned else None,
                prediction_positions=len(cosines), total_positions=len(columns),
                constant_speed_approximation=True)


def analyze_prediction(candidates, arrays):
    y, t, s = arrays['filtered_A'], arrays['interval_center_s'], arrays['s']
    valid = arrays['candidate_valid_mask']
    events = []
    for wall in candidates['walls']:
        w = wall['wall']
        nodes = dict(_phase_nodes(wall, y[:, w], valid[:, w], t))
        phases = {r['ridge_id']:r for r in wall['phase_ridges']}
        for event in wall['events']:
            if event['classification'] not in ('C2F', 'F2C'):
                continue
            details = []
            for ridge in event['phase_ridge_ids']:
                peaks = dict(zip(phases[ridge]['sections'], phases[ridge]['peak_time_s']))
                result = predict_phase(y[:, w], t, s, event['sections'], nodes[ridge],
                                       [peaks[col] for col in event['sections']])
                details.append(dict(ridge_id=ridge, sign=phases[ridge]['sign'], **result))
            def average(name):
                values = [x[name] for x in details if x[name] is not None]
                return float(np.mean(values)) if values else None
            events.append(dict(event_id=event['event_id'], wall=w,
                direction=event['classification'], evidence_kind=event['evidence_kind'],
                sections=event['sections'], phases=details,
                arrival_loo_RMSE_s=average('arrival_loo_RMSE_s'),
                arrival_loo_RMSE_over_range=average('arrival_loo_RMSE_over_range'),
                predicted_shape_cosine_mean=average('predicted_shape_cosine_mean'),
                peak_aligned_shape_cosine_mean=average('peak_aligned_shape_cosine_mean'),
                reviewed=False))
    return dict(events=events, event_count=None, decision_threshold=None,
                interpretation='Internal phase-connection prediction check, not independent physiological truth')
