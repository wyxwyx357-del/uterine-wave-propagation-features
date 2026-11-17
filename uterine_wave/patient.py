"""Fixed-region patient summaries, with explicit observation support."""
import numpy as np
from .amplitude import summarize_interval_strain
from .events import measure_event, pair_wall_events
from .modes import analyze_modes, contiguous_blocks


def _duration(mask, time_s):
    """Union of adjacent supported time intervals; gaps contribute zero."""
    return float(np.diff(time_s)[mask[:-1] & mask[1:]].sum())


def summarize_patient(spacetime, *, s_range, time_range=None, review=None, refine_mode_speed=False):
    """Same preselected anatomical region for amplitude, modes and exposure.

    review=None means no event review, not zero waves. A review contains
    complete (bool), intervals [[start,end], ...], and events (see events.py).
    Frequency is emitted only for an exhaustive review of supported exposure.
    Separate fragments are never concatenated before filtering or FFT.
    """
    t = np.asarray(spacetime['interval_center_s'], float)
    s = np.asarray(spacetime['s'], float)
    a = np.asarray(spacetime['A'], float)
    eps = np.asarray(spacetime['epsilon'], float)
    valid = np.asarray(spacetime['valid_mask'], bool)
    if (t.ndim != 1 or len(t) < 3 or not np.isfinite(t).all() or
            np.any(np.diff(t) <= 0) or not np.allclose(np.diff(t), np.median(np.diff(t)), rtol=1e-5)):
        raise ValueError('Patient summary requires increasing uniform native times')
    if (s.ndim != 1 or not np.isfinite(s).all() or np.any(np.diff(s) <= 0) or
            a.shape != (len(t), 2, len(s)) or eps.shape != a.shape or valid.shape != a.shape):
        raise ValueError('Expected matching time x wall x section maps')
    dt = np.asarray(spacetime['interval_dt_s'], float)
    if dt.shape != t.shape or not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError('interval_dt_s must contain one finite positive duration per sample')
    if ('interval_start_s' in spacetime) != ('interval_end_s' in spacetime):
        raise ValueError('Supply both interval_start_s and interval_end_s')
    # Interval center and duration determine the exact endpoints for older
    # maps that archived only these two fields. No motion is interpolated.
    interval_start = np.asarray(spacetime.get('interval_start_s', t-dt/2), float)
    interval_end = np.asarray(spacetime.get('interval_end_s', t+dt/2), float)
    if (interval_start.shape != t.shape or interval_end.shape != t.shape or
            not np.isfinite(interval_start).all() or not np.isfinite(interval_end).all() or
            not np.allclose(interval_end-interval_start, dt, rtol=1e-5, atol=1e-9) or
            not np.allclose((interval_start+interval_end)/2, t, rtol=0, atol=1e-6)):
        raise ValueError('Interval endpoints must agree with the archived centers and durations')
    if time_range is not None:
        if (len(time_range) != 2 or not np.isfinite(time_range).all() or
                not interval_start.min() <= time_range[0] < time_range[1] <= interval_end.max()):
            raise ValueError('time_range must lie inside the acquisition interval bounds')
        select = (interval_start >= time_range[0]) & (interval_end <= time_range[1])
        if select.sum() < 3:
            raise ValueError('The selected time range needs at least three native samples')
        spacetime = dict(spacetime)
        for key in ('A', 'epsilon', 'valid_mask', 'interval_center_s', 'interval_dt_s',
                    'interval_start_s', 'interval_end_s'):
            if key in spacetime:
                spacetime[key] = np.asarray(spacetime[key])[select]
        t, a, eps, valid = t[select], a[select], eps[select], valid[select]
        interval_start, interval_end, dt = interval_start[select], interval_end[select], dt[select]
    if (len(s_range) != 2 or not np.isfinite(s_range).all() or
            not 0 <= s_range[0] < s_range[1] <= 1):
        raise ValueError('s_range must be ordered cervix-to-fundus coordinates inside 0..1')
    ids = np.flatnonzero((s >= s_range[0]) & (s <= s_range[1]))
    if len(ids) < 3:
        raise ValueError('The selected fixed region needs at least three sampled sections')
    good = valid & np.isfinite(a) & np.isfinite(eps)
    lo, hi = int(ids[0]), int(ids[-1]+1)
    walls = []
    # At least one retained FFT frequency is needed, not a demand for 120 s.
    minimum = max(3, int(np.ceil(60/5/np.median(np.diff(t)))))
    for wall in (0, 1):
        per_pair = []
        for section in ids:
            blocks = []
            for start, end in contiguous_blocks(good[:, wall, section], minimum):
                amplitude = summarize_interval_strain(eps[start:end, wall, section], t[start:end])
                if amplitude['positive_frequency_bins'] == 0:
                    continue
                blocks.append(dict(start_s=float(t[start]), end_s=float(t[end-1]),
                                   samples=end-start, band_SD=float(amplitude['band_SD_mean']),
                                   raw_SD=float(amplitude['raw_SD_mean']),
                                   frequency_resolution_cpm=amplitude['time_resolution_cpm']))
            samples = sum(row['samples'] for row in blocks)
            pooled = {kind: float(np.sqrt(sum(row['samples']*row[kind]**2 for row in blocks)/samples))
                      if samples else None for kind in ('band_SD', 'raw_SD')}
            per_pair.append(dict(section=int(section), s=float(s[section]),
                                 analyzed_samples=samples, blocks=blocks, **pooled))
        analyzed = [row for row in per_pair if row['band_SD'] is not None]
        covers_region = len(analyzed) == len(ids)
        available_amplitude = float(np.mean([row['band_SD'] for row in analyzed])) if analyzed else None
        common = good[:, wall, lo:hi].all(1)
        walls.append(dict(wall=wall, per_pair=per_pair,
                          radial_stretch_activity_amplitude=available_amplitude if covers_region else None,
                          partial_region_activity_amplitude_QC=available_amplitude if not covers_region else None,
                          amplitude_status=('complete_requested_region' if covers_region else
                                            'insufficient_section_support' if analyzed else 'no_analyzable_sections'),
                          raw_interval_strain_SD_QC=float(np.mean([row['raw_SD'] for row in analyzed])) if analyzed else None,
                          analyzed_sections=len(analyzed), requested_sections=len(ids),
                          amplitude_covers_all_requested_sections=covers_region,
                          valid_fraction=float(good[:, wall, lo:hi].mean()),
                          common_region_duration_s=_duration(common, t)))
    # Retain the exact region: a varying set of surviving columns would change
    # the patient's physiological measurement between windows.
    fixed = dict(spacetime, valid_mask=good.copy())
    complete = good[:, :, lo:hi].all(2)
    fixed['valid_mask'][:, :, lo:hi] &= complete[:, :, None]
    modes = analyze_modes(fixed, regions=[(0, lo, hi, 'anterior'), (1, lo, hi, 'posterior')],
                          refine_velocity=refine_mode_speed)
    events, pairs = [], []
    counts = dict(status='not_reviewed', event_count=None, event_frequency_cpm=None)
    if review is not None:
        intervals = np.asarray(review['intervals'], float)
        if (intervals.ndim != 2 or intervals.shape[1] != 2 or not np.isfinite(intervals).all()
                or np.any(intervals[:, 1] <= intervals[:, 0]) or
                np.any(intervals[:, 0] < interval_start.min()) or
                np.any(intervals[:, 1] > interval_end.max())):
            raise ValueError('Reviewed intervals must lie inside the retained acquisition bounds')
        reviewed = np.zeros(len(t), bool)
        for start, end in intervals:
            reviewed |= (interval_start >= start) & (interval_end <= end)
        seen = set()
        for item in review['events']:
            key = (str(item['event_id']), item['wall'])
            if key in seen:
                raise ValueError('Duplicate event_id and wall; merge the annotations first')
            seen.add(key)
            row = measure_event(spacetime, item)
            inside_region = row['sections'][0] >= lo and row['sections'][-1] < hi
            inside_review = any(start <= row['start_s'] and row['end_s'] <= end for start, end in intervals)
            supported = (t >= row['start_s']) & (t <= row['end_s'])
            if not inside_region or not inside_review or not complete[supported, row['wall']].all():
                row['status'] = 'outside_fixed_observation_support'
            events.append(row)
        for event_id in {row['event_id'] for row in events}:
            rows = [row for row in events if row['event_id'] == event_id]
            if len(rows) == 2 and max(row['start_s'] for row in rows) > min(row['end_s'] for row in rows):
                raise ValueError('Shared event_id on two walls requires overlapping event time bounds')
        exposure = complete & reviewed[:, None]
        duration = _duration(exposure.any(1), t)
        accepted = [row for row in events if row['status'] in ('reviewed_propagation', 'reviewed_complex_activity')]
        ids_counted = {row['event_id'] for row in accepted}
        failed = [row for row in events if row['status'] not in
                  ('reviewed_propagation', 'reviewed_complex_activity', 'no_resolved_propagation')]
        exhaustive = bool(review['complete']) and not failed and duration > 0
        directions = {}
        for event_id in sorted(ids_counted):
            values = {row['direction'] for row in accepted if row['event_id'] == event_id}
            directions[event_id] = next(iter(values)) if len(values) == 1 else 'complex'
        speeds = {name: [] for name in ('C2F', 'F2C')}
        for event_id, name in directions.items():
            if name in speeds:
                values = [row['velocity_paths_per_s'] for row in accepted
                          if row['event_id'] == event_id and row['velocity_paths_per_s'] is not None]
                if values:
                    speeds[name].append(float(np.mean(values)))
        counts = dict(status='reviewed' if exhaustive else 'partial_review_or_support',
                      event_count=len(ids_counted) if exhaustive else None,
                      supported_reviewed_event_count=len(ids_counted),
                      event_frequency_cpm=60*len(ids_counted)/duration if exhaustive else None,
                      effective_duration_s=duration,
                      duration_rule='union_of_reviewed_fixed_region_support_on_either_wall',
                      median_event_velocity_paths_per_s={name: float(np.median(values)) if values else None
                                                        for name, values in speeds.items()},
                      velocity_pooling='mean_across_paired_walls_then_median_across_events_per_direction',
                      direction_proportions={name: sum(value == name for value in directions.values())/len(directions)
                                             if directions else None for name in ('C2F', 'F2C', 'complex')})
        pairs = pair_wall_events(accepted, s)
    return dict(definition_version='radial_wave_features_v2',
                requested_s_range=list(s_range), sampled_s_range=[float(s[lo]), float(s[hi-1])],
                analyzed_time_range_s=[float(t[0]), float(t[-1])],
                analyzed_interval_center_range_s=[float(t[0]), float(t[-1])],
                analyzed_acquisition_time_range_s=[float(interval_start.min()), float(interval_end.max())],
                requested_acquisition_time_range_s=list(time_range) if time_range is not None else None,
                time_range_rule='full_measurement_interval_containment',
                sections=ids.tolist(), nominal_interval_s=float(spacetime.get('nominal_interval_s', np.median(dt))),
                actual_interval_s_range=[float(dt.min()), float(dt.max())],
                amplitude_units='dimensionless', amplitude_band_cpm=[.5, 5.],
                amplitude_pooling='sample_weighted_within_segment_variance_per_pair_then_spatial_mean',
                primary_amplitude_requires_all_requested_sections=True,
                amplitude_interpretation='interval_radial_stretch_activity_not_total_single_contraction_strain',
                walls=walls, modes=modes, events=events, event_summary=counts,
                paired_wall_events=pairs, event_velocity_units='full_cervix_fundus_paths_per_second')

