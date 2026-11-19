"""Whole-activity strength after ownership; no new detection or wave counts.

Median per-native-section phase-footprint RMS is compared with the maximum
matched-footprint RMS in calibration rigid residuals. Ratio > 1 is an
engineering prominence decision, not a physiological truth test.
"""
import numpy as np
from uterine_wave.amplitude import band_limited_signal


def filtered_runs(signal, valid, time_s):
    y = np.asarray(signal, float); m = np.asarray(valid, bool)
    t = np.asarray(time_s, float)
    if y.shape != m.shape or y.shape[0] != len(t) or len(t) < 3 or np.any(np.diff(t) <= 0):
        raise ValueError('Expected native time-first signals and matching validity')
    result = np.full_like(y, np.nan)
    minimum = max(3, int(np.ceil(12/np.median(np.diff(t)))))
    for col in np.ndindex(y.shape[1:]):
        key = (slice(None),)+col
        supported = m[key] & np.isfinite(y[key])
        for a, b in np.flatnonzero(np.diff(np.r_[False, supported, False])).reshape(-1, 2):
            if b-a >= minimum:
                result[(slice(a,b),)+col] = band_limited_signal(y[(slice(a,b),)+col], t[a:b])
    return result


class RigidReference:
    """Translate a candidate footprint over complete native noise supports.

    Exact section offsets are retained, including gaps. Temporal offsets and
    inclusive durations are rounded to the reference's native frames, never
    interpolated. Windows crossing a missing reference sample are excluded.
    """
    def __init__(self, banks):
        self.banks = []
        for name, y, t in banks:
            y = np.asarray(y, float); t = np.asarray(t, float)
            if y.ndim != 3 or len(y) != len(t) or len(t) < 3:
                raise ValueError('Expected rigid time x wall x section arrays')
            dt = float(np.median(np.diff(t)))
            if not np.allclose(np.diff(t), dt, rtol=1e-5, atol=1e-9):
                raise ValueError('Reference time must be uniform')
            good = np.isfinite(y)
            energy = np.concatenate([np.zeros((1,)+y.shape[1:]),
                                     np.cumsum(np.where(good, y*y, 0), axis=0)])
            missing = np.concatenate([np.zeros((1,)+y.shape[1:], int),
                                      np.cumsum(~good, axis=0)])
            self.banks.append((name, energy, missing, dt))

    def maximum(self, sections, starts_s, ends_s):
        sections = np.asarray(sections, int)
        offsets = sections-sections[0]
        origin = float(np.min(starts_s))
        maximum = None; count = 0; by_bank = {}
        for name, energy, missing, dt in self.banks:
            lo = np.rint((np.asarray(starts_s)-origin)/dt).astype(int)
            hi = np.rint((np.asarray(ends_s)-origin)/dt).astype(int)+1
            duration = hi-lo
            time_count = len(energy)-int(hi.max())
            spatial_count = energy.shape[2]-int(offsets.max())
            if min(time_count, spatial_count) < 1:
                continue
            start = np.arange(time_count)[:, None, None]
            column = np.arange(spatial_count)[None, :, None]+offsets[None, None, :]
            values = []
            for wall in range(energy.shape[1]):
                sums = energy[start+hi, wall, column]-energy[start+lo, wall, column]
                complete = (missing[start+hi, wall, column]-missing[start+lo, wall, column] == 0).all(2)
                rms = np.sqrt(np.maximum(sums, 0)/duration)
                statistic = np.median(rms, axis=2)[complete]
                if len(statistic):
                    values.append(statistic)
            if values:
                pooled = np.concatenate(values)
                value = float(pooled.max()); count += len(pooled)
                by_bank[name] = dict(maximum=value, matched_windows=len(pooled))
                maximum = value if maximum is None else max(maximum, value)
        return dict(maximum_rms=maximum, matched_windows=count, calibration_banks=by_bank)


def score_activity(event, source_evidence, filtered, valid, time_s, reference):
    """Use canonical source identity so a small split cannot replace full support."""
    entity = source_evidence[event['canonical_source_entity_id']]
    ids = np.array(sorted(int(c) for c in entity['profile']), int)
    if not len(ids) or np.any(np.diff(ids) <= 0):
        raise ValueError('Expected ordered nonempty canonical activity sections')
    t = np.asarray(time_s, float); y = np.asarray(filtered, float); m = np.asarray(valid, bool)
    if y.shape != m.shape or y.shape[0] != len(t) or y.ndim != 2:
        raise ValueError('Expected one wall and matching time/section validity')
    rms = []; starts = []; ends = []
    for col in ids:
        phases = entity['profile'].get(str(col), entity['profile'].get(int(col)))
        start = min(p['start_s'] for p in phases); end = max(p['end_s'] for p in phases)
        a = int(np.searchsorted(t, start-1e-8)); b = int(np.searchsorted(t, end+1e-8, side='right'))
        if b <= a or not m[a:b, col].all() or not np.isfinite(y[a:b, col]).all():
            return dict(event_proposal_id=event['event_proposal_id'], decision='unscored_missing_support',
                        activity_rms=None, strength_ratio=None, event_count=None, event_frequency=None)
        rms.append(float(np.sqrt(np.mean(y[a:b, col]**2))))
        starts.append(float(t[a])); ends.append(float(t[b-1]))
    activity = float(np.median(rms))
    null = reference.maximum(ids, starts, ends)
    floor = null['maximum_rms']
    ratio = activity/floor if floor is not None and floor > 0 else None
    decision = ('unscored_no_matched_reference' if floor is None else
                'above_rigid_reference' if activity > floor else 'not_above_rigid_reference')
    return dict(event_proposal_id=event['event_proposal_id'], canonical_source=entity['entity_id'],
        decision=decision, activity_rms=activity, per_section_rms=rms, sections=ids.tolist(),
        phase_start_s=starts, phase_end_s=ends, strength_ratio=ratio,
        matched_rigid_reference=null, direction=event['direction'],
        ownership_status=event['ownership_status'],
        prominent_propagation_proposal=decision=='above_rigid_reference' and event['direction'] in ('C2F','F2C'),
        event_count=None, event_frequency=None)
