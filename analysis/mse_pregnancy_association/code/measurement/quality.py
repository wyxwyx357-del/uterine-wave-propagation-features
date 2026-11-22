"""Describe saved native-step PCC without changing tracking or event decisions.

The existing mean-centered NCC is a patch Pearson correlation. It is only
computed after flow/FB checks and accepted tracks are already NCC-selected.
Means below are conditional on supported native steps, not an independent
reproduction of Huang's full-recording quality screen.
"""
import numpy as np
from .deformation import validate_time


LITERATURE_PCC_REFERENCE = .8  # Huang 2022 recording screen; descriptive only.


def _pair_ids(grid, point_count):
    ids = np.asarray(grid['A_pair_ids'])
    if (ids.ndim != 3 or ids.shape[0] != 2 or ids.shape[1] == 0 or ids.shape[-1] != 2 or
            not np.issubdtype(ids.dtype, np.integer) or ids.min() < 0 or ids.max() >= point_count):
        raise ValueError('Expected two-wall radial pair IDs indexing saved tracks')
    return ids


def native_pcc(tracks):
    """Exclude seeds, recovery jumps and segment/pose-epoch transitions.

    Rejected and uncomputed scores are missing, never zero or the -1 sentinel.
    A finite score at a re-anchor may compare nonadjacent frames; it is excluded.
    The first confirmed native step after that seed can be included.
    """
    q = np.asarray(tracks['target_ncc'], float)
    if q.ndim != 2:
        raise ValueError('target_ncc must be time x point')
    validate_time(tracks['time_s'], len(q), 1)
    valid = np.asarray(tracks['target_valid'], bool)
    step = np.asarray(tracks['step_valid'], bool)
    segment = np.asarray(tracks['segment_id'])
    reanchor = np.asarray(tracks['reanchor'], bool)
    if any(x.shape != q.shape for x in (valid, step, segment, reanchor)):
        raise ValueError('PCC and track support must share a shape')
    epoch = np.asarray(tracks['pose_epoch'])
    if epoch.shape != (len(q),):
        raise ValueError('pose_epoch must match time')
    use = np.zeros(q.shape, bool)
    use[1:] = (step[1:] & valid[1:] & valid[:-1] & ~reanchor[1:] &
               (segment[1:] == segment[:-1]) & (segment[1:] >= 0) &
               (epoch[1:] == epoch[:-1])[:, None])
    use &= np.isfinite(q) & (q > -1) & (q <= 1)
    return np.where(use, q, np.nan)


def _statistics(values):
    values = np.asarray(values, float)
    measured = values[np.isfinite(values)]
    return dict(measured_samples=int(len(measured)),
                mean=float(measured.mean()) if len(measured) else None,
                p05=float(np.quantile(measured, .05)) if len(measured) else None,
                minimum=float(measured.min()) if len(measured) else None,
                fraction_below_literature_reference=float(np.mean(measured < LITERATURE_PCC_REFERENCE))
                if len(measured) else None)


def _group(q, tracks, ids):
    ids = np.asarray(ids, int)
    values = q[1:, ids]
    counts = np.isfinite(values).sum(0)
    means = np.divide(np.nansum(values, axis=0), counts,
                      out=np.full(len(ids), np.nan), where=counts > 0)
    supported = means[np.isfinite(means)]
    reasons, count = np.unique(np.asarray(tracks['reject_reason'])[1:, ids], return_counts=True)
    return dict(**_statistics(values), requested_samples=int(values.size),
                pcc_coverage_fraction=float(np.isfinite(values).mean()),
                valid_native_step_fraction=float(np.asarray(tracks['step_valid'])[1:, ids].mean()),
                point_mean_average=float(supported.mean()) if len(supported) else None,
                requested_points=len(ids), supported_points=len(supported),
                reason_counts={str(k): int(v) for k, v in zip(reasons, count)})


def tracking_quality(tracks, grid):
    """Report conditional PCC and coverage, separately for each wall/inner/outer."""
    q = native_pcc(tracks)
    ids = _pair_ids(grid, q.shape[1])
    walls = [dict(wall=wall, radial=_group(q, tracks, np.unique(ids[wall])),
                  inner=_group(q, tracks, ids[wall, :, 0]),
                  outer=_group(q, tracks, ids[wall, :, 1])) for wall in (0, 1)]
    return dict(metric='mean_centered_patch_NCC_equals_Pearson_PCC',
                population='supported_native_steps_only_after_existing_LK_FB_NCC_selection',
                literature_reference=LITERATURE_PCC_REFERENCE,
                reference_role='descriptive_not_an_exclusion_or_wave_threshold',
                native_dt_s=float(np.median(np.diff(tracks['time_s']))),
                all_points=_group(q, tracks, np.arange(q.shape[1])),
                radial_points=_group(q, tracks, np.unique(ids)), walls=walls,
                independent_recording_quality_pass=None,
                note='Missing checks, seeds and re-anchor jumps are not assigned correlation values.')


def radial_quality_maps(tracks, grid, spacetime):
    """Summarize native-step PCC over the exact source intervals of radial A.

    For L(end)-L(start), include native transitions start+1 through end for
    both members of each pair. Missing quality remains missing; coverage and
    the unchanged deformation mask are separate fields. No interpolation.
    """
    q = native_pcc(tracks)
    t = np.asarray(tracks['time_s'], float)
    hop = int(spacetime['hop'])
    validate_time(t, len(q), hop)
    ids = _pair_ids(grid, q.shape[1])
    size = len(t)-hop
    mask = np.asarray(spacetime['valid_mask'], bool)
    if mask.shape != (size, 2, len(spacetime['s'])) or ids.shape != (2, mask.shape[2], 2):
        raise ValueError('Radial pairs and spacetime mask differ')
    if (np.shape(spacetime['interval_start_s']) != (size,) or
            np.shape(spacetime['interval_end_s']) != (size,) or
            not np.allclose(spacetime['interval_start_s'], t[:-hop], rtol=0, atol=1e-9) or
            not np.allclose(spacetime['interval_end_s'], t[hop:], rtol=0, atol=1e-9)):
        raise ValueError('Quality intervals must use the same native source times as radial A')
    pair = q[:, ids]
    finite = np.isfinite(pair)
    sums = np.concatenate((np.zeros((1, *mask.shape[1:])),
                           np.cumsum(np.where(finite, pair, 0).sum(-1), axis=0)))
    counts = np.concatenate((np.zeros((1, *mask.shape[1:]), int),
                             np.cumsum(finite.sum(-1), axis=0)))
    total = sums[hop+1:]-sums[1:size+1]
    count = counts[hop+1:]-counts[1:size+1]
    mean = np.divide(total, count, out=np.full(mask.shape, np.nan), where=count > 0)
    native_min = np.where(finite, pair, np.inf).min(-1)
    minimum = np.full(mask.shape, np.inf)
    for offset in range(1, hop+1):
        np.minimum(minimum, native_min[offset:offset+size], out=minimum)
    minimum[count == 0] = np.nan
    return dict(interval_pcc_mean=mean, interval_pcc_min=minimum,
                interval_pcc_coverage_fraction=count/(2*hop),
                deformation_valid_mask=mask.copy(),
                interval_start_s=np.asarray(spacetime['interval_start_s']).copy(),
                interval_end_s=np.asarray(spacetime['interval_end_s']).copy(),
                interval_center_s=np.asarray(spacetime['interval_center_s']).copy(),
                s=np.asarray(spacetime['s']).copy())


def candidate_quality(quality_maps, candidates):
    """Attach QC to existing candidate bounding regions; do not confirm waves.

    Interval means overlap in time and are descriptive, not independent
    observations. Candidate identities, directions and counts are unchanged.
    """
    t = quality_maps['interval_center_s']
    result = []
    for wall in candidates['walls']:
        for event in wall['events']:
            rows = np.flatnonzero((t >= event['start_s']) & (t <= event['end_s']))
            cols = np.asarray(event['sections'], int)
            index = np.ix_(rows, [wall['wall']], cols)
            valid = quality_maps['deformation_valid_mask'][index]
            values = quality_maps['interval_pcc_mean'][index]
            coverage = quality_maps['interval_pcc_coverage_fraction'][index]
            statistics = _statistics(np.where(valid, values, np.nan))
            minima = quality_maps['interval_pcc_min'][index]
            result.append(dict(event_id=event['event_id'], wall=wall['wall'],
                classification=event['classification'], quality_flags=event['quality_flags'],
                interval_pcc=statistics,
                interval_minimum_pcc=_statistics(np.where(valid, minima, np.nan)),
                mean_pcc_coverage_on_valid_intervals=float(coverage[valid].mean()) if valid.any() else None,
                analyzed_interval_samples=int(valid.sum()),
                candidate_start_s=event['start_s'], candidate_end_s=event['end_s'],
                source_start_s=float(quality_maps['interval_start_s'][rows[0]]) if len(rows) else None,
                source_end_s=float(quality_maps['interval_end_s'][rows[-1]]) if len(rows) else None,
                confirmed_event=False))
    return result
