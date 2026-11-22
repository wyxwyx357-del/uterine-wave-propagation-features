"""Experimental P3-guided outer; supplied inner tracks are never modified."""
import numpy as np
from ._lk import match, local_contrast


def polyline_unit_tangent(points):
    tangent = np.gradient(np.asarray(points, float), axis=-2)
    length = np.linalg.norm(tangent, axis=-1, keepdims=True)
    return np.divide(tangent, length, out=np.full_like(tangent, np.nan), where=length > 1e-8)


def radial_reference_from_wall_tracks(wall, offset=21.):
    """wall: time x wall(2) x section x xy; reference is not measurement."""
    wall = np.asarray(wall, float)
    if wall.ndim != 4 or wall.shape[1] != 2 or wall.shape[-1] != 2 or wall.shape[2] < 2:
        raise ValueError('Expected time x 2 walls x sections x xy')
    tangent = polyline_unit_tangent(wall)
    normal = np.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)
    away = wall - wall[:, ::-1]
    normal *= np.where(np.sum(normal*away, axis=-1) >= 0, 1., -1.)[..., None]
    return wall + offset*normal, tangent


def adjacent_radial_pair_crossing_risk(inner, outer, epsilon=1e-6):
    """Same segment-intersection test as the legacy P3 radial implementation."""
    p, q = np.asarray(inner), np.asarray(outer)
    a, b, c, d = p[..., :-1, :], q[..., :-1, :], p[..., 1:, :], q[..., 1:, :]
    def cross(x, y):
        return x[..., 0]*y[..., 1]-x[..., 1]*y[..., 0]
    finite = np.isfinite(np.stack((a, b, c, d))).all(axis=(0, -1))
    overlap = (np.maximum(np.minimum(a, b), np.minimum(c, d)) <=
               np.minimum(np.maximum(a, b), np.maximum(c, d))+epsilon).all(-1)
    hit = finite & overlap & (cross(b-a, c-a)*cross(b-a, d-a) <= epsilon) & (cross(d-c, a-c)*cross(d-c, b-c) <= epsilon)
    result = np.zeros(p.shape[:-1], bool)
    result[..., :-1] |= hit
    result[..., 1:] |= hit
    return result


def track_p3_guided_outer(gray, baseline, grid, wall_reference, *, reference_source):
    """Return a separate track dictionary plus both candidates' diagnostics.

    No duration reset, curve snapping, tangential threshold or spatial filling.
    Failed frames remain invalid; recovery starts a new segment at the measured
    target, with its incoming step excluded. Coordinates are analysis pixels.
    """
    t = np.asarray(baseline['time_s']); poses = np.asarray(baseline['poses'])
    wall = np.asarray(wall_reference, float)
    ids = np.asarray(grid['A_pair_ids']); inner_ids, outer_ids = ids[..., 0], ids[..., 1]
    shape = (len(t), *inner_ids.shape)
    if wall.shape != (*shape, 2) or len(gray) != len(t) or not reference_source:
        raise ValueError('Expected matching frames, time and explicit P3-reference provenance')
    reference, tangent = radial_reference_from_wall_tracks(wall)
    out = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in baseline.items()}
    for key in ('positions', 'corrected'):
        out[key][:, outer_ids] = np.nan
    out['target_valid'][:, outer_ids] = False
    out['step_valid'][:, outer_ids] = False
    out['reanchor'][:, outer_ids] = False
    out['segment_id'][:, outer_ids] = -1
    out['target_ncc'][:, outer_ids] = np.nan
    out['target_fb'][:, outer_ids] = np.nan
    out['recovery_attempted'][:, outer_ids] = False
    out['positions'][0, outer_ids] = baseline['positions'][0, outer_ids]
    out['target_valid'][0, outer_ids] = baseline['target_valid'][0, outer_ids]
    out['segment_id'][0, outer_ids] = 0
    out['reject_reason'][:, outer_ids] = 'waiting'
    sid = np.zeros(inner_ids.shape, int)
    diag = {key: np.full((len(t), 2, *inner_ids.shape), np.nan) for key in
            ('ncc', 'fb', 'reference_distance', 'tangential_slip')}
    for key in ('lk_qc_valid', 'lk_forward_status', 'lk_backward_status', 'orientation', 'crossing'):
        diag[key] = np.zeros((len(t), 2, *inner_ids.shape), bool)
    diag['candidate_reject_reason'] = np.full((len(t), 2, *inner_ids.shape), 'not_attempted', dtype='U28')
    diag['candidate_positions'] = np.full((len(t), 2, *inner_ids.shape, 2), np.nan)
    diag['chosen_branch'] = np.full(shape, -1, int)
    diag['selected_crossing'] = np.zeros(shape, bool)
    diag['selected_flip'] = np.zeros(shape, bool)
    diag['reference_outer'] = reference
    epoch = np.asarray(baseline.get('pose_epoch', np.zeros(len(t), int)))
    previous = local_contrast(gray[0], sigma=5.)
    for i in range(1, len(t)):
        enhanced = local_contrast(gray[i], sigma=5.)
        prior_good = out['target_valid'][i-1, outer_ids]
        out['recovery_attempted'][i, outer_ids] = ~prior_good
        source = np.where(prior_good[..., None], out['positions'][i-1, outer_ids], reference[i-1])
        transform = np.linalg.inv(poses[i])@poses[i-1]
        motion_guess = source@transform[:2, :2].T+transform[:2, 2]
        inner = baseline['positions'][i, inner_ids]
        inner_good = baseline['target_valid'][i, inner_ids] & np.isfinite(inner).all(-1)
        # Current-frame inner residual, rather than adding inner motion twice.
        inherited = inner-wall[i]
        guided_guess = reference[i]+inherited
        usable = (np.isfinite(source).all(-1) & np.isfinite(motion_guess).all(-1) &
                  np.isfinite(guided_guess).all(-1) & inner_good & np.isfinite(tangent[i]).all(-1))
        for branch, guess in enumerate((motion_guess, guided_guess)):
            flat = np.flatnonzero(usable.ravel())
            if len(flat):
                q, good, ncc, fb, reason, lk = match(previous, enhanced, gray[i-1], gray[i],
                    source.reshape(-1, 2)[flat].astype(np.float32), 21,
                    guess.reshape(-1, 2)[flat].astype(np.float32), max_fb_px=.75, return_diagnostics=True)
                wi, ci = np.unravel_index(flat, inner_ids.shape)
                diag['candidate_positions'][i, branch, wi, ci] = q
                diag['lk_qc_valid'][i, branch, wi, ci] = good
                diag['lk_forward_status'][i, branch, wi, ci] = lk['forward_status']
                diag['lk_backward_status'][i, branch, wi, ci] = lk['backward_status']
                diag['candidate_reject_reason'][i, branch, wi, ci] = reason
                diag['ncc'][i, branch, wi, ci] = ncc
                diag['fb'][i, branch, wi, ci] = fb
            q = diag['candidate_positions'][i, branch]
            delta = q-reference[i]
            diag['reference_distance'][i, branch] = np.linalg.norm(delta, axis=-1)
            diag['tangential_slip'][i, branch] = np.abs(np.sum(delta*tangent[i], axis=-1))
            diag['orientation'][i, branch] = np.sum((q-inner)*(reference[i]-wall[i]), axis=-1)>0
            supported_q = np.where(diag['lk_qc_valid'][i, branch, ..., None], q, np.nan)
            diag['crossing'][i, branch] = adjacent_radial_pair_crossing_risk(inner, supported_q)
        selected = np.full((*inner_ids.shape, 2), np.nan)
        accepted = np.zeros(inner_ids.shape, bool)
        for wi, ci in np.ndindex(inner_ids.shape):
            options = [b for b in (0, 1) if diag['lk_qc_valid'][i, b, wi, ci]]
            if not options:
                continue
            def rank(b):
                return (bool(diag['crossing'][i, b, wi, ci]),
                        not bool(diag['orientation'][i, b, wi, ci]),
                        diag['tangential_slip'][i, b, wi, ci],
                        -diag['ncc'][i, b, wi, ci], diag['fb'][i, b, wi, ci], b)
            b = min(options, key=rank)
            diag['chosen_branch'][i, wi, ci] = b
            selected[wi, ci] = diag['candidate_positions'][i, b, wi, ci]
            accepted[wi, ci] = diag['orientation'][i, b, wi, ci] and not diag['crossing'][i, b, wi, ci]
            diag['selected_flip'][i, wi, ci] = not diag['orientation'][i, b, wi, ci]
        # Mixed-branch selection can create a new crossing; check the final set too.
        crossing = adjacent_radial_pair_crossing_risk(inner, np.where(accepted[..., None], selected, np.nan))
        diag['selected_crossing'][i] = crossing | np.any(diag['crossing'][i], axis=0)
        accepted &= ~crossing & inner_good
        for wi, ci in np.ndindex(inner_ids.shape):
            pid = outer_ids[wi, ci]; b = diag['chosen_branch'][i, wi, ci]
            if not accepted[wi, ci]:
                out['reject_reason'][i, pid] = ('inner_invalid' if not inner_good[wi, ci] else
                    'orientation' if diag['selected_flip'][i, wi, ci] else
                    'crossing' if diag['selected_crossing'][i, wi, ci] else 'lk_qc_or_reference')
                continue
            continuous = prior_good[wi, ci] and epoch[i] == epoch[i-1]
            if not continuous:
                sid[wi, ci] += 1
            out['positions'][i, pid] = selected[wi, ci]
            out['target_valid'][i, pid] = True
            out['step_valid'][i, pid] = continuous
            out['segment_id'][i, pid] = sid[wi, ci]
            out['reanchor'][i, pid] = not continuous
            out['target_ncc'][i, pid] = diag['ncc'][i, b, wi, ci]
            out['target_fb'][i, pid] = diag['fb'][i, b, wi, ci]
            out['reject_reason'][i, pid] = 'accepted' if continuous else 'new_segment'
        previous = enhanced
    out['corrected'] = np.einsum('tni,tji->tnj', out['positions'].astype(float), poses[:, :2, :2])+poses[:, None, :2, 2]
    # Keep the unchanged inner and other existing coordinates bit-for-bit.
    other = np.ones(out['positions'].shape[1], bool); other[outer_ids.ravel()] = False
    out['corrected'][:, other] = baseline['corrected'][:, other]
    out['p3_outer_diagnostics'] = diag
    out['p3_reference_source'] = reference_source
    return out
