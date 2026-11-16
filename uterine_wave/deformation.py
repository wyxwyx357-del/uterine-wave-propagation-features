"""Signed interval deformation with explicit topology and continuity masks."""
import numpy as np


def validate_time(time_s, count, hop):
    t = np.asarray(time_s, float)
    if t.shape != (count,) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Expected finite, strictly increasing acquisition times")
    if not isinstance(hop, (int, np.integer)) or not 1 <= hop < count:
        raise ValueError("hop must be an integer between 1 and frame_count-1")
    return t


def interval_support(tracks, hop):
    """No invalid step, segment change or correction-epoch change may be crossed."""
    valid = np.asarray(tracks['target_valid'], bool)
    if valid.ndim != 2:
        raise ValueError("target_valid must be time x point")
    validate_time(tracks['time_s'], len(valid), hop)
    step = np.asarray(tracks['step_valid'], bool).copy()
    segment = np.asarray(tracks['segment_id'])
    if step.shape != valid.shape or segment.shape != valid.shape:
        raise ValueError("Track validity and segment arrays must share a shape")
    epoch = np.asarray(tracks.get('pose_epoch', np.zeros(len(valid), int)))
    if epoch.shape != (len(valid),):
        raise ValueError("pose_epoch must have one entry per frame")
    step[1:] &= valid[1:] & valid[:-1]
    step[1:] &= segment[1:] == segment[:-1]
    step[1:] &= (epoch[1:] == epoch[:-1])[:, None]
    failures = np.vstack((np.zeros((1, valid.shape[1]), int), np.cumsum(~step[1:], axis=0)))
    return valid[hop:] & valid[:-hop] & (failures[hop:] - failures[:-hop] == 0)


def rate_A(points, grid, time_s, hop=8):
    """(L(t)-L(t-hop))/L(t-hop)/dt, positive for radial lengthening; s^-1."""
    points = np.asarray(points, float)
    t = validate_time(time_s, len(points), hop)
    p = points[:, grid['A_pair_ids']]
    lengths = np.linalg.norm(p[:, :, :, 1] - p[:, :, :, 0], axis=-1)
    result = np.full_like(lengths[hop:], np.nan)
    denominator = lengths[:-hop] * (t[hop:] - t[:-hop])[:, None, None]
    np.divide(lengths[hop:] - lengths[:-hop], denominator, out=result, where=denominator > 0)
    return result


def curvature(points, arc):
    """Signed curvature of a local 5-seed quadratic in initial arc coordinates."""
    points = np.asarray(points, float)
    arc = np.asarray(arc, float)
    if points.ndim != 3 or points.shape[2] != 2 or points.shape[1] != len(arc):
        raise ValueError("Expected time x spatial_point x xy and matching arc")
    if len(arc) < 5 or not np.isfinite(arc).all() or np.any(np.diff(arc) <= 0):
        raise ValueError("Curvature needs five distinct ordered spatial samples")
    out = np.full((len(points), len(arc) - 4), np.nan)
    for j in range(2, len(arc) - 2):
        s = arc[j-2:j+3] - arc[j]
        inverse = np.linalg.pinv(np.column_stack((np.ones(5), s, s*s)))
        coef = np.einsum('ij,tjk->tik', inverse, points[:, j-2:j+3])
        v, a = coef[:, 1], 2*coef[:, 2]
        denominator = np.linalg.norm(v, axis=1)**3
        numerator = v[:, 0]*a[:, 1] - v[:, 1]*a[:, 0]
        np.divide(numerator, denominator, out=out[:, j-2], where=denominator > 1e-20)
    return out


def rate_B(points, grid, time_s, hop=8):
    """Curvature change per second; (processed pixel * second)^-1."""
    t = validate_time(time_s, len(points), hop)
    values = []
    arcs = []
    for ids, arc in zip(grid['B_curve_ids'], grid['B_arc_positions']):
        kappa = curvature(np.asarray(points)[:, ids], arc)
        values.append((kappa[hop:] - kappa[:-hop]) / (t[hop:] - t[:-hop])[:, None])
        arcs.append(arc[2:-2])
    return np.stack(values, axis=1), np.asarray(arcs)


def compute_maps(tracks, grid, hop=8):
    """A IDs address A seeds, B IDs address B seeds; tracking order is [A, B]."""
    points = np.asarray(tracks['corrected'], float)
    count_a, count_b = len(grid['A_points']), len(grid['B_points'])
    if points.shape != (len(tracks['time_s']), count_a + count_b, 2):
        raise ValueError("Expected corrected trajectories in explicit [A_points, B_points] order")
    effective = dict(tracks)
    effective['target_valid'] = np.asarray(tracks['target_valid'], bool) & np.isfinite(points).all(2)
    support = interval_support(effective, hop)
    if support.shape[1] != count_a + count_b:
        raise ValueError("Validity shape differs from trajectory shape")
    support &= np.isfinite(points[hop:]).all(2) & np.isfinite(points[:-hop]).all(2)
    a = rate_A(points[:, :count_a], grid, tracks['time_s'], hop)
    good_a = support[:, :count_a][:, grid['A_pair_ids']].all(3) & np.isfinite(a)
    b, arcs = rate_B(points[:, count_a:], grid, tracks['time_s'], hop)
    good_b = np.stack([
        np.stack([support[:, count_a:][:, ids[j-2:j+3]].all(1)
                  for j in range(2, len(ids)-2)], axis=1)
        for ids in grid['B_curve_ids']
    ], axis=1) & np.isfinite(b)
    return dict(A=np.where(good_a, a, np.nan), B=np.where(good_b, b, np.nan),
                A_valid=good_a, B_valid=good_b, time_s=np.asarray(tracks['time_s'])[hop:],
                A_arc_positions=grid['arc_positions'], B_arc_positions=arcs)
