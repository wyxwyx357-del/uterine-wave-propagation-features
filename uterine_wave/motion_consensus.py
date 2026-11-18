"""Rigid agreement among distributed reference points; diagnostic only."""
import itertools
import numpy as np


def rigid_fit(source, target):
    """Least-squares rotation/translation, excluding scale and shear."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    center_a, center_b = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source-center_a).T @ (target-center_b))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    result = np.eye(3)
    result[:2, :2] = rotation
    result[:2, 2] = center_b - rotation @ center_a
    return result


def rigid_consensus(source, target, tolerance_px=1., max_trials=128):
    """RANSAC-style agreement; majority is not a wave acceptance criterion.

    Inputs are corresponding finite image coordinates. Tolerance is a spatial
    model agreement setting, not a physiological amplitude/noise threshold.
    """
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.ndim != 2 or source.shape[1] != 2 or target.shape != source.shape:
        raise ValueError('Expected corresponding Nx2 point arrays')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Expected finite point correspondences')
    if tolerance_px <= 0 or max_trials < 1:
        raise ValueError('Expected positive agreement tolerance and trial count')
    count = len(source)
    result = dict(matrix=np.full((3, 3), np.nan), inliers=np.zeros(count, bool),
                  residual_px=np.full(count, np.nan), valid_points=count,
                  inlier_points=0, inlier_fraction=0., majority_found=False)
    if count < 3:
        return result
    pairs = np.array(list(itertools.combinations(range(count), 2)))
    if len(pairs) > max_trials:
        pairs = pairs[np.random.default_rng(0).choice(len(pairs), max_trials, replace=False)]
    best, best_score = None, (-1, -np.inf)
    for pair in pairs:
        if np.linalg.norm(source[pair[1]]-source[pair[0]]) < 4:
            continue
        model = rigid_fit(source[pair], target[pair])
        residual = np.linalg.norm(source @ model[:2, :2].T + model[:2, 2] - target, axis=1)
        keep = residual <= tolerance_px
        if not keep.any():
            continue
        score = (int(keep.sum()), -float(np.median(residual[keep])))
        if score > best_score:
            best, best_score = keep, score
    if best is None or best.sum() < 3:
        return result
    for _ in range(2):
        model = rigid_fit(source[best], target[best])
        residual = np.linalg.norm(source @ model[:2, :2].T + model[:2, 2] - target, axis=1)
        refined = residual <= tolerance_px
        if refined.sum() < 3:
            break
        best = refined
    # Membership and residual must describe this same final fitted transform.
    model = rigid_fit(source[best], target[best])
    residual = np.linalg.norm(source @ model[:2, :2].T + model[:2, 2] - target, axis=1)
    keep = residual <= tolerance_px
    return dict(matrix=model, inliers=keep, residual_px=residual, valid_points=count,
                inlier_points=int(keep.sum()), inlier_fraction=float(keep.mean()),
                majority_found=bool(keep.sum() > count/2))
