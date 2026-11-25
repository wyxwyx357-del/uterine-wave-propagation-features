"""Known image warps on a static texture; this is not acoustic resimulation."""
import numpy as np
from .geometry import layout
from .deformation import compute_maps
from .spectrum import describe

CONDITIONS = ('static', 'rigid', 'local_once', 'C2F', 'F2C')


def synthetic_geometry():
    """Artificial anatomy, with coordinates already in processed-image pixels."""
    x = np.linspace(65, 355, 33)
    mid = 170 + 9*np.sin((x-65)/290*np.pi)
    return dict(ant_wall=np.column_stack((x, mid-24)),
                post_wall=np.column_stack((x, mid+24)),
                cervix_point=np.array([[65., 170.]]),
                fundus_point=np.array([[355., 170.]]))


def synthetic_texture(seed=20261116):
    """Deterministic speckle-like image for software checks, not an ultrasound phantom."""
    import cv2
    rng = np.random.default_rng(seed)
    image = cv2.GaussianBlur(rng.normal(120, 45, (340, 420)).astype(np.float32), (0, 0), .8)
    return np.uint8(np.clip(np.rint(image), 0, 255))


def known_motion(gray, shapes, grid, time_s, condition):
    """Return warped frames, corrected-coordinate truth and exact global poses.

    Field y'=y+a(x,t)+b(x,t)*(y-center_y(x)); x'=x. The reference geometry
    defines this test field. It does not provide independent physiological truth.
    """
    import cv2
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown control {condition}")
    t = np.asarray(time_s, float)
    if t.ndim != 1 or len(t) < 2 or t[0] != 0 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Controls require increasing real times starting at zero")
    points = np.vstack((grid['A_points'], grid['B_points'])).astype(np.float32)
    midpoint = grid['wall'].mean(0)
    order = np.argsort(midpoint[:, 0])
    center_x, center_y = midpoint[order, 0], midpoint[order, 1]
    cx, fx = float(shapes['cervix_point'][0, 0]), float(shapes['fundus_point'][0, 0])
    if abs(fx-cx) < 1e-8 or np.any(np.diff(center_x) <= 0):
        raise ValueError("This control field requires a nonvertical, single-valued x path")
    yy, xx = np.indices(gray.shape, dtype=np.float32)

    def field(x, time):
        s = (x-cx)/(fx-cx)
        center = np.interp(np.asarray(x).ravel(), center_x, center_y).reshape(np.shape(x))
        if condition in ('C2F', 'F2C'):
            sign = 1 if condition == 'C2F' else -1
            phase = 2*np.pi*(2*time/60-sign*s)
            initial = -sign*2*np.pi*s
            w = np.sin(phase)-np.sin(initial)
            a, b = 2*w, .03*w
        elif condition == 'local_once':
            w = np.exp(-.5*((s-.5)/.12)**2)*np.sin(np.pi*time/60)**2
            a, b = 3*w, .03*w
        else:
            a, b = np.zeros_like(x, dtype=float), np.zeros_like(x, dtype=float)
        return a, b, center

    frames, truth = [], []
    poses = np.repeat(np.eye(3)[None], len(t), axis=0)
    for i, time in enumerate(t):
        dx = 3*np.sin(2*np.pi*time/30) if condition == 'rigid' else 0.
        dy = 2*(np.cos(2*np.pi*time/30)-1) if condition == 'rigid' else 0.
        x = xx-dx
        a, b, c = field(x, time)
        map_y = ((yy-dy-a+b*c)/(1+b)).astype(np.float32)
        frames.append(cv2.remap(gray, x.astype(np.float32), map_y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101))
        a, b, c = field(points[:, 0], time)
        q = points.astype(float).copy()
        q[:, 1] += a+b*(q[:, 1]-c)
        truth.append(q)
        poses[i, :2, 2] = [-dx, -dy]
    return frames, np.asarray(truth), poses


def reference_tracks(points, time_s):
    shape = np.asarray(points).shape[:2]
    step = np.ones(shape, bool)
    step[0] = False
    return dict(corrected=points, target_valid=np.ones(shape, bool),
                step_valid=step, segment_id=np.zeros(shape, int), time_s=time_s)


def evaluate(tracks, truth, grid, hop=8):
    measured = compute_maps(tracks, grid, hop)
    reference = compute_maps(reference_tracks(truth, tracks['time_s']), grid, hop)
    valid = tracks['target_valid'] & np.isfinite(tracks['corrected']).all(2)
    error = np.linalg.norm(tracks['corrected']-truth, axis=2)
    report = dict(point_count=truth.shape[1], point_valid_fraction=float(valid.mean()),
                  point_rmse_px=float(np.sqrt(np.mean(error[valid]**2))) if valid.any() else None)
    for name in ('A', 'B'):
        report['route_'+name] = [
            describe(measured[name][:, i], reference[name][:, i], measured[name+'_valid'][:, i],
                     measured['time_s'], measured[name+'_arc_positions'][i])
            for i in range(measured[name].shape[1])]
    return report


def run_controls(gray, shapes, *, time_s=None, conditions=CONDITIONS, hop=8, progress=None):
    from .tracking import track_segments
    grid = layout(shapes)
    time_s = np.arange(501)*.12 if time_s is None else np.asarray(time_s, float)
    points = np.vstack((grid['A_points'], grid['B_points']))
    results = []
    for condition in conditions:
        if progress is not None:
            progress(condition)
        frames, truth, poses = known_motion(gray, shapes, grid, time_s, condition)
        tracks = track_segments(frames, points, time_s, poses)
        report = evaluate(tracks, truth, grid, hop)
        report['condition'] = condition
        results.append(report)
    return results
