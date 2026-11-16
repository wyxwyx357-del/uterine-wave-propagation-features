"""Prototype per-wave candidate detection on deformation time-space maps.

This module deliberately separates "there is a candidate traveling event" from
the directional spectral ER. Defaults are V0 engineering rules for method
development, not validated physiological thresholds.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class EventDetectionConfig:
    z_threshold: float = 3.0
    min_component_cells: int = 6
    min_space_points: int = 4
    min_path_fraction: float = 0.15
    min_fit_r2: float = 0.50
    synchronous_tolerance_s: float = 0.50

    def validate(self):
        if not np.isfinite(self.z_threshold) or self.z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        if self.min_component_cells < 1 or self.min_space_points < 2:
            raise ValueError("component and spatial support thresholds are invalid")
        if not 0 <= self.min_path_fraction <= 1:
            raise ValueError("min_path_fraction must lie in [0,1]")
        if not 0 <= self.min_fit_r2 <= 1:
            raise ValueError("min_fit_r2 must lie in [0,1]")
        if self.synchronous_tolerance_s < 0:
            raise ValueError("synchronous_tolerance_s must be non-negative")
        return self


def _validate_map(signal, valid, time_s, space):
    y = np.asarray(signal, float)
    m = np.asarray(valid, bool)
    t = np.asarray(time_s, float)
    s = np.asarray(space, float)
    if y.ndim != 2 or y.shape != m.shape:
        raise ValueError("signal and valid must be matching time x space arrays")
    if y.shape != (len(t), len(s)) or len(t) < 3 or len(s) < 3:
        raise ValueError("coordinates must match a time x space map with >=3 samples per axis")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("time_s must be finite and strictly increasing")
    if not np.isfinite(s).all() or np.any(np.diff(s) <= 0):
        raise ValueError("space must be finite and strictly increasing cervix-to-fundus")
    m = m & np.isfinite(y)
    return y, m, t, s


def robust_activity(signal, valid):
    """Return absolute robust-z activity without filling missing samples.

    Each spatial column is centered and scaled independently. A column with
    zero/undefined robust scale remains NaN rather than receiving an arbitrary
    noise floor.
    """
    y = np.asarray(signal, float)
    m = np.asarray(valid, bool)
    out = np.full_like(y, np.nan, dtype=float)
    for j in range(y.shape[1]):
        good = m[:, j] & np.isfinite(y[:, j])
        if good.sum() < 3:
            continue
        x = y[good, j]
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.std(x))
        if not np.isfinite(scale) or scale <= 1e-12:
            continue
        out[good, j] = np.abs((x - med) / scale)
    return out


def _components(active):
    """8-connected components in a boolean time-space map."""
    active = np.asarray(active, bool)
    seen = np.zeros_like(active, bool)
    comps = []
    nt, ns = active.shape
    for ti, si in np.argwhere(active):
        if seen[ti, si]:
            continue
        stack = [(int(ti), int(si))]
        seen[ti, si] = True
        cells = []
        while stack:
            a, b = stack.pop()
            cells.append((a, b))
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    if da == 0 and db == 0:
                        continue
                    na, nb = a + da, b + db
                    if 0 <= na < nt and 0 <= nb < ns and active[na, nb] and not seen[na, nb]:
                        seen[na, nb] = True
                        stack.append((na, nb))
        comps.append(np.asarray(cells, int))
    return comps


def _fit_ridge(cells, activity, signal, time_s, space):
    """Collapse one connected component to one peak time per spatial column."""
    ridge = []
    for sj in np.unique(cells[:, 1]):
        rows = cells[cells[:, 1] == sj, 0]
        scores = activity[rows, sj]
        k = int(rows[np.nanargmax(scores)])
        ridge.append((int(sj), k, float(scores[np.nanargmax(scores)])))
    ridge.sort(key=lambda x: x[0])
    cols = np.asarray([r[0] for r in ridge], int)
    tids = np.asarray([r[1] for r in ridge], int)
    weights = np.asarray([r[2] for r in ridge], float)
    x = space[cols]
    tt = time_s[tids]
    amp = np.abs(signal[tids, cols])

    fit = dict(slope_s_per_space=None, intercept_s=None, fit_r2=None,
               speed_space_per_s=None, speed_path_per_s=None,
               traversal_time_s=None)
    if len(x) >= 2 and x[-1] > x[0]:
        w = np.sqrt(np.maximum(weights, 1e-12))
        slope, intercept = np.polyfit(x, tt, 1, w=w)
        pred = slope * x + intercept
        ss_res = float(np.sum((tt - pred) ** 2))
        ss_tot = float(np.sum((tt - tt.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0
        full_span = float(space[-1] - space[0])
        speed = None if abs(slope) <= 1e-15 else float(1.0 / abs(slope))
        fit.update(
            slope_s_per_space=float(slope),
            intercept_s=float(intercept),
            fit_r2=float(r2),
            speed_space_per_s=speed,
            speed_path_per_s=None if speed is None or full_span <= 0 else float(speed / full_span),
            traversal_time_s=float(abs(slope) * (x[-1] - x[0])),
        )
    return cols, tids, amp, fit


def detect_events(signal, valid, time_s, space, config=None):
    """Detect candidate time-space events and describe their propagation geometry.

    Returns candidate events only. The result is not yet a validated uterine
    peristaltic-wave detector and must be compared with real-video references.
    """
    config = (config or EventDetectionConfig()).validate()
    y, m, t, s = _validate_map(signal, valid, time_s, space)
    activity = robust_activity(y, m)
    active = m & np.isfinite(activity) & (activity >= config.z_threshold)
    comps = _components(active)
    events = []
    full_span = float(s[-1] - s[0])

    for cells in comps:
        if len(cells) < config.min_component_cells:
            continue
        cols, tids, amp, fit = _fit_ridge(cells, activity, y, t, s)
        unique_space = len(cols)
        start_t_idx, end_t_idx = int(cells[:, 0].min()), int(cells[:, 0].max())
        start_s_idx, end_s_idx = int(cols.min()), int(cols.max())
        span = float(s[end_s_idx] - s[start_s_idx])
        path_fraction = span / full_span if full_span > 0 else 0.0
        duration_s = float(t[end_t_idx] - t[start_t_idx])

        if unique_space < config.min_space_points or path_fraction < config.min_path_fraction:
            kind = "local"
        elif fit["slope_s_per_space"] is None:
            kind = "uncertain"
        elif fit["traversal_time_s"] <= config.synchronous_tolerance_s:
            kind = "synchronous"
        elif fit["fit_r2"] is None or fit["fit_r2"] < config.min_fit_r2:
            kind = "complex"
        else:
            kind = "C2F" if fit["slope_s_per_space"] > 0 else "F2C"

        events.append(dict(
            event_id=0,
            classification=kind,
            start_time_s=float(t[start_t_idx]),
            end_time_s=float(t[end_t_idx]),
            duration_s=duration_s,
            start_space=float(s[start_s_idx]),
            end_space=float(s[end_s_idx]),
            path_fraction=float(path_fraction),
            spatial_points=int(unique_space),
            component_cells=int(len(cells)),
            peak_abs_amplitude=float(np.nanmax(amp)) if len(amp) else None,
            rms_amplitude=float(np.sqrt(np.nanmean(amp * amp))) if len(amp) else None,
            ridge_space=s[cols].astype(float),
            ridge_time_s=t[tids].astype(float),
            ridge_time_indices=tids.astype(int),
            ridge_space_indices=cols.astype(int),
            **fit,
        ))

    events.sort(key=lambda e: (e["start_time_s"], e["start_space"], e["classification"]))
    for i, event in enumerate(events, 1):
        event["event_id"] = i
    return dict(
        config=asdict(config),
        activity=activity,
        active_mask=active,
        events=events,
        event_count=len(events),
    )
