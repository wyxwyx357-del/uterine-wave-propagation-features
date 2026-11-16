"""Single-event and event-set summaries for propagation candidates."""
import numpy as np


TRAVELING_CLASSES = ("C2F", "F2C")


def _finite_median(values):
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    return float(np.median(x)) if len(x) else None


def _finite_mean(values):
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    return float(np.mean(x)) if len(x) else None


def summarize_events(events):
    """Summarize a list of event dictionaries without inventing observation time."""
    events = list(events)
    classes = ("C2F", "F2C", "synchronous", "local", "complex", "uncertain")
    counts = {name: sum(e.get("classification") == name for e in events) for name in classes}
    traveling = [e for e in events if e.get("classification") in TRAVELING_CLASSES]
    denom = len(traveling)
    return dict(
        event_count=len(events),
        traveling_event_count=denom,
        class_counts=counts,
        C2F_fraction=(counts["C2F"] / denom) if denom else None,
        F2C_fraction=(counts["F2C"] / denom) if denom else None,
        median_speed_space_per_s=_finite_median([e.get("speed_space_per_s") for e in traveling]),
        median_speed_path_per_s=_finite_median([e.get("speed_path_per_s") for e in traveling]),
        median_path_fraction=_finite_median([e.get("path_fraction") for e in traveling]),
        median_duration_s=_finite_median([e.get("duration_s") for e in traveling]),
        median_peak_abs_amplitude=_finite_median([e.get("peak_abs_amplitude") for e in traveling]),
        mean_fit_r2=_finite_mean([e.get("fit_r2") for e in traveling]),
    )


def effective_observation_seconds(valid, time_s, min_spatial_fraction=0.5):
    """Duration supported by enough spatial columns; no temporal filling."""
    m = np.asarray(valid, bool)
    t = np.asarray(time_s, float)
    if m.ndim != 2 or m.shape[0] != len(t) or len(t) < 2:
        raise ValueError("valid must be time x space and match time_s")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("time_s must be finite and strictly increasing")
    if not 0 < min_spatial_fraction <= 1:
        raise ValueError("min_spatial_fraction must lie in (0,1]")
    enough = m.mean(axis=1) >= min_spatial_fraction
    dt = np.diff(t)
    supported = enough[:-1] & enough[1:]
    return float(dt[supported].sum())


def event_rate_per_minute(events, observation_seconds, *, traveling_only=True):
    """Event count per valid observed minute."""
    if observation_seconds is None or not np.isfinite(observation_seconds) or observation_seconds <= 0:
        return None
    chosen = [e for e in events if (e.get("classification") in TRAVELING_CLASSES or not traveling_only)]
    return float(len(chosen) / (observation_seconds / 60.0))
