"""Patient-level candidate features aggregated from validated-support event lists."""
import numpy as np
from .event_features import TRAVELING_CLASSES, effective_observation_seconds, summarize_events


def _common_observation_seconds(anterior_valid, posterior_valid, time_s, min_spatial_fraction):
    a = np.asarray(anterior_valid, bool)
    b = np.asarray(posterior_valid, bool)
    t = np.asarray(time_s, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != len(t) or b.shape[0] != len(t):
        raise ValueError("bilateral valid maps must be time x space and match time_s")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("time_s must be finite and strictly increasing")
    if not 0 < min_spatial_fraction <= 1:
        raise ValueError("min_spatial_fraction must lie in (0,1]")
    enough = (a.mean(1) >= min_spatial_fraction) & (b.mean(1) >= min_spatial_fraction)
    return float(np.diff(t)[enough[:-1] & enough[1:]].sum())


def _event_index(events):
    return {e.get("event_id"): e for e in events}


def aggregate_patient_features(
    anterior_events, posterior_events, anterior_valid, posterior_valid,
    time_s, coordination, *, min_spatial_fraction=0.5,
):
    """Aggregate A-route event candidates into patient-level research features."""
    ant = list(anterior_events)
    post = list(posterior_events)
    ant_obs = effective_observation_seconds(anterior_valid, time_s, min_spatial_fraction)
    post_obs = effective_observation_seconds(posterior_valid, time_s, min_spatial_fraction)
    common_obs = _common_observation_seconds(
        anterior_valid, posterior_valid, time_s, min_spatial_fraction
    )

    ant_idx, post_idx = _event_index(ant), _event_index(post)
    used_a, used_b = set(), set()
    unique_classes = []
    unique_traveling_count = 0

    for pair in coordination.get("pairs", []):
        aid = pair.get("anterior_event_id")
        pid = pair.get("posterior_event_id")
        used_a.add(aid)
        used_b.add(pid)
        a = ant_idx.get(aid)
        b = post_idx.get(pid)
        ac = a.get("classification") if a else None
        bc = b.get("classification") if b else None
        a_travel = ac in TRAVELING_CLASSES
        b_travel = bc in TRAVELING_CLASSES
        if a_travel or b_travel:
            unique_traveling_count += 1
            if a_travel and b_travel and ac == bc:
                unique_classes.append(ac)
            elif a_travel and b_travel and ac != bc:
                unique_classes.append("discordant")
            else:
                unique_classes.append("unilateral_direction")

    for e in ant:
        if e.get("event_id") not in used_a and e.get("classification") in TRAVELING_CLASSES:
            unique_traveling_count += 1
            unique_classes.append(e["classification"])
    for e in post:
        if e.get("event_id") not in used_b and e.get("classification") in TRAVELING_CLASSES:
            unique_traveling_count += 1
            unique_classes.append(e["classification"])

    class_counts = {
        key: unique_classes.count(key)
        for key in ("C2F", "F2C", "discordant", "unilateral_direction")
    }
    directional_denom = class_counts["C2F"] + class_counts["F2C"]

    ant_summary = summarize_events(ant)
    post_summary = summarize_events(post)
    rate = float(unique_traveling_count / (common_obs / 60.0)) if common_obs > 0 else None
    wall_traveling = [e for e in ant + post if e.get("classification") in TRAVELING_CLASSES]

    speeds = np.asarray([
        e.get("speed_path_per_s") for e in wall_traveling
        if e.get("speed_path_per_s") is not None and np.isfinite(e.get("speed_path_per_s"))
    ], float)
    amplitudes = np.asarray([
        e.get("peak_abs_amplitude") for e in wall_traveling
        if e.get("peak_abs_amplitude") is not None and np.isfinite(e.get("peak_abs_amplitude"))
    ], float)
    ranges = np.asarray([
        e.get("path_fraction") for e in wall_traveling
        if e.get("path_fraction") is not None and np.isfinite(e.get("path_fraction"))
    ], float)

    return dict(
        status="candidate_features_not_clinically_validated",
        min_spatial_fraction=float(min_spatial_fraction),
        anterior_observation_s=ant_obs,
        posterior_observation_s=post_obs,
        common_bilateral_observation_s=common_obs,
        anterior=ant_summary,
        posterior=post_summary,
        unique_traveling_event_count=int(unique_traveling_count),
        unique_traveling_events_per_min=rate,
        unique_direction_counts=class_counts,
        C2F_fraction=(class_counts["C2F"] / directional_denom) if directional_denom else None,
        F2C_fraction=(class_counts["F2C"] / directional_denom) if directional_denom else None,
        median_wall_event_speed_path_per_s=float(np.median(speeds)) if len(speeds) else None,
        median_wall_event_peak_abs_amplitude=float(np.median(amplitudes)) if len(amplitudes) else None,
        median_wall_event_path_fraction=float(np.median(ranges)) if len(ranges) else None,
        coordination=dict(coordination),
    )
