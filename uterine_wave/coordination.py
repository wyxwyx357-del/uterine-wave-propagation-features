"""Anterior/posterior event matching for candidate uterine-wave events."""
import numpy as np

TRAVELING_CLASSES = ("C2F", "F2C")


def _midpoint(event):
    return 0.5 * (float(event["start_time_s"]) + float(event["end_time_s"]))


def _temporal_overlap(a, b):
    lo = max(float(a["start_time_s"]), float(b["start_time_s"]))
    hi = min(float(a["end_time_s"]), float(b["end_time_s"]))
    return max(0.0, hi - lo)


def match_bilateral_events(anterior_events, posterior_events, *, max_midpoint_lag_s=2.0):
    """Greedy one-to-one matching of anterior/posterior event candidates.

    Only temporal proximity is used for matching; direction agreement is
    reported afterward and is not required to create a match.
    """
    if not np.isfinite(max_midpoint_lag_s) or max_midpoint_lag_s < 0:
        raise ValueError("max_midpoint_lag_s must be non-negative")
    ant = list(anterior_events)
    post = list(posterior_events)
    candidates = []
    for i, a in enumerate(ant):
        for j, b in enumerate(post):
            lag = abs(_midpoint(a) - _midpoint(b))
            if lag <= max_midpoint_lag_s:
                overlap = _temporal_overlap(a, b)
                candidates.append((lag, -overlap, i, j))
    candidates.sort()

    used_a, used_b, pairs = set(), set(), []
    for lag, neg_overlap, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        a, b = ant[i], post[j]
        same_direction = (
            a.get("classification") in TRAVELING_CLASSES
            and a.get("classification") == b.get("classification")
        )
        pairs.append(dict(
            anterior_event_id=a.get("event_id"),
            posterior_event_id=b.get("event_id"),
            midpoint_lag_s=float(lag),
            overlap_s=float(-neg_overlap),
            anterior_direction=a.get("classification"),
            posterior_direction=b.get("classification"),
            same_direction=bool(same_direction),
        ))

    unmatched_a = [a.get("event_id") for i, a in enumerate(ant) if i not in used_a]
    unmatched_b = [b.get("event_id") for j, b in enumerate(post) if j not in used_b]
    matched_traveling = [
        p for p in pairs
        if p["anterior_direction"] in TRAVELING_CLASSES and p["posterior_direction"] in TRAVELING_CLASSES
    ]
    agreement = (
        float(np.mean([p["same_direction"] for p in matched_traveling]))
        if matched_traveling else None
    )
    lag_median = (
        float(np.median([p["midpoint_lag_s"] for p in pairs]))
        if pairs else None
    )
    return dict(
        pairs=pairs,
        matched_pair_count=len(pairs),
        unmatched_anterior_event_ids=unmatched_a,
        unmatched_posterior_event_ids=unmatched_b,
        traveling_direction_agreement_fraction=agreement,
        median_midpoint_lag_s=lag_median,
        max_midpoint_lag_s=float(max_midpoint_lag_s),
    )
