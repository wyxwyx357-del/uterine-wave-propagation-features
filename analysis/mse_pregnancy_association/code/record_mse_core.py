"""Frozen numeric-only ER-MSE core, copied verbatim from the historical implementation.

No files, clinical fields, labels or association/model modules are imported.
"""
import numpy as np


def _segments(rows):
    """Split one spatial block into chronological runs without crossing gaps."""
    rows = sorted(rows, key=lambda row: row["window_center_s"])
    times = np.asarray([row["window_center_s"] for row in rows], float)
    if len(times) == 1:
        return [rows]
    positive = np.diff(times)
    positive = positive[positive > 0]
    if not len(positive):
        raise ValueError("Repeated paired-window timestamps")
    step = float(np.median(positive))
    cuts = np.flatnonzero(np.diff(times) > 1.5 * step) + 1
    bounds = np.r_[0, cuts, len(rows)]
    return [rows[int(left):int(right)] for left, right in zip(bounds[:-1], bounds[1:])]


def _coordination(rows):
    x = np.asarray([row["anterior_ER"] for row in rows], float)
    y = np.asarray([row["posterior_ER"] for row in rows], float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Paired ER values must be finite")
    mse = float(np.mean((x - y) ** 2))
    cc = (
        float(np.corrcoef(x, y)[0, 1])
        if len(rows) >= 3 and np.std(x) > 1e-12 and np.std(y) > 1e-12
        else None
    )
    return mse, cc


def _decompose(ant, post, observed_mse):
    """Decompose observed ER-MSE into bias, scale and time-shape terms."""
    ant = np.asarray(ant, float)
    post = np.asarray(post, float)
    mean_ant = float(ant.mean())
    mean_post = float(post.mean())
    sd_ant = float(ant.std(ddof=0))
    sd_post = float(post.std(ddof=0))
    mean_bias_sq = float((mean_ant - mean_post) ** 2)
    scale_difference_sq = float((sd_ant - sd_post) ** 2)
    if len(ant) >= 3 and sd_ant > 1e-12 and sd_post > 1e-12:
        cc = float(np.corrcoef(ant, post)[0, 1])
        shape_term = float(2 * sd_ant * sd_post * (1 - cc))
        standardized_mse = float(
            np.mean(
                (
                    (ant - mean_ant) / sd_ant
                    - (post - mean_post) / sd_post
                )
                ** 2
            )
        )
    else:
        shape_term = float(observed_mse - mean_bias_sq - scale_difference_sq)
        standardized_mse = None
    return dict(
        record_ER_mean_anterior=mean_ant,
        record_ER_mean_posterior=mean_post,
        record_ER_sd_anterior=sd_ant,
        record_ER_sd_posterior=sd_post,
        record_mean_bias_sq=mean_bias_sq,
        record_sd_scale_sq=scale_difference_sq,
        record_time_shape_term=shape_term,
        record_MSE_z=standardized_mse,
    )


def summarize_paired_windows(paired):
    """Return record-level coordination and all contributing temporal runs.

    Primary MSE gives each unique window time equal weight, then each spatial
    block at that time equal weight. ``pooled_MSE`` is a sensitivity metric in
    which every window-block observation is equally weighted. No missing time
    or unsupported spatial section is represented as zero.
    """
    if not paired:
        return dict(
            status="no_shared_supported_ER_windows",
            record_MSE_time_equal=None,
            record_MSE_pooled=None,
            record_CC_time_equal=None,
            record_ER_mean_anterior=None,
            record_ER_mean_posterior=None,
            record_ER_sd_anterior=None,
            record_ER_sd_posterior=None,
            record_mean_bias_sq=None,
            record_sd_scale_sq=None,
            record_time_shape_term=None,
            record_MSE_z=None,
            decomposition_status="unavailable_no_shared_support",
            paired_window_count=0,
            unique_window_count=0,
            spatial_block_count=0,
            temporal_segment_count=0,
            segment_MSE_mean=None,
            segment_MSE_median=None,
            window_span_s=None,
            segment_evidence=[],
        )

    by_block = {}
    for row in paired:
        key = (int(row["first_section"]), int(row["last_section"]))
        by_block.setdefault(key, []).append(row)

    segment_evidence = []
    all_blocks = []
    for (lo, hi), block_rows in sorted(by_block.items()):
        all_blocks.append(block_rows)
        for index, part in enumerate(_segments(block_rows), 1):
            mse, cc = _coordination(part)
            segment_evidence.append(
                dict(
                    first_section=lo,
                    last_section=hi,
                    segment_index=index,
                    window_count=len(part),
                    start_center_s=float(min(row["window_center_s"] for row in part)),
                    end_center_s=float(max(row["window_center_s"] for row in part)),
                    ER_MSE=mse,
                    ER_CC=cc,
                )
            )

    flat = [row for block in all_blocks for row in block]
    pooled_mse = _coordination(flat)[0]

    # One series per recording: average available spatial-block ER at each
    # identical window time, then calculate the MSE and CC over time.
    by_time = {}
    for row in flat:
        key = (float(row["window_start_s"]), float(row["window_end_s"]), float(row["window_center_s"]))
        by_time.setdefault(key, []).append(row)
    ordered_times = sorted(by_time)
    ant = np.asarray([np.mean([row["anterior_ER"] for row in by_time[key]]) for key in ordered_times])
    post = np.asarray([np.mean([row["posterior_ER"] for row in by_time[key]]) for key in ordered_times])
    error_by_time = np.asarray(
        [np.mean([(row["anterior_ER"] - row["posterior_ER"]) ** 2 for row in by_time[key]]) for key in ordered_times]
    )
    observed_mse = float(np.mean(error_by_time))
    if max(len(rows) for rows in by_time.values()) == 1:
        decomposition = _decompose(ant, post, observed_mse)
        decomposition["decomposition_status"] = "one_block_per_window_time"
    else:
        decomposition = {
            "record_ER_mean_anterior": None,
            "record_ER_mean_posterior": None,
            "record_ER_sd_anterior": None,
            "record_ER_sd_posterior": None,
            "record_mean_bias_sq": None,
            "record_sd_scale_sq": None,
            "record_time_shape_term": None,
            "record_MSE_z": None,
            "decomposition_status": "not_defined_for_multiple_blocks_per_time",
        }
    record_cc = (
        float(np.corrcoef(ant, post)[0, 1])
        if len(ordered_times) >= 3 and np.std(ant) > 1e-12 and np.std(post) > 1e-12
        else None
    )
    centers = np.asarray([key[2] for key in ordered_times], float)
    positive = np.diff(centers)
    positive = positive[positive > 0]
    step = float(np.median(positive)) if len(positive) else 0.0
    gaps = np.diff(centers) > 1.5 * step if step else np.zeros(0, bool)
    segment_array = np.asarray([row["ER_MSE"] for row in segment_evidence], float)
    return dict(
        status="available_on_shared_ER_support",
        record_MSE_time_equal=observed_mse,
        record_MSE_pooled=float(pooled_mse),
        record_CC_time_equal=record_cc,
        **decomposition,
        paired_window_count=len(flat),
        unique_window_count=len(ordered_times),
        spatial_block_count=len(by_block),
        temporal_segment_count=len(segment_evidence),
        temporal_gap_count=int(gaps.sum()),
        segment_MSE_mean=float(segment_array.mean()),
        segment_MSE_median=float(np.median(segment_array)),
        window_span_s=float(max(row["window_end_s"] for row in flat) - min(row["window_start_s"] for row in flat)),
        first_section=min(key[0] for key in by_block),
        last_section=max(key[1] for key in by_block),
        segment_evidence=segment_evidence,
    )
