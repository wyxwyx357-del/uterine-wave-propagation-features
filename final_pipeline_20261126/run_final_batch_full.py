from __future__ import annotations

"""Full frozen measurement runner layered over run_final_batch.py.

Adds only engineering orchestration that was requested after the first package:
- reuse verified tracks.npz + layout.npz without retracking;
- preserve additional existing amplitude / ER / directional-rms / MSE QC fields;
- write RAW and median-imputed tables with missing indicators.

It does not change A0, M1, MSE or R2 algorithms or thresholds.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_pipeline_20261126 import run_final_batch as base

PIPELINE_VERSION = "UTERINE_DYNAMIC_FINAL_BATCH_20261126_V2_FULL"

IMPUTABLE_FEATURES = [
    "R2_frequency_cpm",
    "m1_prop_rate_min_per_min", "m1_prop_rate_max_per_min",
    "anterior_amplitude", "posterior_amplitude",
    "anterior_partial_region_amplitude_QC", "posterior_partial_region_amplitude_QC",
    "anterior_raw_interval_strain_SD_QC", "posterior_raw_interval_strain_SD_QC",
    "anterior_ER_mean", "anterior_ER_median", "anterior_ER_sd",
    "posterior_ER_mean", "posterior_ER_median", "posterior_ER_sd",
    "anterior_directional_rms_mean", "anterior_directional_rms_median", "anterior_directional_rms_sd",
    "posterior_directional_rms_mean", "posterior_directional_rms_median", "posterior_directional_rms_sd",
    "record_MSE_time_equal", "record_MSE_pooled", "record_CC_time_equal",
    "record_ER_mean_anterior", "record_ER_mean_posterior",
    "record_ER_sd_anterior", "record_ER_sd_posterior",
    "record_mean_bias_sq", "record_sd_scale_sq", "record_time_shape_term", "record_MSE_z",
    "segment_MSE_mean", "segment_MSE_median",
]


def _tracking_qc(tracks, grid=None):
    valid = np.asarray(tracks["target_valid"], bool)
    step = np.asarray(tracks["step_valid"], bool)
    segment = np.asarray(tracks["segment_id"])
    reanchor = np.asarray(tracks.get("reanchor", np.zeros(valid.shape, bool)), bool)
    attempt = np.asarray(tracks.get("recovery_attempted", np.zeros(valid.shape, bool)), bool)
    ncc = np.asarray(tracks.get("target_ncc", np.full(valid.shape, np.nan)), float)
    fb = np.asarray(tracks.get("target_fb", np.full(valid.shape, np.nan)), float)
    out = dict(
        tracking_target_valid_fraction=float(valid.mean()),
        tracking_step_valid_fraction=float(step[1:].mean()) if len(step) > 1 else None,
        tracking_reanchor_count=int(reanchor.sum()),
        tracking_recovery_attempt_count=int(attempt.sum()),
        tracking_segment_count=int(len(np.unique(segment[segment >= 0]))),
        tracking_ncc_median=float(np.nanmedian(ncc)) if np.isfinite(ncc).any() else None,
        tracking_fb_median_px=float(np.nanmedian(fb)) if np.isfinite(fb).any() else None,
    )
    if grid is not None and "A_pair_ids" in grid:
        ids = np.asarray(grid["A_pair_ids"])
        if ids.ndim == 3 and ids.shape[0] == 2:
            for wall, name in ((0, "anterior"), (1, "posterior")):
                point_ids = np.unique(ids[wall].ravel())
                out[f"tracking_{name}_target_valid_fraction"] = float(valid[:, point_ids].mean())
    return out


def run_or_load_a0(row, case_dir):
    cache = base.optional_path(row.get("a0_spacetime_cache"))
    if cache is not None:
        if not cache.is_file():
            raise FileNotFoundError(f"a0_spacetime_cache not found: {cache}")
        maps = base.load_npz(cache)
        base.validate_spacetime(maps)
        return maps, dict(
            a0_source="external_spacetime_cache",
            a0_cache_path=str(cache), a0_cache_sha256=base.sha256(cache),
            a0_reused_existing_tracking=True,
        )

    tracks_path = base.optional_path(row.get("a0_tracks_cache"))
    layout_path = base.optional_path(row.get("a0_layout_cache"))
    if tracks_path is not None or layout_path is not None:
        if tracks_path is None or layout_path is None or not tracks_path.is_file() or not layout_path.is_file():
            raise FileNotFoundError("tracks/layout reuse requires both a0_tracks_cache and a0_layout_cache")
        tracks = base.load_npz(tracks_path)
        grid = base.load_npz(layout_path)
        maps = base.compute_spacetime(tracks, grid, interval_s=2.0)
        base.validate_spacetime(maps)
        folder = case_dir / "a0"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "spacetime_2s.npz", **maps)
        meta = dict(
            a0_source="external_tracks_layout_cache",
            a0_tracks_cache_path=str(tracks_path), a0_layout_cache_path=str(layout_path),
            a0_tracks_cache_sha256=base.sha256(tracks_path),
            a0_layout_cache_sha256=base.sha256(layout_path),
            a0_reused_existing_tracking=True,
            **_tracking_qc(tracks, grid),
        )
        base.save_json(folder / "input_provenance.json", meta)
        return maps, meta

    video = base.optional_path(row.get("video"))
    annotation = base.optional_path(row.get("annotation"))
    pose_path = base.optional_path(row.get("pose"))
    for name, path in (("video", video), ("annotation", annotation), ("pose", pose_path)):
        if path is None or not path.is_file():
            raise FileNotFoundError(f"missing_{name}_input")
    time_factor = float(base.text(row.get("time_factor")) or "1.0")
    frames, t = base.read_video(video, time_factor=time_factor, coordinate_scale=.5)
    grid = base.layout(base.read_labelme(annotation, coordinate_scale=.5))
    pose = base.load_npz(pose_path)
    for key in ("time_s", "poses", "pose_epoch"):
        if key not in pose:
            raise ValueError(f"pose cache missing {key}")
    if pose["time_s"].shape != t.shape or not np.allclose(pose["time_s"], t, atol=1e-6, rtol=0):
        raise ValueError("pose_timeline_mismatch")
    tracks = base.track_segments(
        frames, np.vstack((grid["A_points"], grid["B_points"])), t, pose["poses"],
        pose_epoch=pose["pose_epoch"], pose_valid=pose.get("pose_valid"), recover=True,
    )
    maps = base.compute_spacetime(tracks, grid, interval_s=2.0)
    base.validate_spacetime(maps)
    folder = case_dir / "a0"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "tracks.npz", **tracks)
    np.savez_compressed(folder / "layout.npz", **grid)
    np.savez_compressed(folder / "spacetime_2s.npz", **maps)
    meta = dict(
        a0_source="fresh_frozen_A0", a0_reused_existing_tracking=False,
        video=str(video), annotation=str(annotation), pose=str(pose_path),
        time_factor=time_factor, decoded_frame_count=len(t), decoded_video_span_s=float(t[-1] - t[0]),
        video_sha256=base.sha256(video), annotation_sha256=base.sha256(annotation), pose_sha256=base.sha256(pose_path),
        **_tracking_qc(tracks, grid),
    )
    base.save_json(folder / "input_provenance.json", meta)
    return maps, meta


def _stat(values, kind):
    x = np.asarray([float(v) for v in values if v is not None and np.isfinite(v)], float)
    if not len(x):
        return None
    if kind == "mean": return float(x.mean())
    if kind == "median": return float(np.median(x))
    if kind == "sd": return float(x.std(ddof=1)) if len(x) >= 2 else 0.0
    raise ValueError(kind)


def summarize_maps(case_id, maps, case_dir):
    base.validate_spacetime(maps)
    features = base.summarize_patient(maps, s_range=base.MSE_S_RANGE, include_mode_velocity=False)
    paired = features["modes"]["paired_ER_windows"]
    mse = base.summarize_paired_windows(paired)
    with base.intervention("M1"):
        m1 = base.count_map(maps, base.M1_NOISE_FLOOR)

    seconds = float(m1.get("effective_observation_seconds") or 0.0)
    completed = m1.get("status") == "completed" and seconds > 0
    m1_status = "completed" if completed else ("no_effective_support" if m1.get("status") == "no_analyzable_event_support" else str(m1.get("status") or "failed"))
    windows = features["modes"]["windows"]
    ant = [r for r in windows if int(r["wall"]) == 0 and r.get("ER") is not None]
    post = [r for r in windows if int(r["wall"]) == 1 and r.get("ER") is not None]
    walls = features["walls"]
    raw_valid = np.asarray(maps["valid_mask"], bool) & np.isfinite(np.asarray(maps["A"], float))
    observation = m1.get("observation") or {}

    row = dict(
        case_id=case_id, pipeline_version=PIPELINE_VERSION, a0_status="completed",
        raw_valid_fraction=float(raw_valid.mean()),
        anterior_amplitude=walls[0].get("radial_stretch_activity_amplitude"),
        posterior_amplitude=walls[1].get("radial_stretch_activity_amplitude"),
        anterior_partial_region_amplitude_QC=walls[0].get("partial_region_activity_amplitude_QC"),
        posterior_partial_region_amplitude_QC=walls[1].get("partial_region_activity_amplitude_QC"),
        anterior_raw_interval_strain_SD_QC=walls[0].get("raw_interval_strain_SD_QC"),
        posterior_raw_interval_strain_SD_QC=walls[1].get("raw_interval_strain_SD_QC"),
        anterior_amplitude_status=walls[0].get("amplitude_status"), posterior_amplitude_status=walls[1].get("amplitude_status"),
        anterior_analyzed_sections=walls[0].get("analyzed_sections"), posterior_analyzed_sections=walls[1].get("analyzed_sections"),
        anterior_requested_sections=walls[0].get("requested_sections"), posterior_requested_sections=walls[1].get("requested_sections"),
        anterior_common_region_duration_s=walls[0].get("common_region_duration_s"),
        posterior_common_region_duration_s=walls[1].get("common_region_duration_s"),
        anterior_valid_fraction=walls[0].get("valid_fraction"), posterior_valid_fraction=walls[1].get("valid_fraction"),
        anterior_ER_mean=_stat([r.get("ER") for r in ant], "mean"), anterior_ER_median=_stat([r.get("ER") for r in ant], "median"),
        anterior_ER_sd=_stat([r.get("ER") for r in ant], "sd"), anterior_ER_window_count=len(ant),
        posterior_ER_mean=_stat([r.get("ER") for r in post], "mean"), posterior_ER_median=_stat([r.get("ER") for r in post], "median"),
        posterior_ER_sd=_stat([r.get("ER") for r in post], "sd"), posterior_ER_window_count=len(post),
        anterior_directional_rms_mean=_stat([r.get("directional_rms") for r in ant], "mean"),
        anterior_directional_rms_median=_stat([r.get("directional_rms") for r in ant], "median"),
        anterior_directional_rms_sd=_stat([r.get("directional_rms") for r in ant], "sd"),
        posterior_directional_rms_mean=_stat([r.get("directional_rms") for r in post], "mean"),
        posterior_directional_rms_median=_stat([r.get("directional_rms") for r in post], "median"),
        posterior_directional_rms_sd=_stat([r.get("directional_rms") for r in post], "sd"),
        record_MSE_time_equal=mse.get("record_MSE_time_equal"), record_MSE_pooled=mse.get("record_MSE_pooled"),
        record_CC_time_equal=mse.get("record_CC_time_equal"), record_ER_mean_anterior=mse.get("record_ER_mean_anterior"),
        record_ER_mean_posterior=mse.get("record_ER_mean_posterior"), record_ER_sd_anterior=mse.get("record_ER_sd_anterior"),
        record_ER_sd_posterior=mse.get("record_ER_sd_posterior"), record_mean_bias_sq=mse.get("record_mean_bias_sq"),
        record_sd_scale_sq=mse.get("record_sd_scale_sq"), record_time_shape_term=mse.get("record_time_shape_term"),
        record_MSE_z=mse.get("record_MSE_z"), mse_status=mse.get("status"), mse_decomposition_status=mse.get("decomposition_status"),
        unique_window_count=mse.get("unique_window_count"), paired_window_count=mse.get("paired_window_count"),
        spatial_block_count=mse.get("spatial_block_count"), temporal_segment_count=mse.get("temporal_segment_count"),
        temporal_gap_count=mse.get("temporal_gap_count"), segment_MSE_mean=mse.get("segment_MSE_mean"),
        segment_MSE_median=mse.get("segment_MSE_median"), window_span_s=mse.get("window_span_s"),
        mean_spatial_columns=_stat([(int(r["last_section"]) - int(r["first_section"]) + 1) for r in paired], "mean"),
        m1_status=m1_status, m1_failure_reason="" if completed else str(m1.get("failure_reason") or ""),
        m1_algorithm_version=base.M1_VERSION,
        m1_prop_count_min=m1.get("significant_propagation_activity_count_min") if completed else None,
        m1_prop_count_max=m1.get("significant_propagation_activity_count_max") if completed else None,
        m1_prop_rate_min_per_min=m1.get("significant_propagation_activity_frequency_cpm_min") if completed else None,
        m1_prop_rate_max_per_min=m1.get("significant_propagation_activity_frequency_cpm_max") if completed else None,
        m1_other_significant_activity_count=m1.get("other_significant_activity_count") if completed else None,
        ambiguous_cross_wall_prop_pairs=m1.get("ambiguous_cross_wall_propagation_pairs") if completed else None,
        effective_observation_seconds=seconds if seconds > 0 else None,
        effective_observation_minutes=seconds / 60.0 if seconds > 0 else None,
        raw_observation_seconds=observation.get("raw_observation_seconds"),
        filtered_observation_seconds=observation.get("filtered_observation_seconds"),
        filtered_observation_fraction=observation.get("filtered_observation_fraction"),
        m1_minmax_physical_cutoff=(m1.get("normalization") or {}).get("physical_cutoff") if completed else None,
    )
    base.save_json(case_dir / "a0_mse_features.json", features)
    base.save_json(case_dir / "mse_record.json", mse)
    base.save_json(case_dir / "m1_record.json", m1)
    return row


def _output_from_argv(argv):
    for i, item in enumerate(argv):
        if item == "--output" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if item.startswith("--output="):
            return Path(item.split("=", 1)[1])
    raise ValueError("--output is required")


def _median_impute(raw_path, output):
    df = pd.read_csv(raw_path)
    records = {}
    for col in IMPUTABLE_FEATURES:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        missing = x.isna()
        df[col + "__missing"] = missing.astype(int)
        available = x[~missing]
        if not len(available):
            records[col] = dict(median=None, available_n=0, missing_n=int(missing.sum()), imputed=False)
            continue
        med = float(available.median())
        df[col] = x.fillna(med)
        records[col] = dict(median=med, available_n=int((~missing).sum()), missing_n=int(missing.sum()), imputed=True)
    path = output / "FINAL_DYNAMIC_FEATURES_MEDIAN_IMPUTED.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    base.save_json(output / "MEDIAN_IMPUTATION_VALUES.json", records)
    return path, records


def main():
    output = _output_from_argv(sys.argv[1:])
    base.PIPELINE_VERSION = PIPELINE_VERSION
    base.run_or_load_a0 = run_or_load_a0
    base.summarize_maps = summarize_maps
    base.main()

    legacy = output / "FINAL_DYNAMIC_FEATURES.csv"
    raw = output / "FINAL_DYNAMIC_FEATURES_RAW.csv"
    shutil.copyfile(legacy, raw)
    imputed, records = _median_impute(raw, output)
    summary_path = output / "FINAL_RUN_SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    table = pd.read_csv(raw)
    summary.update(
        pipeline_version=PIPELINE_VERSION,
        pregnancy_outcome_read=False,
        a0_reused_spacetime_n=int((table.get("a0_source") == "external_spacetime_cache").sum()) if "a0_source" in table else 0,
        a0_reused_tracks_layout_n=int((table.get("a0_source") == "external_tracks_layout_cache").sum()) if "a0_source" in table else 0,
        a0_fresh_tracking_n=int((table.get("a0_source") == "fresh_frozen_A0").sum()) if "a0_source" in table else 0,
        a0_failed_n=int((table.get("a0_status") == "failed").sum()) if "a0_status" in table else 0,
        all_three_valid=int(table[["R2_frequency_cpm", "m1_prop_rate_min_per_min", "record_MSE_time_equal"]].notna().all(axis=1).sum()),
        raw_table=str(raw), median_imputed_table=str(imputed),
        median_imputation_values=str(output / "MEDIAN_IMPUTATION_VALUES.json"),
        median_imputation_feature_count=len(records),
    )
    base.save_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
