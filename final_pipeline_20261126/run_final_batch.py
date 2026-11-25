from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.mse_pregnancy_association.code.measurement.io import read_video, jsonable
from analysis.mse_pregnancy_association.code.measurement.geometry import read_labelme, layout
from analysis.mse_pregnancy_association.code.measurement.tracking import track_segments
from analysis.mse_pregnancy_association.code.measurement.deformation import compute_spacetime
from analysis.mse_pregnancy_association.code.measurement.patient import summarize_patient
from analysis.mse_pregnancy_association.code.record_mse_core import summarize_paired_windows
from analysis.mse_pregnancy_association.code.batch_wave_counts import count_map
from experiments.step2_salience_sandbox.mechanism_2x2.mechanisms import intervention

PIPELINE_VERSION = "UTERINE_DYNAMIC_FINAL_BATCH_20261126_V1"
M1_VERSION = "m1-global-pairing-unseen-v1"
M1_NOISE_FLOOR = 0.0007509593468916695
MSE_S_RANGE = (0.2, 0.8)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text(v):
    return "" if v is None else str(v).strip()


def optional_path(v):
    s = text(v)
    return Path(s) if s else None


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def validate_spacetime(maps: dict):
    needed = {"A", "epsilon", "valid_mask", "s", "interval_start_s", "interval_end_s", "interval_center_s", "interval_dt_s"}
    missing = needed - set(maps)
    if missing:
        raise ValueError("A0 spacetime cache missing keys: " + ",".join(sorted(missing)))
    a = np.asarray(maps["A"], float)
    v = np.asarray(maps["valid_mask"], bool)
    s = np.asarray(maps["s"], float)
    t = np.asarray(maps["interval_center_s"], float)
    if a.shape != v.shape or a.shape != (len(t), 2, len(s)):
        raise ValueError("A0 spacetime cache shape mismatch")
    if len(t) < 3 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("A0 spacetime cache invalid timeline")
    nominal = float(np.asarray(maps.get("nominal_interval_s", 2.0)).reshape(-1)[0])
    if abs(nominal - 2.0) > 1e-6:
        raise ValueError(f"Expected frozen nominal_interval_s=2.0, got {nominal}")


def run_or_load_a0(row: dict, case_dir: Path):
    cache = optional_path(row.get("a0_spacetime_cache"))
    if cache is not None:
        if not cache.is_file():
            raise FileNotFoundError(f"a0_spacetime_cache not found: {cache}")
        maps = load_npz(cache)
        validate_spacetime(maps)
        return maps, {"a0_source": "external_spacetime_cache", "a0_cache_path": str(cache), "a0_cache_sha256": sha256(cache)}

    video = optional_path(row.get("video")); annotation = optional_path(row.get("annotation")); pose_path = optional_path(row.get("pose"))
    for name, p in (("video", video), ("annotation", annotation), ("pose", pose_path)):
        if p is None or not p.is_file():
            raise FileNotFoundError(f"missing_{name}_input")
    time_factor = float(text(row.get("time_factor")) or "1.0")
    frames, t = read_video(video, time_factor=time_factor, coordinate_scale=.5)
    grid = layout(read_labelme(annotation, coordinate_scale=.5))
    pose = load_npz(pose_path)
    for key in ("time_s", "poses", "pose_epoch"):
        if key not in pose:
            raise ValueError(f"pose cache missing {key}")
    if pose["time_s"].shape != t.shape or not np.allclose(pose["time_s"], t, atol=1e-6, rtol=0):
        raise ValueError("pose_timeline_mismatch")
    tracks = track_segments(
        frames, np.vstack((grid["A_points"], grid["B_points"])), t, pose["poses"],
        pose_epoch=pose["pose_epoch"], pose_valid=pose.get("pose_valid"), recover=True
    )
    maps = compute_spacetime(tracks, grid, interval_s=2.0)
    validate_spacetime(maps)
    a0_dir = case_dir / "a0"
    a0_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a0_dir / "tracks.npz", **tracks)
    np.savez_compressed(a0_dir / "layout.npz", **grid)
    np.savez_compressed(a0_dir / "spacetime_2s.npz", **maps)
    meta = {
        "a0_source": "fresh_frozen_A0",
        "video": str(video), "annotation": str(annotation), "pose": str(pose_path),
        "time_factor": time_factor, "decoded_frame_count": len(t),
        "decoded_video_span_s": float(t[-1] - t[0]),
        "video_sha256": sha256(video), "annotation_sha256": sha256(annotation), "pose_sha256": sha256(pose_path),
    }
    save_json(a0_dir / "input_provenance.json", meta)
    return maps, meta


def mean_or_none(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def summarize_maps(case_id: str, maps: dict, case_dir: Path):
    validate_spacetime(maps)
    features = summarize_patient(maps, s_range=MSE_S_RANGE, include_mode_velocity=False)
    paired = features["modes"]["paired_ER_windows"]
    mse = summarize_paired_windows(paired)
    with intervention("M1"):
        m1 = count_map(maps, M1_NOISE_FLOOR)

    m1_seconds = float(m1.get("effective_observation_seconds") or 0.0)
    if m1.get("status") == "completed" and m1_seconds > 0:
        m1_status = "completed"
        count_min = m1.get("significant_propagation_activity_count_min")
        count_max = m1.get("significant_propagation_activity_count_max")
        rate_min = m1.get("significant_propagation_activity_frequency_cpm_min")
        rate_max = m1.get("significant_propagation_activity_frequency_cpm_max")
        m1_reason = ""
    else:
        m1_status = "no_effective_support" if m1.get("status") == "no_analyzable_event_support" else str(m1.get("status") or "failed")
        count_min = count_max = rate_min = rate_max = None
        m1_reason = str(m1.get("failure_reason") or "")

    windows = features["modes"]["windows"]
    ant_windows = [r for r in windows if int(r["wall"]) == 0]
    post_windows = [r for r in windows if int(r["wall"]) == 1]
    mean_cols = mean_or_none([(int(r["last_section"]) - int(r["first_section"]) + 1) for r in paired])
    raw_valid = np.asarray(maps["valid_mask"], bool) & np.isfinite(np.asarray(maps["A"], float))
    walls = features["walls"]

    row = {
        "case_id": case_id,
        "pipeline_version": PIPELINE_VERSION,
        "a0_status": "completed",
        "raw_valid_fraction": float(raw_valid.mean()),
        "anterior_amplitude": walls[0].get("radial_stretch_activity_amplitude"),
        "posterior_amplitude": walls[1].get("radial_stretch_activity_amplitude"),
        "anterior_amplitude_status": walls[0].get("amplitude_status"),
        "posterior_amplitude_status": walls[1].get("amplitude_status"),
        "anterior_common_region_duration_s": walls[0].get("common_region_duration_s"),
        "posterior_common_region_duration_s": walls[1].get("common_region_duration_s"),
        "anterior_ER_mean_paired": mean_or_none([r.get("anterior_ER") for r in paired]),
        "posterior_ER_mean_paired": mean_or_none([r.get("posterior_ER") for r in paired]),
        "anterior_directional_rms_mean": mean_or_none([r.get("directional_rms") for r in ant_windows]),
        "posterior_directional_rms_mean": mean_or_none([r.get("directional_rms") for r in post_windows]),
        "record_MSE_time_equal": mse.get("record_MSE_time_equal"),
        "record_MSE_z": mse.get("record_MSE_z"),
        "record_mean_bias_sq": mse.get("record_mean_bias_sq"),
        "record_sd_scale_sq": mse.get("record_sd_scale_sq"),
        "record_time_shape_term": mse.get("record_time_shape_term"),
        "mse_status": mse.get("status"),
        "mse_decomposition_status": mse.get("decomposition_status"),
        "unique_window_count": mse.get("unique_window_count"),
        "paired_window_count": mse.get("paired_window_count"),
        "mean_spatial_columns": mean_cols,
        "m1_status": m1_status,
        "m1_failure_reason": m1_reason,
        "m1_algorithm_version": M1_VERSION,
        "m1_prop_count_min": count_min,
        "m1_prop_count_max": count_max,
        "effective_observation_seconds": (m1_seconds if m1_seconds > 0 else None),
        "effective_observation_minutes": (m1_seconds / 60.0 if m1_seconds > 0 else None),
        "m1_prop_rate_min_per_min": rate_min,
        "m1_prop_rate_max_per_min": rate_max,
        "m1_other_significant_activity_count": m1.get("other_significant_activity_count") if m1_status == "completed" else None,
        "ambiguous_cross_wall_prop_pairs": m1.get("ambiguous_cross_wall_propagation_pairs") if m1_status == "completed" else None,
        "m1_minmax_physical_cutoff": (m1.get("normalization") or {}).get("physical_cutoff") if m1_status == "completed" else None,
    }
    save_json(case_dir / "a0_mse_features.json", features)
    save_json(case_dir / "mse_record.json", mse)
    save_json(case_dir / "m1_record.json", m1)
    return row


def run_a0_m1_mse(rows: list[dict], output: Path, resume: bool):
    case_root = output / "cases"
    case_root.mkdir(parents=True, exist_ok=True)
    results = []
    for index, row in enumerate(rows, 1):
        case_id = text(row.get("case_id"))
        if not case_id:
            raise ValueError("Every manifest row needs case_id")
        folder = case_root / case_id
        folder.mkdir(parents=True, exist_ok=True)
        saved = folder / "final_dynamic_row.json"
        if resume and saved.is_file():
            results.append(json.loads(saved.read_text(encoding="utf-8")))
            print(f"[{index}/{len(rows)}] {case_id}: reuse completed row", flush=True)
            continue
        print(f"[{index}/{len(rows)}] {case_id}: A0/M1/MSE", flush=True)
        base = {"case_id": case_id, "pipeline_version": PIPELINE_VERSION}
        try:
            maps, meta = run_or_load_a0(row, folder)
            base.update(meta)
            base.update(summarize_maps(case_id, maps, folder))
            base["a0_failure_reason"] = ""
        except Exception as e:
            base.update(
                a0_status="failed", a0_failure_reason=f"{type(e).__name__}: {e}",
                record_MSE_time_equal=None, mse_status="unavailable_a0_failure",
                m1_status="unavailable_a0_failure", m1_failure_reason="a0_failure",
                m1_algorithm_version=M1_VERSION, m1_prop_rate_min_per_min=None,
                m1_prop_rate_max_per_min=None,
            )
            (folder / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        save_json(saved, base)
        results.append(base)
    return results


def write_csv(path: Path, rows: list[dict]):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def run_command(cmd: list[str], cwd: Path, log: Path):
    print("RUN:", " ".join(cmd), flush=True)
    with log.open("w", encoding="utf-8") as out:
        p = subprocess.run(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, check=False)
    if p.returncode:
        raise RuntimeError(f"Command failed ({p.returncode}); see {log}")


def run_r2(rows: list[dict], output: Path, workers: int, resume: bool):
    r2_root = Path(__file__).resolve().parent / "r2_frozen"
    work = output / "r2"
    work.mkdir(parents=True, exist_ok=True)
    eligible, missing = [], {}
    for row in rows:
        case = text(row.get("case_id"))
        video = optional_path(row.get("video")); tr = optional_path(row.get("r2_tracking")); nodes = optional_path(row.get("r2_nodes"))
        if video and video.is_file() and tr and tr.is_file() and nodes and nodes.is_file():
            eligible.append({
                "case_id": case, "video": str(video), "tracking": str(tr), "nodes": str(nodes),
                "mm_per_processed_pixel": text(row.get("mm_per_processed_pixel")),
            })
        else:
            missing[case] = "missing_r2_video_tracking_or_nodes"
    manifest = work / "r2_manifest.csv"
    write_csv(manifest, eligible)
    if not eligible:
        return {}, missing
    validate_report = work / "input_validation.csv"
    tracking_run = work / "tracking_run"
    longitudinal = work / "longitudinal_run"
    reproducibility = work / "reproducibility_run"
    consensus = work / "medium_consensus_run"
    frozen_inputs = work / "frozen_model_inputs"
    if not (resume and validate_report.is_file()):
        run_command([sys.executable, str(r2_root / "verify_tracking_inputs.py"), "--manifest", str(manifest), "--report", str(validate_report)], r2_root, work / "01_verify.log")
    run_command([sys.executable, str(r2_root / "run_tracking_batch.py"), "--manifest", str(manifest), "--run-dir", str(tracking_run), "--workers", str(workers)], r2_root, work / "02_tracking.log")
    if not (resume and (longitudinal / "longitudinal_region_frequency.csv").is_file()):
        run_command([sys.executable, str(r2_root / "extract_longitudinal_frequency.py"), "--source-run", str(tracking_run), "--output", str(longitudinal)], r2_root, work / "03_longitudinal.log")
    if not (resume and (reproducibility / "region_reproducibility_details.csv").is_file()):
        run_command([sys.executable, str(r2_root / "validate_reproducibility.py"), "--source-run", str(tracking_run), "--longitudinal-run", str(longitudinal), "--output", str(reproducibility)], r2_root, work / "04_reproducibility.log")
    if not (resume and (consensus / "medium_consensus_details.csv").is_file()):
        run_command([sys.executable, str(r2_root / "build_medium_consensus.py"), "--source-run", str(tracking_run), "--longitudinal-run", str(longitudinal), "--reproducibility-run", str(reproducibility), "--output", str(consensus)], r2_root, work / "05_consensus.log")
    if not (resume and (frozen_inputs / "main.csv").is_file()):
        if frozen_inputs.exists():
            shutil.rmtree(frozen_inputs)
        run_command([sys.executable, str(r2_root / "prepare_frozen_inputs.py"), "--longitudinal-regions", str(longitudinal / "longitudinal_region_frequency.csv"), "--medium-consensus", str(consensus / "medium_consensus_details.csv"), "--output", str(frozen_inputs)], r2_root, work / "06_prepare.log")
    data = pd.read_csv(frozen_inputs / "main.csv")
    retain = data["retained_for_patient_model"].astype(str).str.lower().isin(["true", "1"])
    posterior = data.loc[retain & data["wall"].eq("posterior")].copy()
    posterior["frequency"] = pd.to_numeric(posterior["optimized_candidate_frequency_cpm"], errors="coerce")
    posterior = posterior[np.isfinite(posterior["frequency"])]
    out = {}
    for case, group in posterior.groupby("case_id"):
        out[str(case)] = {
            "R2_frequency_cpm": float(group["frequency"].mean()),
            "R2_retained_posterior_region_count": int(len(group)),
            "R2_status": "completed",
            "R2_failure_reason": "",
            "R2_version": "R2_LONGITUDINAL_POSTERIOR_FROZEN_CANDIDATE_V1_1_20261106",
        }
    for row in eligible:
        if row["case_id"] not in out:
            out[row["case_id"]] = {
                "R2_frequency_cpm": None, "R2_retained_posterior_region_count": 0,
                "R2_status": "no_retained_posterior_region", "R2_failure_reason": "no_eligible_posterior_region_after_frozen_rules",
                "R2_version": "R2_LONGITUDINAL_POSTERIOR_FROZEN_CANDIDATE_V1_1_20261106",
            }
    return out, missing


def load_manifest(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Manifest is empty")
    ids = [text(r.get("case_id")) for r in rows]
    if any(not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("case_id must be nonempty and unique")
    return rows


def main():
    p = argparse.ArgumentParser(description="Outcome-blind frozen R2 + M1 + MSE batch pipeline")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--r2-workers", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-r2", action="store_true")
    p.add_argument("--skip-a0", action="store_true", help="Only useful when final_dynamic_row.json already exists and --resume is used")
    args = p.parse_args()
    if args.r2_workers < 1:
        p.error("--r2-workers must be >=1")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest)
    if args.skip_a0:
        a0_rows = []
        for row in rows:
            saved = args.output / "cases" / text(row["case_id"]) / "final_dynamic_row.json"
            if not saved.is_file():
                raise FileNotFoundError(f"--skip-a0 requires existing {saved}")
            a0_rows.append(json.loads(saved.read_text(encoding="utf-8")))
    else:
        a0_rows = run_a0_m1_mse(rows, args.output, args.resume)
    r2_map, r2_missing = ({}, {}) if args.skip_r2 else run_r2(rows, args.output, args.r2_workers, args.resume)
    by_id = {r["case_id"]: dict(r) for r in a0_rows}
    final = []
    for src in rows:
        case = text(src["case_id"])
        row = by_id.get(case, {"case_id": case, "pipeline_version": PIPELINE_VERSION})
        if args.skip_r2:
            row.update(R2_frequency_cpm=None, R2_status="skipped", R2_failure_reason="skip_r2_requested")
        elif case in r2_map:
            row.update(r2_map[case])
        else:
            row.update(R2_frequency_cpm=None, R2_retained_posterior_region_count=None,
                       R2_status="missing_input", R2_failure_reason=r2_missing.get(case, "missing_r2_input"),
                       R2_version="R2_LONGITUDINAL_POSTERIOR_FROZEN_CANDIDATE_V1_1_20261106")
        final.append(row)
    table = args.output / "FINAL_DYNAMIC_FEATURES.csv"
    write_csv(table, final)
    summary = {
        "pipeline_version": PIPELINE_VERSION, "outcomes_read": False,
        "case_count": len(final),
        "R2_valid": sum(r.get("R2_frequency_cpm") not in (None, "") for r in final),
        "M1_valid": sum(r.get("m1_prop_rate_min_per_min") not in (None, "") for r in final),
        "MSE_valid": sum(r.get("record_MSE_time_equal") not in (None, "") for r in final),
        "missing_values_are_zero": False,
        "main_dynamic_candidates": ["R2_frequency_cpm", "m1_prop_rate_min_per_min", "record_MSE_time_equal"],
        "final_table": str(table),
    }
    save_json(args.output / "FINAL_RUN_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
