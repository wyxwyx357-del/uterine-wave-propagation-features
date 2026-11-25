"""Outcome-blind measurement bridge between 2-s A and Huang-like frame RSR.

Phase A only. It reads saved A0 trajectories, layouts, and frozen features.
It does not read pregnancy outcomes, fit outcome models, or alter production
features.
"""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.mse_pregnancy_association.code.complete_record_coordination import summarize_paired_windows
from analysis.mse_pregnancy_association.code.frozen_feature_auc import save
from analysis.mse_pregnancy_association.code.measurement.deformation import compute_spacetime
from analysis.mse_pregnancy_association.code.measurement.modes import analyze_modes, contiguous_blocks

S_RANGE = (0.2, 0.8)
WINDOW_S = 20.0
STEP_S = 1.0
BAND_CPM = (0.5, 5.0)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def _control_grid(initial_points):
    initial = np.asarray(initial_points, float)
    total = len(initial)
    if total % 7:
        raise ValueError("Control point count is not divisible by wall (2) and midline (3) tracks")
    count = total // 7
    pair_ids = np.arange(4 * count).reshape(2, count, 2)
    inner = initial[pair_ids[:, :, 0]]
    arc = []
    for wall in inner:
        distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wall, axis=0), axis=1))]
        arc.append(distance / distance[-1] if distance[-1] > 0 else np.linspace(0, 1, count))
    arc = np.asarray(arc)
    s = arc.mean(axis=0)
    b_curve_ids = np.arange(3 * count).reshape(3, count)
    b_arc_positions = []
    for curve in initial[b_curve_ids]:
        distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        b_arc_positions.append(distance)
    return dict(
        u=s,
        A_points=initial[:4 * count],
        B_points=initial[4 * count:],
        A_pair_ids=pair_ids,
        arc_positions=arc,
        B_curve_ids=b_curve_ids,
        B_arc_positions=np.asarray(b_arc_positions),
        wall_lengths=np.ptp(initial[pair_ids[:, :, 0]], axis=1),
    )


def huang_like_rsr(tracks, pair_ids):
    """Signed frame-to-frame radial velocity-difference / previous distance."""
    points = np.asarray(tracks["corrected"], float)
    time_s = np.asarray(tracks["time_s"], float)
    if len(time_s) < 2 or np.any(np.diff(time_s) <= 0):
        raise ValueError("Strictly increasing trajectory times are required")
    if np.any(~np.isfinite(points).all(2)):
        points = points.copy()
    pair_ids = np.asarray(pair_ids, int)
    walls, sections = pair_ids.shape[:2]
    result = np.full((len(time_s) - 1, walls, sections), np.nan)
    valid = np.zeros_like(result, bool)

    target_valid = np.asarray(tracks["target_valid"], bool)
    step_valid = np.asarray(tracks["step_valid"], bool)
    segment = np.asarray(tracks["segment_id"])
    epoch = np.asarray(tracks.get("pose_epoch", np.zeros(len(time_s), int)))
    transition = (
        target_valid[1:]
        & target_valid[:-1]
        & step_valid[1:]
        & (segment[1:] == segment[:-1])
        & (epoch[1:] == epoch[:-1])[:, None]
    )

    for wall in range(walls):
        for section in range(sections):
            first, second = pair_ids[wall, section]
            p1_prev, p2_prev = points[:-1, first], points[:-1, second]
            p1_now, p2_now = points[1:, first], points[1:, second]
            length = np.linalg.norm(p2_prev - p1_prev, axis=1)
            unit = np.divide(
                p2_prev - p1_prev,
                length[:, None],
                out=np.zeros((len(length), 2), float),
                where=length[:, None] > 1e-12,
            )
            displacement_difference = (p2_now - p2_prev) - (p1_now - p1_prev)
            radial_velocity_difference = np.einsum(
                "ij,ij->i", displacement_difference, unit
            ) / np.diff(time_s)
            values = np.divide(
                radial_velocity_difference,
                length,
                out=np.full_like(length, np.nan),
                where=length > 1e-12,
            )
            support = (
                transition[:, first]
                & transition[:, second]
                & np.isfinite(values)
            )
            result[:, wall, section] = values
            valid[:, wall, section] = support
    dt = np.diff(time_s)
    return dict(
        A=result,
        epsilon=result * dt[:, None, None],
        valid_mask=valid,
        s=None,
        interval_start_s=time_s[:-1].copy(),
        interval_end_s=time_s[1:].copy(),
        interval_center_s=(time_s[:-1] + time_s[1:]) / 2,
        interval_dt_s=dt,
        hop=1,
        nominal_interval_s=float(np.median(dt)),
        A_arc_positions=None,
        definition="Huang_like_RSR",
        formula="dot((delta_outer-delta_inner),previous_inner_to_outer_unit)/(previous_pair_length*dt)",
    )


def _with_space(maps, s):
    maps = dict(maps)
    maps["s"] = np.asarray(s, float)
    maps["A_arc_positions"] = np.asarray(s, float)
    return maps


def _common_time(left, right):
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    key_right = {int(round(value * 1e9)): index for index, value in enumerate(right)}
    pairs = [
        (index, key_right[int(round(value * 1e9))])
        for index, value in enumerate(left)
        if int(round(value * 1e9)) in key_right
    ]
    if not pairs:
        raise ValueError("Branches have no common native interval times")
    left_index = np.asarray([pair[0] for pair in pairs], int)
    right_index = np.asarray([pair[1] for pair in pairs], int)
    common = left[left_index]
    return left_index, right_index, common


def aligned_modes(left, right):
    if left.get("s") is None or right.get("s") is None:
        raise ValueError("Both branches require spatial coordinates")
    left_index, right_index, common = _common_time(
        left["interval_end_s"], right["interval_end_s"]
    )
    s = np.asarray(left["s"], float)
    if not np.allclose(s, np.asarray(right["s"], float), atol=1e-12, rtol=0):
        raise ValueError("Branches do not share the same spatial grid")
    ids = np.flatnonzero((s >= S_RANGE[0]) & (s <= S_RANGE[1]))
    if len(ids) < 3:
        raise ValueError("Fixed bridge region has fewer than three sections")
    lo, hi = int(ids[0]), int(ids[-1] + 1)
    outputs = []
    for maps, index, name in (
        (left, left_index, "A_2s"),
        (right, right_index, "B_Huang_like"),
    ):
        signal = np.asarray(maps["A"], float)[index]
        valid = np.asarray(maps["valid_mask"], bool)[index].copy()
        good = valid & np.isfinite(signal)
        complete = good[:, :, lo:hi].all(2)
        good[:, :, lo:hi] &= complete[:, :, None]
        mode = analyze_modes(
            dict(
                A=signal,
                valid_mask=good,
                interval_center_s=common,
                interval_start_s=np.asarray(maps["interval_start_s"])[index],
                interval_end_s=np.asarray(maps["interval_end_s"])[index],
                interval_dt_s=np.asarray(maps["interval_dt_s"])[index],
                s=s,
            ),
            window_s=WINDOW_S,
            step_s=STEP_S,
            regions=[(0, lo, hi, "anterior"), (1, lo, hi, "posterior")],
            include_velocity=False,
        )
        outputs.append((name, signal, good, mode))
    return outputs


def signal_metrics(signal, valid):
    signal = np.asarray(signal, float)
    valid = np.asarray(valid, bool)
    rows = []
    for wall in range(signal.shape[1]):
        all_values = signal[:, wall][valid[:, wall]]
        all_values = all_values[np.isfinite(all_values)]
        row = dict(
            wall=wall,
            samples=len(all_values),
            RMS=float(np.sqrt(np.mean(all_values**2))) if len(all_values) else None,
            SD=float(np.std(all_values)) if len(all_values) else None,
            median_abs=float(np.median(np.abs(all_values))) if len(all_values) else None,
            P5=float(np.quantile(all_values, 0.05)) if len(all_values) else None,
            P50=float(np.quantile(all_values, 0.50)) if len(all_values) else None,
            P95=float(np.quantile(all_values, 0.95)) if len(all_values) else None,
        )
        crossings = []
        autocorrelations = []
        band_fractions = []
        for section in range(signal.shape[2]):
            for start, end in contiguous_blocks(valid[:, wall, section], minimum=3):
                values = signal[start:end, wall, section]
                values = values - values.mean()
                if len(values) > 1:
                    signs = np.sign(values)
                    nonzero = signs != 0
                    if nonzero.sum() > 1:
                        s = signs[nonzero]
                        crossings.append(float(np.mean(s[1:] != s[:-1])))
                if len(values) >= 3 and np.std(values) > 1e-12:
                    autocorrelations.append(float(np.corrcoef(values[:-1], values[1:])[0, 1]))
                    power = np.abs(np.fft.rfft(values)) ** 2
                    frequencies = np.fft.rfftfreq(len(values), 1.0)
                    band = (frequencies >= BAND_CPM[0] / 60) & (frequencies <= BAND_CPM[1] / 60)
                    if power.sum() > 1e-25:
                        band_fractions.append(float(power[band].sum() / power.sum()))
        row.update(
            zero_crossing_fraction=float(np.median(crossings)) if crossings else None,
            lag1_autocorrelation=float(np.median(autocorrelations)) if autocorrelations else None,
            band_energy_fraction=float(np.median(band_fractions)) if band_fractions else None,
        )
        rows.append(row)
    return rows


def _paired_frame(paired):
    if not paired:
        return pd.DataFrame()
    frame = pd.DataFrame(paired)
    needed = ["first_section", "last_section", "window_start_s", "window_end_s"]
    frame = frame.drop_duplicates(needed)
    return frame.set_index(needed)


def er_comparison(left_modes, right_modes):
    left = _paired_frame(left_modes["paired_ER_windows"])
    right = _paired_frame(right_modes["paired_ER_windows"])
    if left.empty or right.empty:
        return pd.DataFrame(
            [
                dict(
                    wall=wall,
                    paired_windows=0,
                    common_windows=0,
                    Pearson=None,
                    Spearman=None,
                    median_abs_difference=None,
                    sign_agreement=None,
                )
                for wall in ("anterior", "posterior")
            ]
        )
    common = left.join(right, how="inner", lsuffix="_A", rsuffix="_B")
    rows = []
    for wall_name, columns in (
        ("anterior", ("anterior_ER_A", "anterior_ER_B")),
        ("posterior", ("posterior_ER_A", "posterior_ER_B")),
    ):
        use = common[list(columns)].dropna()
        if len(use) >= 3 and use.iloc[:, 0].std() > 1e-12 and use.iloc[:, 1].std() > 1e-12:
            pearson = float(pearsonr(use.iloc[:, 0], use.iloc[:, 1]).statistic)
            spearman = float(spearmanr(use.iloc[:, 0], use.iloc[:, 1]).statistic)
        else:
            pearson = spearman = None
        rows.append(
            dict(
                wall=wall_name,
                paired_windows=len(common),
                common_windows=len(use),
                Pearson=pearson,
                Spearman=spearman,
                median_abs_difference=float(
                    np.median(np.abs(use.iloc[:, 0] - use.iloc[:, 1]))
                ) if len(use) else None,
                sign_agreement=float(np.mean(np.sign(use.iloc[:, 0]) == np.sign(use.iloc[:, 1])))
                if len(use) else None,
            )
        )
    return pd.DataFrame(rows)


def control_analysis(control_root):
    rows = []
    for texture, condition in product(range(4), ("C2F", "F2C", "rigid", "local_once", "static")):
        path = control_root / f"{texture}_{condition}_tracks.npz"
        if not path.exists():
            continue
        tracks = _load_npz(path)
        grid = _control_grid(tracks["initial_points"])
        a_maps = _with_space(compute_spacetime(tracks, grid, interval_s=2.0), grid["u"])
        b_maps = _with_space(huang_like_rsr(tracks, grid["A_pair_ids"]), grid["u"])
        aligned = aligned_modes(a_maps, b_maps)
        mode_by_branch = {name: mode for name, _, _, mode in aligned}
        for name, _, _, mode in aligned:
            record = summarize_paired_windows(mode["paired_ER_windows"])
            rows.append(
                dict(
                    texture=texture,
                    condition=condition,
                    branch=name,
                    expected_direction=("C2F" if condition == "C2F" else
                                        "F2C" if condition == "F2C" else "none"),
                    paired_windows=record["paired_window_count"],
                    temporal_segments=record["temporal_segment_count"],
                    mean_ER_anterior=record["record_ER_mean_anterior"],
                    mean_ER_posterior=record["record_ER_mean_posterior"],
                    MSE=record["record_MSE_time_equal"],
                    CC=record["record_CC_time_equal"],
                )
            )
    return pd.DataFrame(rows)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    cases = sorted(
        path for path in args.feature_root.iterdir()
        if path.is_dir() and (path / "features.json").exists()
    )
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("No frozen patient feature cases found")
    hashes = {}
    case_rows = []
    er_rows = []
    signal_rows = []
    failures = []

    for folder in cases:
        case_id = folder.name
        try:
            features_path = folder / "features.json"
            tracks_path = folder / "tracks.npz"
            layout_path = folder / "layout.npz"
            maps_path = folder / "spacetime_2s.npz"
            for path in (features_path, tracks_path, layout_path, maps_path):
                hashes[str(path.resolve())] = _hash(path)
            features = json.loads(features_path.read_text(encoding="utf-8"))
            tracks = _load_npz(tracks_path)
            layout = _load_npz(layout_path)
            a_maps = _load_npz(maps_path)
            b_maps = _with_space(
                huang_like_rsr(tracks, layout["A_pair_ids"]), layout["u"]
            )
            aligned = aligned_modes(a_maps, b_maps)
            mode_by_branch = {name: mode for name, _, _, mode in aligned}
            official = summarize_paired_windows(
                features["modes"]["paired_ER_windows"]
            )
            aligned_records = {
                name: summarize_paired_windows(mode["paired_ER_windows"])
                for name, _, _, mode in aligned
            }
            row = dict(case_id=case_id)
            row.update({f"A_official_{key}": value for key, value in official.items()
                        if key != "segment_evidence"})
            for name, record in aligned_records.items():
                row.update({f"{name}_{key}": value for key, value in record.items()
                            if key != "segment_evidence"})
            er = er_comparison(mode_by_branch["A_2s"], mode_by_branch["B_Huang_like"])
            er.insert(0, "case_id", case_id)
            er_rows.append(er)
            for name, signal, valid, _ in aligned:
                for metric in signal_metrics(signal, valid):
                    signal_rows.append(dict(case_id=case_id, branch=name, **metric))
            case_rows.append(row)
        except Exception as error:
            failures.append(dict(case_id=case_id, error_type=type(error).__name__, reason=str(error)))

    case_frame = pd.DataFrame(case_rows)
    er_frame = pd.concat(er_rows, ignore_index=True) if er_rows else pd.DataFrame()
    signal_frame = pd.DataFrame(signal_rows)
    failure_frame = pd.DataFrame(failures)
    controls = control_analysis(args.controls)
    if er_frame.empty:
        er_frame = pd.DataFrame(
            columns=["case_id", "wall", "paired_windows", "common_windows",
                     "Pearson", "Spearman", "median_abs_difference", "sign_agreement"]
        )
    if signal_frame.empty:
        signal_frame = pd.DataFrame(
            columns=["case_id", "branch", "wall", "samples", "RMS", "SD",
                     "median_abs", "P5", "P50", "P95",
                     "zero_crossing_fraction", "lag1_autocorrelation",
                     "band_energy_fraction"]
        )
    coverage = case_frame[
        ["case_id",
         "A_official_paired_window_count",
         "A_2s_paired_window_count",
         "B_Huang_like_paired_window_count",
         "A_2s_record_MSE_time_equal",
         "B_Huang_like_record_MSE_time_equal"]
    ].copy()
    coverage.rename(
        columns={
            "A_2s_record_MSE_time_equal": "A_aligned_MSE_available",
            "B_Huang_like_record_MSE_time_equal": "B_Huang_like_MSE_available",
        },
        inplace=True,
    )
    mse_frame = case_frame[
        ["case_id", "A_official_record_MSE_time_equal",
         "A_2s_record_MSE_time_equal", "B_Huang_like_record_MSE_time_equal"]
    ].copy()
    mse_frame["B_minus_A"] = (
        mse_frame["B_Huang_like_record_MSE_time_equal"]
        - mse_frame["A_2s_record_MSE_time_equal"]
    )
    decomposition_rows = []
    for row in case_frame.itertuples(index=False):
        for branch, prefix in (
            ("A_2s", "A_2s_"),
            ("B_Huang_like", "B_Huang_like_"),
        ):
            decomposition_rows.append(
                dict(
                    case_id=row.case_id,
                    branch=branch,
                    mean_ant=getattr(row, prefix + "record_ER_mean_anterior"),
                    mean_post=getattr(row, prefix + "record_ER_mean_posterior"),
                    delta_ER=(
                        getattr(row, prefix + "record_ER_mean_anterior")
                        - getattr(row, prefix + "record_ER_mean_posterior")
                    ) if pd.notna(getattr(row, prefix + "record_ER_mean_anterior")) else np.nan,
                    mean_bias_sq=getattr(row, prefix + "record_mean_bias_sq"),
                    sd_scale_sq=getattr(row, prefix + "record_sd_scale_sq"),
                    time_shape_term=getattr(row, prefix + "record_time_shape_term"),
                    MSE_z=getattr(row, prefix + "record_MSE_z"),
                    CC=getattr(row, prefix + "record_CC_time_equal"),
                    MSE=getattr(row, prefix + "record_MSE_time_equal"),
                )
            )
    decomposition = pd.DataFrame(decomposition_rows)

    for name, frame in (
        ("case_level_bridge", case_frame),
        ("ER_comparison", er_frame),
        ("signal_numeric_comparison", signal_frame),
        ("measurement_QC", coverage),
        ("measurement_failures", failure_frame),
        ("control_comparison", controls),
        ("MSE_comparison", mse_frame),
        ("MSE_decomposition_comparison", decomposition),
    ):
        frame.to_csv(args.output / f"{name}.csv", index=False, encoding="utf-8-sig")

    common = case_frame.dropna(
        subset=["A_2s_record_MSE_time_equal", "B_Huang_like_record_MSE_time_equal"]
    )
    if len(common) >= 3:
        a_mse = common["A_2s_record_MSE_time_equal"].to_numpy(float)
        b_mse = common["B_Huang_like_record_MSE_time_equal"].to_numpy(float)
        difference = b_mse - a_mse
        mse_comparison = dict(
            common_cases=len(common),
            Pearson=float(pearsonr(a_mse, b_mse).statistic),
            Spearman=float(spearmanr(a_mse, b_mse).statistic),
            median_abs_difference=float(np.median(np.abs(difference))),
            Bland_Altman_bias=float(difference.mean()),
            Bland_Altman_lower=float(np.quantile(difference, 0.025)),
            Bland_Altman_upper=float(np.quantile(difference, 0.975)),
        )
    else:
        mse_comparison = dict(common_cases=len(common))

    signal_summary = (
        signal_frame.groupby(["branch", "wall"], as_index=False)
        .agg(
            cases=("case_id", "nunique"),
            median_RMS=("RMS", "median"),
            median_SD=("SD", "median"),
            median_abs=("median_abs", "median"),
            median_lag1_autocorrelation=("lag1_autocorrelation", "median"),
            median_band_energy_fraction=("band_energy_fraction", "median"),
        )
    )
    signal_summary.to_csv(args.output / "signal_summary.csv", index=False, encoding="utf-8-sig")
    er_summary = (
        er_frame.groupby("wall", as_index=False)
        .agg(
            cases=("case_id", "nunique"),
            median_common_windows=("common_windows", "median"),
            median_Pearson=("Pearson", "median"),
            median_Spearman=("Spearman", "median"),
            median_MAD=("median_abs_difference", "median"),
            median_sign_agreement=("sign_agreement", "median"),
        )
    )
    er_summary.to_csv(args.output / "ER_summary.csv", index=False, encoding="utf-8-sig")

    summary = dict(
        cases_requested=len(cases),
        cases_completed=len(case_frame),
        failures=failure_frame.to_dict(orient="records"),
        common_mse_cases=mse_comparison.get("common_cases", 0),
        MSE_comparison=mse_comparison,
        ER_summary=er_summary.to_dict(orient="records"),
        signal_summary=signal_summary.to_dict(orient="records"),
        controls=controls.to_dict(orient="records"),
        outcome_fields_read=False,
        tracking_rerun=False,
        production_features_changed=False,
        source_sha256=hashes,
        method=dict(
            A="frozen 2-s finite-interval radial stretch rate",
            B="Huang_like frame-to-frame signed radial velocity-difference / previous pair length",
            bridge_time_alignment="right-endpoint intersection of native interval times",
            window_s=WINDOW_S,
            step_s=STEP_S,
            frequency_band_cpm=BAND_CPM,
            s_range=S_RANGE,
            ER="(E_C2F-E_F2C)/(E_C2F+E_F2C)",
            coordination="same-window anterior/posterior ER MSE and CC",
            Huang_not_specified=(
                "moving-window step, FFT normalization, window function, and exact signed "
                "projection convention; project settings retained and recorded"
            ),
        ),
    )
    save(args.output / "method_comparison.json", _json_safe(summary))

    lines = [
        "# Huang 2022测量桥接：Phase A",
        "",
        "本阶段未读取临床妊娠、结局或AUC。",
        "",
        f"请求病例{len(cases)}，完成{len(case_frame)}，失败{len(failure_frame)}。",
        f"2 s与Huang-like共同可计算MSE记录{summary['common_mse_cases']}。",
        "",
        "## 方法对齐",
        "",
        "- Branch A：正式约2秒有限区间径向伸缩率。",
        "- Branch B：`Huang_like_RSR`，由相邻帧外侧/内侧位移差在上一帧径向方向上的投影，除以点距和dt。",
        "- 两分支使用同一右侧原生时间轴、ROI 0.2–0.8、20秒窗、1秒步长、0.5–5 cpm、ER和完整记录MSE。",
        "- 未重新追踪；正式特征未修改。",
        "",
        "## MSE一致性",
        "",
        f"- Pearson：{mse_comparison.get('Pearson', 'NA')}",
        f"- Spearman：{mse_comparison.get('Spearman', 'NA')}",
        f"- 中位绝对差：{mse_comparison.get('median_abs_difference', 'NA')}",
        f"- Bland–Altman bias：{mse_comparison.get('Bland_Altman_bias', 'NA')}",
        "",
        "## 患者级失败",
        "",
    ]
    if failures:
        for item in failures:
            lines.append(f"- {item['case_id']}: {item['error_type']}: {item['reason']}")
    else:
        lines.append("无。")
    lines += [
        "",
        "## 控制检查",
        "",
        "|纹理|条件|分支|配对ER窗|前壁平均ER|后壁平均ER|MSE|CC|",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in controls.itertuples(index=False):
        lines.append(
            f"|{row.texture}|{row.condition}|{row.branch}|{row.paired_windows}|"
            f"{row.mean_ER_anterior if np.isfinite(row.mean_ER_anterior) else float('nan'):.4f}|"
            f"{row.mean_ER_posterior if np.isfinite(row.mean_ER_posterior) else float('nan'):.4f}|"
            f"{row.MSE if np.isfinite(row.MSE) else float('nan'):.4f}|"
            f"{row.CC if np.isfinite(row.CC) else float('nan'):.4f}|"
        )
    lines += [
        "",
        "## Phase A状态",
        "",
        "桥接计算已完成并输出；是否冻结需根据逐例失败、控制方向、rigid/static假方向以及两分支ER/MSE一致性综合判断。",
        "本报告未读取任何临床妊娠字段。",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    run(parser.parse_args())
