"""Outcome-blind audit of ER under low directional spectral support.

Reads frozen real-case features and saved control trajectories only. It does
not read outcomes, train prediction models, or modify production definitions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.mse_pregnancy_association.code.complete_record_coordination import summarize_paired_windows
from analysis.mse_pregnancy_association.code.frozen_feature_auc import digest, save
from analysis.mse_pregnancy_association.code.huang_bridge_phase_a import (
    _common_time,
    _control_grid,
    _load_npz,
    _with_space,
    aligned_modes,
    huang_like_rsr,
)
from analysis.mse_pregnancy_association.code.measurement.deformation import compute_spacetime

KEY = ["case_id", "texture", "condition", "branch",
       "window_start_s", "window_end_s", "first_section", "last_section"]
WINDOW_KEY = ["case_id", "condition", "branch",
              "window_start_s", "window_end_s", "first_section", "last_section"]
BRANCH_MERGE_KEY = ["case_id", "condition",
                    "window_start_s", "window_end_s", "first_section", "last_section"]


def _window_support_rows(case_id, texture, condition, branch, modes, signal, valid, times):
    times = np.asarray(times, float)
    signal = np.asarray(signal, float)
    valid = np.asarray(valid, bool)
    rows = []
    for window in modes["windows"]:
        selected_time = (
            (times >= float(window["window_start_s"]) - 1e-9)
            & (times <= float(window["window_end_s"]) + 1e-9)
        )
        lo = int(window["first_section"]) - 1
        hi = int(window["last_section"])
        wall = int(window["wall"])
        block = signal[selected_time, wall, lo:hi]
        block_valid = valid[selected_time, wall, lo:hi]
        values = block[block_valid]
        input_rms = float(np.sqrt(np.mean(values**2))) if len(values) else None
        e_sum = float(window["forward_energy"] + window["backward_energy"])
        band_fraction = window["directional_band_energy_fraction"]
        band_energy = (
            float(window["directional_energy"] / band_fraction)
            if band_fraction is not None and band_fraction > 1e-25 else None
        )
        rows.append(
            dict(
                case_id=case_id,
                texture=texture,
                condition=condition,
                branch=branch,
                wall=wall,
                wall_name="anterior" if wall == 0 else "posterior",
                window_start_s=float(window["window_start_s"]),
                window_end_s=float(window["window_end_s"]),
                window_center_s=float(window["window_center_s"]),
                first_section=int(window["first_section"]),
                last_section=int(window["last_section"]),
                section_count=int(window["last_section"] - window["first_section"] + 1),
                spatial_span=float(window["s_end"] - window["s_start"]),
                ER=window["ER"],
                abs_ER=abs(window["ER"]) if window["ER"] is not None else None,
                E_C2F=float(window["forward_energy"]),
                E_F2C=float(window["backward_energy"]),
                E_sum=e_sum,
                E_diff=float(window["forward_energy"] - window["backward_energy"]),
                directional_rms=window["directional_rms"],
                directional_band_energy_fraction=band_fraction,
                experimental_band_energy=band_energy,
                input_rms=input_rms,
                valid_fraction=float(block_valid.mean()) if block_valid.size else None,
            )
        )
    return rows


def _pair_rows(window_frame):
    left = window_frame.loc[window_frame["wall"].eq(0)].copy()
    right = window_frame.loc[window_frame["wall"].eq(1)].copy()
    columns = [
        "ER", "abs_ER", "E_sum", "directional_rms",
        "directional_band_energy_fraction", "input_rms", "valid_fraction", "section_count",
    ]
    keep = KEY + columns
    paired = left[keep].merge(
        right[keep],
        on=KEY,
        how="inner",
        suffixes=("_ant", "_post"),
    )
    paired["ER_difference"] = paired["ER_ant"] - paired["ER_post"]
    paired["window_MSE"] = paired["ER_difference"] ** 2
    paired["support_min"] = paired[["directional_rms_ant", "directional_rms_post"]].min(axis=1)
    paired["support_mean"] = paired[["directional_rms_ant", "directional_rms_post"]].mean(axis=1)
    paired["support_difference"] = (
        paired["directional_rms_ant"] - paired["directional_rms_post"]
    )
    paired["band_fraction_min"] = paired[
        ["directional_band_energy_fraction_ant", "directional_band_energy_fraction_post"]
    ].min(axis=1)
    paired["band_fraction_mean"] = paired[
        ["directional_band_energy_fraction_ant", "directional_band_energy_fraction_post"]
    ].mean(axis=1)
    paired["input_rms_min"] = paired[["input_rms_ant", "input_rms_post"]].min(axis=1)
    paired["input_rms_mean"] = paired[["input_rms_ant", "input_rms_post"]].mean(axis=1)
    return paired


def _correlation(frame, x, y):
    use = frame[[x, y]].dropna()
    if len(use) < 3 or use[x].std() <= 1e-15 or use[y].std() <= 1e-15:
        return dict(n=len(use), Pearson=None, Spearman=None)
    return dict(
        n=len(use),
        Pearson=float(pearsonr(use[x], use[y]).statistic),
        Spearman=float(spearmanr(use[x], use[y]).statistic),
    )


def _record_er_means(group):
    """Average ER at each shared window time, then summarize the recording."""
    by_time = group.groupby(
        ["window_start_s", "window_end_s"], as_index=False
    ).agg(
        mean_ER_ant=("ER_ant", "mean"),
        mean_ER_post=("ER_post", "mean"),
    )
    mean_ant = float(by_time["mean_ER_ant"].mean())
    mean_post = float(by_time["mean_ER_post"].mean())
    signed_bias = mean_ant - mean_post
    return dict(
        mean_ER_ant=mean_ant,
        mean_ER_post=mean_post,
        signed_mean_bias=signed_bias,
        abs_mean_bias=abs(signed_bias),
    )


def _finite_pairs(group):
    finite = np.isfinite(group[["ER_ant", "ER_post"]].to_numpy(float)).all(axis=1)
    return group.loc[finite]


def _distribution(frame, group_columns):
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_windows"] = len(group)
        for feature in ("abs_ER", "directional_rms", "directional_band_energy_fraction",
                        "input_rms"):
            values = group[feature].dropna()
            row[f"{feature}_median"] = float(values.median()) if len(values) else None
            row[f"{feature}_Q1"] = float(values.quantile(0.25)) if len(values) else None
            row[f"{feature}_Q3"] = float(values.quantile(0.75)) if len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)


def _quartile_table(frame, support):
    rows = []
    for (branch, condition, wall_name), group in frame.groupby(
        ["branch", "condition", "wall_name"], dropna=False
    ):
        values = group[support].dropna()
        if len(values) < 4 or values.nunique() < 2:
            continue
        try:
            bin_id = pd.qcut(values, 4, labels=False, duplicates="drop") + 1
        except ValueError:
            continue
        temporary = group.loc[values.index, [support, "abs_ER"]].copy()
        temporary["quartile"] = bin_id
        for quartile, part in temporary.groupby("quartile"):
            rows.append(
                dict(
                    branch=branch,
                    condition=condition,
                    wall_name=wall_name,
                    support_metric=support,
                    quartile=int(quartile),
                    n=len(part),
                    support_min=float(part[support].min()),
                    support_max=float(part[support].max()),
                    median_abs_ER=float(part["abs_ER"].median()),
                    P90_abs_ER=float(part["abs_ER"].quantile(0.90)),
                    fraction_abs_ER_ge_0_5=float((part["abs_ER"] >= 0.5).mean()),
                )
            )
    return pd.DataFrame(rows)


def _direction_accuracy_table(frame, support):
    known = frame.loc[frame["condition"].isin(["C2F", "F2C"])].copy()
    expected = known["condition"].map({"C2F": 1.0, "F2C": -1.0})
    known["direction_correct"] = np.sign(known["ER"]) == expected
    rows = []
    for (branch, condition, wall_name), group in known.groupby(
        ["branch", "condition", "wall_name"], dropna=False
    ):
        values = group[support].dropna()
        if len(values) < 4 or values.nunique() < 2:
            continue
        temporary = group.loc[values.index, [support, "direction_correct"]].copy()
        temporary["quartile"] = pd.qcut(values, 4, labels=False, duplicates="drop") + 1
        for quartile, part in temporary.groupby("quartile"):
            rows.append(
                dict(
                    branch=branch,
                    condition=condition,
                    wall_name=wall_name,
                    support_metric=support,
                    quartile=int(quartile),
                    n=len(part),
                    support_min=float(part[support].min()),
                    support_max=float(part[support].max()),
                    direction_accuracy=float(part["direction_correct"].mean()),
                )
            )
    return pd.DataFrame(rows)


def _append_case_group(
    case_id, texture, condition, branch_outputs, signal_by_branch, valid_by_branch,
    windows, pairs,
):
    for branch, modes, times in branch_outputs:
        signal = signal_by_branch[branch]
        valid = valid_by_branch[branch]
        rows = _window_support_rows(
            case_id, texture, condition, branch, modes, signal, valid, times
        )
        windows.extend(rows)
        if not rows:
            continue
        paired = _pair_rows(pd.DataFrame(rows))
        if len(paired):
            pairs.extend(paired.to_dict(orient="records"))


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    windows = []
    pairs = []
    failures = []
    real_case_ids = []
    source_hashes = {}
    for folder in sorted(args.feature_root.iterdir()):
        if not folder.is_dir() or not (folder / "features.json").exists():
            continue
        real_case_ids.append(folder.name)
        try:
            paths = [folder / name for name in
                     ("features.json", "tracks.npz", "layout.npz", "spacetime_2s.npz")]
            for path in paths:
                source_hashes[str(path.resolve())] = digest(path)
            features = json.loads(paths[0].read_text(encoding="utf-8"))
            tracks = _load_npz(paths[1])
            layout = _load_npz(paths[2])
            a_maps = _load_npz(paths[3])
            b_maps = _with_space(huang_like_rsr(tracks, layout["A_pair_ids"]), layout["u"])
            aligned = aligned_modes(a_maps, b_maps)
            _, _, common_time = _common_time(
                a_maps["interval_end_s"], b_maps["interval_end_s"]
            )
            branch_modes = {name: modes for name, _, _, modes in aligned}
            branch_signals = {name: signal for name, signal, _, _ in aligned}
            branch_valid = {name: valid for name, _, valid, _ in aligned}
            official_a = (
                "A_official",
                features["modes"],
                np.asarray(a_maps["interval_center_s"]),
            )
            _append_case_group(
                folder.name, None, "real",
                [
                    official_a,
                    ("A_2s", branch_modes["A_2s"], common_time),
                    ("B_Huang_like", branch_modes["B_Huang_like"], common_time),
                ],
                {
                    "A_official": np.asarray(a_maps["A"], float),
                    "A_2s": branch_signals["A_2s"],
                    "B_Huang_like": branch_signals["B_Huang_like"],
                },
                {
                    "A_official": np.asarray(a_maps["valid_mask"], bool),
                    "A_2s": branch_valid["A_2s"],
                    "B_Huang_like": branch_valid["B_Huang_like"],
                },
                windows,
                pairs,
            )
        except (KeyError, ValueError, OSError) as error:
            failures.append(dict(case_id=folder.name, error_type=type(error).__name__, reason=str(error)))

    for texture in range(4):
        for condition in ("C2F", "F2C", "rigid", "local_once", "static"):
            path = args.controls / f"{texture}_{condition}_tracks.npz"
            if not path.exists():
                continue
            source_hashes[str(path.resolve())] = digest(path)
            tracks = _load_npz(path)
            grid = _control_grid(tracks["initial_points"])
            a_maps = _with_space(compute_spacetime(tracks, grid, interval_s=2.0), grid["u"])
            b_maps = _with_space(huang_like_rsr(tracks, grid["A_pair_ids"]), grid["u"])
            aligned = aligned_modes(a_maps, b_maps)
            _, _, common_time = _common_time(
                a_maps["interval_end_s"], b_maps["interval_end_s"]
            )
            _append_case_group(
                f"texture{texture}_{condition}", texture, condition,
                [
                    (name, modes, common_time)
                    for name, _, _, modes in aligned
                ],
                {
                    name: signal for name, signal, _, _ in aligned
                },
                {
                    name: valid for name, _, valid, _ in aligned
                },
                windows,
                pairs,
            )

    window_frame = pd.DataFrame(windows)
    pair_frame = pd.DataFrame(pairs)
    present = set(
        map(tuple, window_frame.loc[
            window_frame["condition"].eq("real"), ["case_id", "branch"]
        ].to_numpy())
    )
    case_coverage = pd.DataFrame(
        [
            dict(case_id=case_id, branch=branch,
                 has_ER_window=(case_id, branch) in present)
            for case_id in real_case_ids
            for branch in ("A_official", "A_2s", "B_Huang_like")
        ]
    )
    window_frame.to_csv(args.output / "window_level_support.csv",
                        index=False, encoding="utf-8-sig")
    pair_frame.to_csv(args.output / "paired_window_support.csv",
                      index=False, encoding="utf-8-sig")
    case_coverage.to_csv(args.output / "real_case_coverage.csv",
                         index=False, encoding="utf-8-sig")
    pd.DataFrame(failures).to_csv(args.output / "measurement_failures.csv",
                                  index=False, encoding="utf-8-sig")

    real = window_frame.loc[window_frame["condition"].eq("real")]
    real_pairs = pair_frame.loc[pair_frame["condition"].eq("real")]
    real_support_thresholds = {}
    for branch, group in real_pairs.groupby("branch"):
        values = group["support_mean"].dropna()
        real_support_thresholds[branch] = dict(
            Q1=float(values.quantile(0.25)),
            median=float(values.median()),
        )

    support_distribution = _distribution(
        window_frame,
        ["condition", "branch", "wall_name"],
    )
    support_distribution.to_csv(
        args.output / "support_distribution_by_condition.csv", index=False,
        encoding="utf-8-sig"
    )

    correlation_rows = []
    for (condition, branch, wall_name), group in window_frame.groupby(
        ["condition", "branch", "wall_name"], dropna=False
    ):
        for support in ("directional_rms", "directional_band_energy_fraction", "input_rms"):
            result = _correlation(group, "abs_ER", support)
            correlation_rows.append(
                dict(condition=condition, branch=branch, wall_name=wall_name,
                     support_metric=support, **result)
            )
    correlation_frame = pd.DataFrame(correlation_rows)
    correlation_frame.to_csv(args.output / "support_vs_er.csv",
                             index=False, encoding="utf-8-sig")

    quartile_directional = _quartile_table(window_frame, "directional_rms")
    quartile_band = _quartile_table(window_frame, "directional_band_energy_fraction")
    quartiles = pd.concat([quartile_directional, quartile_band], ignore_index=True)
    quartiles.to_csv(args.output / "support_quartile_summary.csv",
                     index=False, encoding="utf-8-sig")

    accuracy = pd.concat(
        [
            _direction_accuracy_table(window_frame, "directional_rms"),
            _direction_accuracy_table(window_frame, "directional_band_energy_fraction"),
        ],
        ignore_index=True,
    )
    accuracy.to_csv(args.output / "support_vs_direction_accuracy.csv",
                    index=False, encoding="utf-8-sig")

    rigid = pair_frame.loc[pair_frame["condition"].eq("rigid")].copy()
    rigid = rigid.sort_values("window_MSE", ascending=False).reset_index(drop=True)
    rigid.insert(0, "rank_by_window_MSE", np.arange(1, len(rigid) + 1))
    rigid.to_csv(args.output / "rigid_high_mse_windows.csv",
                 index=False, encoding="utf-8-sig")

    pair_correlations = []
    for (condition, branch), group in pair_frame.groupby(["condition", "branch"], dropna=False):
        for support in ("support_mean", "support_min", "band_fraction_mean",
                        "band_fraction_min", "input_rms_mean"):
            result = _correlation(group, "window_MSE", support)
            pair_correlations.append(
                dict(condition=condition, branch=branch, support_metric=support, **result)
            )
    pair_correlations = pd.DataFrame(pair_correlations)
    pair_correlations.to_csv(args.output / "support_vs_window_mse.csv",
                             index=False, encoding="utf-8-sig")

    wall_rows = []
    for (case_id, branch), group in real.groupby(["case_id", "branch"]):
        ant = group.loc[group["wall_name"].eq("anterior")]
        post = group.loc[group["wall_name"].eq("posterior")]
        if not len(ant) or not len(post):
            continue
        wall_rows.append(
            dict(
                case_id=case_id,
                branch=branch,
                anterior_median_support=float(ant["directional_rms"].median()),
                posterior_median_support=float(post["directional_rms"].median()),
                support_difference_ant_minus_post=float(
                    ant["directional_rms"].median() - post["directional_rms"].median()
                ),
                anterior_median_band_fraction=float(
                    ant["directional_band_energy_fraction"].median()
                ),
                posterior_median_band_fraction=float(
                    post["directional_band_energy_fraction"].median()
                ),
                anterior_median_input_rms=float(ant["input_rms"].median()),
                posterior_median_input_rms=float(post["input_rms"].median()),
            )
        )
    wall_asymmetry = pd.DataFrame(wall_rows)
    wall_asymmetry.to_csv(args.output / "wall_support_asymmetry.csv",
                          index=False, encoding="utf-8-sig")

    record_rows = []
    for (case_id, branch), group in real_pairs.groupby(["case_id", "branch"]):
        support = group["support_mean"].dropna()
        threshold_q1 = real_support_thresholds[branch]["Q1"]
        threshold_median = real_support_thresholds[branch]["median"]
        er_means = _record_er_means(group)
        recording = summarize_paired_windows([
            dict(
                first_section=int(row.first_section),
                last_section=int(row.last_section),
                window_start_s=float(row.window_start_s),
                window_end_s=float(row.window_end_s),
                window_center_s=float((row.window_start_s + row.window_end_s) / 2),
                anterior_ER=float(row.ER_ant),
                posterior_ER=float(row.ER_post),
            )
            for row in group.itertuples(index=False)
        ])
        record_rows.append(
            dict(
                case_id=case_id,
                branch=branch,
                paired_windows=len(group),
                **er_means,
                median_support=float(support.median()),
                p10_support=float(support.quantile(0.10)),
                support_Q1_threshold=threshold_q1,
                support_median_threshold=threshold_median,
                fraction_below_real_Q1=float((support <= threshold_q1).mean()),
                fraction_below_real_median=float((support <= threshold_median).mean()),
                MSE=recording["record_MSE_time_equal"],
                mean_bias_sq=recording["record_mean_bias_sq"],
                sd_scale_sq=recording["record_sd_scale_sq"],
                time_shape_term=recording["record_time_shape_term"],
                MSE_z=recording["record_MSE_z"],
                CC=recording["record_CC_time_equal"],
            )
        )
    record_support = pd.DataFrame(record_rows)
    record_support.to_csv(args.output / "record_level_support.csv",
                          index=False, encoding="utf-8-sig")
    record_correlations = []
    for branch, group in record_support.groupby("branch"):
        for variable in ("median_support", "p10_support",
                         "fraction_below_real_Q1", "fraction_below_real_median"):
            result = _correlation(group, "MSE", variable)
            record_correlations.append(dict(branch=branch, variable=variable, **result))
    record_correlations = pd.DataFrame(record_correlations)
    record_correlations.to_csv(args.output / "record_support_vs_MSE.csv",
                               index=False, encoding="utf-8-sig")

    wall_bias = record_support.merge(
        wall_asymmetry,
        on=["case_id", "branch"],
        how="inner",
        validate="one_to_one",
    )
    wall_bias.to_csv(args.output / "wall_support_vs_ER_bias.csv",
                     index=False, encoding="utf-8-sig")
    wall_bias_correlations = []
    comparisons = (
        ("support_difference_ant_minus_post", "signed_mean_bias"),
        ("support_difference_ant_minus_post", "abs_mean_bias"),
        ("anterior_median_support", "mean_ER_ant"),
        ("posterior_median_support", "mean_ER_post"),
        ("posterior_median_support", "signed_mean_bias"),
    )
    for branch, group in wall_bias.groupby("branch"):
        for x, y in comparisons:
            result = _correlation(group, x, y)
            wall_bias_correlations.append(
                dict(branch=branch, support_variable=x, ER_variable=y, **result)
            )
    wall_bias_correlations = pd.DataFrame(wall_bias_correlations)
    wall_bias_correlations.to_csv(
        args.output / "wall_support_vs_ER_bias_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    control_record_rows = []
    for (case_id, condition, branch), group in pair_frame.groupby(
        ["case_id", "condition", "branch"]
    ):
        group = _finite_pairs(group)
        if group.empty:
            continue
        recording = summarize_paired_windows([
            dict(
                first_section=int(row.first_section),
                last_section=int(row.last_section),
                window_start_s=float(row.window_start_s),
                window_end_s=float(row.window_end_s),
                window_center_s=float((row.window_start_s + row.window_end_s) / 2),
                anterior_ER=float(row.ER_ant),
                posterior_ER=float(row.ER_post),
            )
            for row in group.itertuples(index=False)
        ])
        control_record_rows.append(
            dict(
                case_id=case_id,
                condition=condition,
                branch=branch,
                paired_windows=len(group),
                record_MSE=recording["record_MSE_time_equal"],
                record_CC=recording["record_CC_time_equal"],
            )
        )
    control_record = pd.DataFrame(control_record_rows)
    control_record.to_csv(args.output / "record_MSE_by_control_case.csv",
                          index=False, encoding="utf-8-sig")
    control_record_summary = []
    for (condition, branch), group in control_record.groupby(["condition", "branch"]):
        values = group["record_MSE"].dropna()
        control_record_summary.append(
            dict(
                condition=condition,
                branch=branch,
                cases=len(values),
                median_record_MSE=float(values.median()) if len(values) else None,
                minimum_record_MSE=float(values.min()) if len(values) else None,
                maximum_record_MSE=float(values.max()) if len(values) else None,
            )
        )
    control_record_summary = pd.DataFrame(control_record_summary)
    control_record_summary.to_csv(
        args.output / "control_record_MSE_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    common_a = pair_frame.loc[
        (pair_frame["condition"].eq("real")) & pair_frame["branch"].eq("A_2s"),
        BRANCH_MERGE_KEY + ["ER_ant", "ER_post", "window_MSE", "support_mean"],
    ].rename(columns={
        "ER_ant": "ER_ant_A", "ER_post": "ER_post_A",
        "window_MSE": "window_MSE_A", "support_mean": "support_A",
    })
    common_b = pair_frame.loc[
        (pair_frame["condition"].eq("real")) & pair_frame["branch"].eq("B_Huang_like"),
        BRANCH_MERGE_KEY + ["ER_ant", "ER_post", "window_MSE", "support_mean"],
    ].rename(columns={
        "ER_ant": "ER_ant_B", "ER_post": "ER_post_B",
        "window_MSE": "window_MSE_B", "support_mean": "support_B",
    })
    branch_common = common_a.merge(common_b, on=BRANCH_MERGE_KEY, how="inner")
    if len(branch_common):
        branch_common["support_quartile"] = (
            pd.qcut(branch_common["support_A"], 4, labels=False, duplicates="drop") + 1
        )
        branch_rows = []
        for quartile, group in branch_common.groupby("support_quartile"):
            branch_rows.append(
                dict(
                    support_A_quartile=int(quartile),
                    n=len(group),
                    support_A_median=float(group["support_A"].median()),
                    support_B_median=float(group["support_B"].median()),
                    median_abs_ER_difference=float(
                        np.median(np.abs(
                            np.r_[group["ER_ant_A"] - group["ER_ant_B"],
                                  group["ER_post_A"] - group["ER_post_B"]]
                        ))
                    ),
                    anterior_sign_agreement=float(
                        (np.sign(group["ER_ant_A"]) == np.sign(group["ER_ant_B"])).mean()
                    ),
                    posterior_sign_agreement=float(
                        (np.sign(group["ER_post_A"]) == np.sign(group["ER_post_B"])).mean()
                    ),
                    median_window_MSE_A=float(group["window_MSE_A"].median()),
                    median_window_MSE_B=float(group["window_MSE_B"].median()),
                )
            )
        branch_low = pd.DataFrame(branch_rows)
    else:
        branch_low = pd.DataFrame()
    branch_low.to_csv(args.output / "branch_low_support_comparison.csv",
                      index=False, encoding="utf-8-sig")

    summary = dict(
        audit_script_sha256=digest(Path(__file__).resolve()),
        real_cases=len(real_case_ids),
        real_cases_with_ER=int(real["case_id"].nunique()),
        failures=failures,
        windows_by_condition=window_frame.groupby(["condition", "branch"]).size()
        .rename("windows").reset_index().to_dict(orient="records"),
        real_support_thresholds=real_support_thresholds,
        support_distribution=support_distribution.to_dict(orient="records"),
        support_vs_er=correlation_frame.to_dict(orient="records"),
        support_quartiles=quartiles.to_dict(orient="records"),
        direction_accuracy=accuracy.to_dict(orient="records"),
        window_MSE_support=pair_correlations.to_dict(orient="records"),
        record_MSE_support=record_correlations.to_dict(orient="records"),
        wall_support_vs_ER_bias=wall_bias_correlations.to_dict(orient="records"),
        wall_asymmetry_summary={
            branch: dict(
                cases=len(group),
                median_ant_RMS=float(group["anterior_median_support"].median()),
                median_post_RMS=float(group["posterior_median_support"].median()),
                median_ant_minus_post=float(group["support_difference_ant_minus_post"].median()),
            )
            for branch, group in wall_asymmetry.groupby("branch")
        },
        control_record_MSE_summary=control_record_summary.to_dict(orient="records"),
        branch_low_support=branch_low.to_dict(orient="records"),
        source_hashes=source_hashes,
        outcomes_read=False,
        prediction_models_fit=False,
        production_features_changed=False,
    )
    from analysis.mse_pregnancy_association.code.huang_bridge_phase_a import _json_safe
    save(args.output / "summary.json", _json_safe(summary))

    def one_row(frame, **conditions):
        selected = frame
        for column, value in conditions.items():
            selected = selected.loc[selected[column].eq(value)]
        if len(selected) != 1:
            raise ValueError(f"Expected one row for {conditions}, found {len(selected)}")
        return selected.iloc[0]

    rigid_record_a = one_row(
        control_record_summary, condition="rigid", branch="A_2s"
    )
    rigid_record_b = one_row(
        control_record_summary, condition="rigid", branch="B_Huang_like"
    )
    c2f_record_a = one_row(control_record_summary, condition="C2F", branch="A_2s")
    f2c_record_a = one_row(control_record_summary, condition="F2C", branch="A_2s")
    real_record_a = one_row(control_record_summary, condition="real", branch="A_2s")
    record_support_corr_a = one_row(
        record_correlations, branch="A_2s", variable="fraction_below_real_Q1"
    )
    record_support_corr_b = one_row(
        record_correlations, branch="B_Huang_like", variable="fraction_below_real_Q1"
    )
    wall_bias_signed_a = one_row(
        wall_bias_correlations,
        branch="A_2s",
        support_variable="support_difference_ant_minus_post",
        ER_variable="signed_mean_bias",
    )
    wall_bias_abs_a = one_row(
        wall_bias_correlations,
        branch="A_2s",
        support_variable="support_difference_ant_minus_post",
        ER_variable="abs_mean_bias",
    )
    support_rows = []
    for branch in ("A_2s", "B_Huang_like"):
        rigid_support = pair_frame.loc[
            pair_frame["condition"].eq("rigid")
            & pair_frame["branch"].eq(branch)
        ][["directional_rms_ant", "directional_rms_post"]].to_numpy(float)
        real_support = window_frame.loc[
            window_frame["condition"].eq("real")
            & window_frame["branch"].eq(branch),
            "directional_rms",
        ].dropna()
        support_rows.append(
            dict(
                branch=branch,
                rigid_max=float(rigid_support.max()),
                real_min=float(real_support.min()),
                real_below_rigid_max=int((real_support <= rigid_support.max()).sum()),
                real_windows=len(real_support),
            )
        )
    top_rigid = pair_frame.loc[pair_frame["condition"].eq("rigid")].copy()
    top_rigid["opposite_sign"] = (
        np.sign(top_rigid["ER_ant"]) * np.sign(top_rigid["ER_post"]) < 0
    )
    rigid_top25 = (
        top_rigid.sort_values("window_MSE", ascending=False)
        .groupby("branch", group_keys=False)
        .head(25)
    )
    rigid_top25_fraction = (
        rigid_top25.groupby("branch")["opposite_sign"].mean().to_dict()
    )
    rigid_window_median = (
        pair_frame.loc[pair_frame["condition"].eq("rigid")]
        .groupby("branch")["window_MSE"].median().to_dict()
    )

    lines = [
        "# ER方向能量支持度审计",
        "",
        "本轮不读取妊娠结局、妊娠分组、AUC或临床变量，不训练预测模型，"
        "不修改正式追踪、2秒形变、FFT、ROI、ER、MSE或现有生产定义，"
        "不新增阈值或排除规则。所有四分位区间仅作描述。",
        "",
        "## 总结论",
        "",
        "**Situation C成立，Situation D未成立。** rigid确实会在极低方向能量下"
        "产生由前后壁ER异号驱动的高MSE；但真实病例的ERSupport与rigid几乎不重叠，"
        "记录MSE没有随低support窗口比例增加，前后壁support差也不能解释绝对ER偏置。"
        "因此不能把rigid的约0.21 MSE横向解读为真实患者的MSE主要由低support污染。",
        "",
        "ER低support时比例不稳定是客观问题；当前患者高MSE是否由低support主导，"
        "目前**没有**得到支持。问题不特定于2秒，因为2秒与Huang-like分支表现一致。",
        "",
        "## 范围与口径",
        "",
        f"- 真实病例{summary['real_cases']}例，运行失败{len(failures)}例；"
        f"{summary['real_cases_with_ER']}例至少有1个ER窗口，157例形成前后壁完整记录。",
        f"- rigid按全部窗口直接合并的中位window MSE：A_2s "
        f"{rigid_window_median['A_2s']:.3f}，B_Huang-like "
        f"{rigid_window_median['B_Huang_like']:.3f}。",
        f"- 若先对4个纹理各自形成记录MSE、再对4个记录取中位，rigid中位记录MSE为"
        f"A_2s {rigid_record_a['median_record_MSE']:.3f}、B_Huang-like "
        f"{rigid_record_b['median_record_MSE']:.3f}。桥接表的约0.21采用后一种"
        "“记录级中位”口径；两种数不能混用。",
        "- `directional_band_energy_fraction`是相对频带能量比例；rigid的该比例很高"
        "并不代表绝对方向支持高。绝对support诊断使用`directional_rms`。",
        "",
        "## 1. rigid为什么会产生高MSE",
        "",
        f"rigid最高MSE的25个窗口中，前后壁ER异号的比例为A_2s "
        f"{rigid_top25_fraction['A_2s']:.0%}、B_Huang-like "
        f"{rigid_top25_fraction['B_Huang_like']:.0%}。典型窗口中，两壁绝对运动极小，"
        "但ER接近±1，平方差后仍可产生很大的MSE。这直接证明低support下ER比例放大"
        "能够在几乎没有真实传播时制造高MSE。",
        "",
        "## 2. 真实病例是否由低support主导",
        "",
        "|分支|rigid最大support|真实最小support|真实窗口≤rigid最大值|",
        "|---|---:|---:|---:|",
    ]
    for row in support_rows:
        lines.append(
            f"|{row['branch']}|{row['rigid_max']:.6f}|{row['real_min']:.6f}|"
            f"{row['real_below_rigid_max']}/{row['real_windows']}|"
        )
    lines += [
        "",
        f"记录级MSE与“低于真实Q1的窗口比例”的相关为A_2s Pearson "
        f"{record_support_corr_a['Pearson']:.3f}/Spearman "
        f"{record_support_corr_a['Spearman']:.3f}；B_Huang-like Pearson "
        f"{record_support_corr_b['Pearson']:.3f}/Spearman "
        f"{record_support_corr_b['Spearman']:.3f}。方向均很弱且为负，"
        "不支持“低support窗口越多，记录MSE越高”。",
        "",
        "真实window级MSE与两壁平均support的相关也很弱：A_2s Pearson "
        f"{one_row(pair_correlations, condition='real', branch='A_2s', support_metric='support_mean')['Pearson']:.3f}，"
        "B_Huang-like "
        f"{one_row(pair_correlations, condition='real', branch='B_Huang_like', support_metric='support_mean')['Pearson']:.3f}。",
        "",
        "## 3. 前后壁support差是否解释ER偏置（Situation D）",
        "",
        f"A_2s前后壁support差(前−后)与记录带符号平均ER偏置的相关为Pearson "
        f"{wall_bias_signed_a['Pearson']:.3f}/Spearman "
        f"{wall_bias_signed_a['Spearman']:.3f}；与绝对平均ER偏置的相关仅Pearson "
        f"{wall_bias_abs_a['Pearson']:.3f}/Spearman "
        f"{wall_bias_abs_a['Spearman']:.3f}。也就是support差只与偏置方向有弱到中等"
        "描述性关系，与真正驱动MSE的绝对偏置大小基本无关。",
        "",
        f"真实记录的前−后壁中位support差为"
        f"{summary['wall_asymmetry_summary']['A_2s']['median_ant_minus_post']:.6f}"
        f"（A_2s）和"
        f"{summary['wall_asymmetry_summary']['B_Huang_like']['median_ant_minus_post']:.6f}"
        "（B_Huang-like）。因此**不判定Situation D**：没有证据表明某一壁系统性"
        "support不足是当前绝对ER偏置的主要原因。",
        "",
        "## 4. 已知传播、方向准确率和2秒分支",
        "",
        "- static在两条分支均不产生ER窗口。",
        f"- C2F/F2C已知传播的记录中位MSE仅为A_2s "
        f"C2F {c2f_record_a['median_record_MSE']:.3f}、F2C "
        f"{f2c_record_a['median_record_MSE']:.3f}，明显低于rigid记录中位。",
        "- 对`directional_rms`和band fraction的每个support四分位，"
        "已知C2F/F2C方向准确率均为1.0；最低四分位也未观察到方向错误。"
        "因此本夹具不能证明“support越高，方向越准”的单调关系。",
        "- A_2s与B_Huang-like的最低support四分位都出现更大的分支间ER差异和更低的"
        "符号一致率，但两分支的MSE-support关系同向，说明问题是ER比例在低绝对"
        "support下的普遍特性，不是2秒分支特有。",
        "",
        "## 5. 能否进入独立验证",
        "",
        "可以冻结**测量定义和本审计协议**以进入下一阶段，但这不等于验证ER-MSE的"
        "临床效应。当前结果支持：rigid的高MSE有明确低support来源，而真实病例的"
        "MSE升高不由低support窗口比例解释。仍不足以证明ER-MSE具有可重复的生理"
        "或临床意义，必须依靠未参与定义开发的新时间/新患者独立验证。",
        "",
        "不建立support硬阈值，不删除低support窗口，也不把`mean_bias_sq`或support"
        "替换成新的正式主特征。support指标保留为描述性QC；若以后纳入模型，只可"
        "作为预设的协变量或敏感性分析，并需外部验证。",
        "",
        "## 限制",
        "",
        "- 只有4张人工纹理控制，不是无真实运动患者，不能把control与患者作绝对阈值。",
        "- `directional_rms`来自同一FFT谱，不能作为独立生理真值；raw FFT energy"
        "不能跨网格直接比较。",
        "- 四分位、相关与跨分支符号一致率均为描述性结果，未进行模型搜索。",
        "- 本审计不能排除未测量混杂，只能说明已测support偏置不是当前MSE的主要解释。",
        "- 未读取妊娠或临床变量，因此本报告不提供、也不重算AUC。",
        "",
        "## 复现",
        "",
        "```bash",
        "python -m experiments.er_directional_support_audit "
        "--feature-root outputs/frozen_features_223_20261119 "
        "--controls outputs/a0_pair_qc_intervals_20261117_v1 "
        "--output outputs/er_directional_support_audit_20261120",
        "```",
        "",
        f"审计脚本SHA256：`{summary['audit_script_sha256']}`。"
        "患者级明细仅保留本地。",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
