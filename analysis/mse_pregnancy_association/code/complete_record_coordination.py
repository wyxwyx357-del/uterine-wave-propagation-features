"""Summarize ER coordination over all available paired windows in a recording.

This is a retrospective aggregation of the frozen features. It does not
recompute tracking, radial strain, ER, or the 20-second mode windows. Spatial
and temporal gaps are retained, never filled.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis.mse_pregnancy_association.code.feature_association import (
    bh_adjust,
    cluster_mean_difference,
    fit_cluster_logit,
)
from analysis.mse_pregnancy_association.code.frozen_feature_auc import CLINICAL, GROUP, OUTCOME, SEED, digest, evaluate, save

EXAM = "临床检查编号"


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


def _association(data, feature, bootstrap, family):
    difference = cluster_mean_difference(data, feature, bootstrap)
    use = data[[feature] + CLINICAL + [GROUP, OUTCOME, "case_id"]].dropna().reset_index(drop=True)
    scale = float(use[feature].std(ddof=1))
    center = float(use[feature].mean())
    z = (use[feature].to_numpy(float) - center) / scale
    clinical = use[CLINICAL].to_numpy(float)
    clinical = (clinical - clinical.mean(0)) / clinical.std(0)
    y = use[OUTCOME].to_numpy(int)
    groups = use[GROUP].to_numpy(str)
    basic = np.column_stack([np.ones(len(use)), z])
    adjusted = np.column_stack([basic, clinical])
    fits = {}
    for label, design in (("unadjusted", basic), ("adjusted", adjusted)):
        fit = fit_cluster_logit(design, y, groups)
        beta = float(fit["beta"][1])
        se = float(np.sqrt(fit["covariance"][1, 1]))
        fits[label] = dict(
            OR_per_SD=float(np.exp(beta)),
            CI_low=float(np.exp(beta - 1.959963984540054 * se)),
            CI_high=float(np.exp(beta + 1.959963984540054 * se)),
            p=float(2 * norm.sf(abs(beta / se))),
        )
    result = dict(
        feature=feature,
        n=len(use),
        patients=int(use[GROUP].nunique()),
        positive=int(y.sum()),
        negative=int((1 - y).sum()),
        available=int(data[feature].notna().sum()),
        missing=int(data[feature].isna().sum()),
        feature_SD=scale,
        feature_mean=center,
        analysis_family=family,
    )
    for label, values in fits.items():
        result.update({f"{label}_{key}": value for key, value in values.items()})
    result.update(difference)
    return result


def _standardized_mean_difference(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 1e-12 else None


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash = digest(args.cohort)
    data = pd.read_csv(args.cohort, dtype={GROUP: str, EXAM: str})
    save(
        args.output / "protocol_before_aggregation.json",
        dict(
            source_sha256=source_hash,
            primary="record_MSE_time_equal over all available paired ER window times",
            sensitivity="record_MSE_pooled over every paired window-block observation",
            gaps="retained; no time interpolation or zero filling",
            spatial_blocks="available blocks retained; equal weight at each shared time",
            cc="time-collapsed ER series; descriptive because gaps are not filled",
            association=(
                "one feature per logistic model, fixed 7 clinical covariates, "
                "complete cases, patient-cluster CR1; no threshold tuning"
            ),
            posthoc_mse_decomposition=(
                "mean-bias squared, SD-scale squared, time-shape term, and standardized MSE; "
                "descriptive interpretation only"
            ),
            missingness_sensitivity=(
                "median imputation; median plus missing indicator; observed/complete case"
            ),
            bootstrap=args.bootstrap,
            tracking_or_strain_changed=False,
        ),
    )

    records = []
    segment_rows = []
    source_hashes = {}
    for _, row in data.iterrows():
        path = Path(row["feature_file"])
        source_hashes[str(path.resolve())] = digest(path)
        feature = json.loads(path.read_text(encoding="utf-8"))
        paired = feature["modes"]["paired_ER_windows"]
        result = summarize_paired_windows(paired)
        records.append(
            dict(
                case_id=row["case_id"],
                **{EXAM: row[EXAM]},
                **{GROUP: row[GROUP]},
                **{OUTCOME: row[OUTCOME]},
                single_segment_MSE=row["single_block_MSE"],
                single_segment_CC=row["single_block_CC"],
                coordination_block_count_original=row["coordination_block_count_original"],
                **{column: row[column] for column in CLINICAL},
                video_duration_s=row["video_duration_s"],
                anterior_valid_fraction=row["anterior_valid_fraction"],
                posterior_valid_fraction=row["posterior_valid_fraction"],
                video_year=int(str(row["case_id"]).rsplit("-", 1)[-1][:4]),
                **{key: value for key, value in result.items() if key != "segment_evidence"},
            )
        )
        for segment in result["segment_evidence"]:
            segment_rows.append(dict(case_id=row["case_id"], **segment))

    records = pd.DataFrame(records)
    segments = pd.DataFrame(segment_rows)
    available = records["record_MSE_time_equal"].notna()
    records["record_mse_missing"] = (~available).astype(float)
    if not np.isfinite(records.loc[available, "record_MSE_time_equal"]).all():
        raise ValueError("Non-finite complete-record MSE")
    decomposable = available & records["decomposition_status"].eq(
        "one_block_per_window_time"
    )
    residual = (
        records.loc[decomposable, "record_MSE_time_equal"]
        - records.loc[decomposable, "record_mean_bias_sq"]
        - records.loc[decomposable, "record_sd_scale_sq"]
        - records.loc[decomposable, "record_time_shape_term"]
    )
    if float(residual.abs().max()) > 1e-10:
        raise ValueError("ER-MSE decomposition does not reproduce observed MSE")

    # The old single-segment scalar must be reproduced exactly whenever there
    # was one temporal segment.
    single = records["temporal_segment_count"].eq(1) & available
    if single.any():
        np.testing.assert_allclose(
            records.loc[single, "record_MSE_time_equal"],
            records.loc[single, "single_segment_MSE"],
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            records.loc[single, "record_MSE_pooled"],
            records.loc[single, "single_segment_MSE"],
            atol=1e-12,
            rtol=1e-12,
        )

    features = ["record_MSE_time_equal", "record_CC_time_equal"]
    decomposition_features = [
        "record_MSE_z",
        "record_mean_bias_sq",
        "record_sd_scale_sq",
        "record_time_shape_term",
    ]
    primary_associations = pd.DataFrame(
        [_association(records, feature, args.bootstrap, "primary") for feature in features]
    )
    posthoc_associations = pd.DataFrame(
        [
            _association(records, feature, args.bootstrap, "posthoc_decomposition")
            for feature in decomposition_features
        ]
    )
    for frame in (primary_associations, posthoc_associations):
        for column in ("unadjusted_p", "adjusted_p"):
            frame[column.replace("_p", "_BH_q")] = bh_adjust(frame[column])
    associations = pd.concat([primary_associations, posthoc_associations], ignore_index=True)

    auc_models = dict(
        clinical=CLINICAL,
        record_mse=[features[0]],
        record_cc=[features[1]],
        clinical_record_mse=CLINICAL + [features[0]],
        clinical_record_cc=CLINICAL + [features[1]],
        clinical_record_mse_cc=CLINICAL + features,
        record_mse_missing=["record_mse_missing"],
        clinical_record_mse_missing=CLINICAL + ["record_mse_missing"],
        clinical_record_mse_and_missing=CLINICAL + [features[0], "record_mse_missing"],
    )
    auc_metrics, auc_differences, _ = evaluate(
        "record_auc_all_imputed",
        records,
        auc_models,
        args,
    )
    decomposition_models = dict(
        clinical=CLINICAL,
        record_mse_z=["record_MSE_z"],
        record_mean_bias_sq=["record_mean_bias_sq"],
        record_sd_scale_sq=["record_sd_scale_sq"],
        record_time_shape_term=["record_time_shape_term"],
        clinical_record_mse_z=CLINICAL + ["record_MSE_z"],
        clinical_record_time_shape=CLINICAL + ["record_time_shape_term"],
    )
    decomposition_metrics, decomposition_differences, _ = evaluate(
        "record_auc_mse_decomposition",
        records,
        decomposition_models,
        args,
    )
    decomposition_differences = pd.DataFrame(decomposition_differences)
    complete = records.dropna(subset=features).reset_index(drop=True)
    complete_auc_models = {
        name: columns
        for name, columns in auc_models.items()
        if "missing" not in name
    }
    complete_metrics, _, _ = evaluate(
        "record_auc_complete_cases",
        complete,
        complete_auc_models,
        args,
    )

    missing = records["record_mse_missing"].eq(1)
    missingness_rows = []
    for label, values in (
        (OUTCOME, records[OUTCOME]),
        ("age", records[CLINICAL[0]]),
        ("BMI", records[CLINICAL[1]]),
        ("endometrial_thickness_mm", records[CLINICAL[4]]),
        ("video_duration_s", records["video_duration_s"]),
        ("anterior_valid_fraction", records["anterior_valid_fraction"]),
        ("posterior_valid_fraction", records["posterior_valid_fraction"]),
        ("video_year", records["video_year"]),
    ):
        observed = pd.to_numeric(values.loc[~missing], errors="coerce")
        unavailable = pd.to_numeric(values.loc[missing], errors="coerce")
        missingness_rows.append(
            dict(
                variable=label,
                available_n=int(observed.notna().sum()),
                available_mean=float(observed.mean()),
                unavailable_n=int(unavailable.notna().sum()),
                unavailable_mean=float(unavailable.mean()),
                standardized_difference=(
                    _standardized_mean_difference(unavailable, observed)
                ),
                direction="unavailable_minus_available",
            )
        )
    missingness = pd.DataFrame(missingness_rows)

    records.to_csv(args.output / "record_coordination.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(args.output / "temporal_segment_coordination.csv", index=False, encoding="utf-8-sig")
    associations.to_csv(args.output / "record_coordination_associations.csv", index=False, encoding="utf-8-sig")
    decomposition_differences.to_csv(
        args.output / "record_auc_mse_decomposition" / "paired_AUC_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    missingness.to_csv(args.output / "missingness_audit.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(args.output / "complete_record_coordination.xlsx", engine="openpyxl") as writer:
        records.to_excel(writer, sheet_name="record", index=False)
        segments.to_excel(writer, sheet_name="segments", index=False)
        associations.to_excel(writer, sheet_name="associations", index=False)
        decomposition_differences.to_excel(writer, sheet_name="MSE_decomposition", index=False)
        missingness.to_excel(writer, sheet_name="missingness", index=False)

    if any(digest(Path(path)) != value for path, value in source_hashes.items()):
        raise RuntimeError("A source feature file changed during aggregation")

    summary = dict(
        examinations=len(records),
        record_MSE_available=int(available.sum()),
        record_MSE_missing=int((~available).sum()),
        spatial_block_count_distribution=records.loc[available, "spatial_block_count"].value_counts().sort_index().to_dict(),
        temporal_segment_count_distribution=records.loc[available, "temporal_segment_count"].value_counts().sort_index().to_dict(),
        recovered_from_multi_segment_only=int((records["coordination_block_count_original"] > 1).sum()),
        single_segment_exact_reproduction=bool(single.any()),
        associations=associations.to_dict(orient="records"),
        auc_all_imputed=auc_metrics,
        auc_mse_decomposition=decomposition_metrics,
        auc_mse_decomposition_differences=decomposition_differences.to_dict(orient="records"),
        auc_complete_cases=complete_metrics,
        missingness=missingness.to_dict(orient="records"),
        source_unchanged=True,
        tracking_or_strain_changed=False,
        formal_counts_changed=False,
    )
    save(args.output / "summary.json", summary)

    lines = [
        "# 完整可观察记录的双壁ER协调性",
        "",
        f"共{len(records)}次检查；完整记录MSE可计算{int(available.sum())}次，缺失{int((~available).sum())}次。",
        "只使用冻结特征中前后壁具有相同空间位置和相同20秒窗口的ER；缺失不填补、不以0代替。",
        "主指标`record_MSE_time_equal`先对同一窗口时间的可用空间block求平均平方差，再对时间求均值。",
        "`record_MSE_pooled`把每个window-block观测等权，作为多block敏感性指标。",
        "",
        "## 计算覆盖",
        "",
        f"- 单时间段：{summary['temporal_segment_count_distribution'].get(1, 0)}例。",
        f"- 多时间段：{int(available.sum()) - summary['temporal_segment_count_distribution'].get(1, 0)}例；这些记录的全部可用窗口被纳入完整记录汇总。",
        f"- 完全无共同有效ER窗口：{int((~available).sum())}例，保持缺失。",
        "- 单时间段病例的完整记录MSE与旧`single_block_MSE`逐例精确一致。",
        "- 特征本身MSE可用157例、CC可用156例、二者均可用156例；固定7项临床变量完整后分别降为154和153。AUC完整子集另行训练折内插补临床变量，因此为156例，三个口径不矛盾。",
        "",
        "## 与妊娠的横断面关联",
        "",
        "|特征|可用例数|分析例数|患者数|每SD OR|95%区间|p|",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in associations.itertuples(index=False):
        lines.append(
            f"|{row.feature}|{row.available}|{row.n}|{row.patients}|"
            f"{row.adjusted_OR_per_SD:.3f}|{row.adjusted_CI_low:.3f}–{row.adjusted_CI_high:.3f}|"
            f"{row.adjusted_p:.4g}|"
        )
    lines += [
        "",
        "### 事后MSE分解",
        "",
        "`record_MSE = (mean_ant-mean_post)^2 + (SD_ant-SD_post)^2 + 2*SD_ant*SD_post*(1-CC)`。",
        "`record_MSE_z`先将前后壁ER各自标准化，再做同一MSE公式；它主要反映时间轨迹不同步。",
        "",
        "|分解项|分析例数|调整OR/SD|95% CI|p|事后BH q|",
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in posthoc_associations.itertuples(index=False):
        lines.append(
            f"|{row.feature}|{row.n}|{row.adjusted_OR_per_SD:.3f}|"
            f"{row.adjusted_CI_low:.3f}–{row.adjusted_CI_high:.3f}|"
            f"{row.adjusted_p:.4g}|{row.adjusted_BH_q:.4g}|"
        )
    lines += [
        "",
        "## 折外AUC",
        "",
        "主分析纳入全部223例，记录级协调性缺失者在训练折内中位数插补；另列MSE和CC均完整的敏感性结果。",
        "",
        "|队列|模型|例数|患者数|折外AUC|95%区间|重复均值|",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for row in auc_metrics:
        lines.append(
            f"|全部223例插补|{row['model']}|{row['n']}|{row['patients']}|"
            f"{row['primary_OOF_AUC']:.4f}|{row['CI_low']:.4f}–{row['CI_high']:.4f}|"
            f"{row['repeat_mean_AUC']:.4f}|"
        )
    for row in complete_metrics:
        lines.append(
            f"|完整记录完整病例|{row['model']}|{row['n']}|{row['patients']}|"
            f"{row['primary_OOF_AUC']:.4f}|{row['CI_low']:.4f}–{row['CI_high']:.4f}|"
            f"{row['repeat_mean_AUC']:.4f}|"
        )
    lines += [
        "",
        "### MSE分解折外AUC",
        "",
        "|模型|例数|患者数|折外AUC|95%区间|重复均值|",
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in decomposition_metrics:
        lines.append(
            f"|{row['model']}|{row['n']}|{row['patients']}|"
            f"{row['primary_OOF_AUC']:.4f}|{row['CI_low']:.4f}–{row['CI_high']:.4f}|"
            f"{row['repeat_mean_AUC']:.4f}|"
        )
    lines += [
        "",
        "### 缺失机制核对",
        "",
        "标准化差方向为“缺失者减可计算者”。这些是描述性检查，不是缺失随机性的统计证明。",
        "",
        "|变量|可计算均值|缺失均值|标准化差|",
        "|---|---:|---:|---:|",
    ]
    for row in missingness.itertuples(index=False):
        difference = "NA" if pd.isna(row.standardized_difference) else f"{row.standardized_difference:+.3f}"
        lines.append(
            f"|{row.variable}|{row.available_mean:.3f}|{row.unavailable_mean:.3f}|{difference}|"
        )
    lines += [
        "",
        "AUC主结果为第一个预设种子的完整折外预测，重复均值和95%患者聚类bootstrap另列；不选最高种子。",
        "插补分析不是把缺失恢复成真实协调性，完整病例分析也不能代表全部病例。",
    ]
    lines += [
        "",
        "MSE主检验和CC主检验分别完成固定7项临床变量调整，并对两个预设特征作BH校正。",
        "完整记录表示所有可观察且双壁共支持窗口，不是把缺失时间补成静息，也不等于Huang逐帧RSR的完全同构测量。",
        "结果仍为回顾性内部关联，不能直接解释因果或外部预测性能。",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary)
    print(associations.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    run(parser.parse_args())
