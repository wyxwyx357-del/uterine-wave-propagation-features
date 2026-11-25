"""Verify the sealed hold-out measurements, then run the frozen validation."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from analysis.mse_pregnancy_association.code.frozen_feature_auc import digest, save

OUTCOME_LABELS = {"妊娠是": 1, "妊娠否": 0}


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _outcome_from_path(path):
    parts = Path(os.path.realpath(path)).parts
    values = [OUTCOME_LABELS[part] for part in parts if part in OUTCOME_LABELS]
    if len(values) != 1:
        raise ValueError("pregnancy_outcome_path_not_unique")
    period = next((part for part in parts if re.fullmatch(r"20\d{2}-\d{2}", part)), None)
    embryo_source = next(
        (part for part in parts if part in ("囊胚", "细胞胚")),
        "unspecified",
    )
    return values[0], period, embryo_source


def _bootstrap_mean_difference(left, right, iterations, seed):
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    if len(left) < 2 or len(right) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, float)
    for index in range(iterations):
        values[index] = (
            rng.choice(left, len(left), replace=True).mean()
            - rng.choice(right, len(right), replace=True).mean()
        )
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def _fit_logistic(feature, outcome):
    feature = np.asarray(feature, float)
    outcome = np.asarray(outcome, int)
    center = float(feature.mean())
    scale = float(feature.std(ddof=1))
    z = (feature - center) / scale
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000,
    )
    model.fit(z[:, None], outcome)
    beta = float(model.coef_[0, 0])
    probability = model.predict_proba(z[:, None])[:, 1]
    variance = float(
        np.sum((probability * (1 - probability)) * z**2)
    )
    if variance <= 1e-12:
        standard_error = np.inf
    else:
        standard_error = float(np.sqrt(1 / variance))
    return dict(
        n=len(outcome),
        positive=int(outcome.sum()),
        negative=int((1 - outcome).sum()),
        feature_mean=center,
        feature_SD=scale,
        OR_per_SD=float(np.exp(beta)),
        CI_low=float(np.exp(beta - norm.ppf(0.975) * standard_error)),
        CI_high=float(np.exp(beta + norm.ppf(0.975) * standard_error)),
        p=float(2 * norm.sf(abs(beta / standard_error))),
    )


def _bootstrap_auc(outcome, score, iterations, seed):
    outcome = np.asarray(outcome, int)
    score = np.asarray(score, float)
    positive = np.flatnonzero(outcome == 1)
    negative = np.flatnonzero(outcome == 0)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        index = np.r_[
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ]
        values.append(roc_auc_score(outcome[index], score[index]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    sealed = _load_json(args.frozen_manifest)
    if sealed.get("freeze_status") != "sealed_before_outcome_access":
        raise ValueError("Measurement manifest is not sealed")
    if sealed.get("outcomes_read") is not False:
        raise ValueError("Measurement manifest does not certify outcome blindness")
    if digest(args.measurement_csv) != sealed["frozen_measurement_sha256"]:
        raise ValueError("Frozen measurement file changed after seal")
    if digest(args.measurement_manifest) != sealed["source_measurement_manifest_sha256"]:
        raise ValueError("Source measurement manifest changed after seal")
    measurements = pd.read_csv(args.measurement_csv, dtype={"case_id": str})
    inputs = _load_json(args.input_manifest)["cases"]
    input_by_case = {row["case_id"]: row for row in inputs}
    rows = []
    for row in measurements.itertuples(index=False):
        item = row._asdict()
        if item["MSE"] is None or not np.isfinite(item["MSE"]):
            continue
        try:
            outcome, period, embryo_source = _outcome_from_path(
                input_by_case[item["case_id"]]["video_path"]
            )
        except (KeyError, ValueError):
            continue
        item["pregnancy"] = outcome
        item["acquisition_period"] = period
        item["acquisition_year"] = int(period[:4]) if period else None
        item["embryo_source"] = embryo_source
        item["source_code"] = hashlib.sha256(
            str(input_by_case[item["case_id"]]["video_path"]).encode("utf-8")
        ).hexdigest()[:12]
        rows.append(item)
    data = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    data.to_csv(
        args.output / "holdout_explanatory_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if (data["pregnancy"].nunique() if len(data) else 0) != 2:
        raise ValueError("Both pregnancy outcome classes are required")

    positive = data.loc[data["pregnancy"].eq(1), "MSE"].to_numpy(float)
    negative = data.loc[data["pregnancy"].eq(0), "MSE"].to_numpy(float)
    difference = float(positive.mean() - negative.mean())
    ci_low, ci_high = _bootstrap_mean_difference(
        positive, negative, args.bootstrap, args.seed
    )
    descriptive_rows = []
    for label, values in (("pregnancy", positive), ("non_pregnancy", negative)):
        descriptive_rows.append(dict(
            group=label,
            n=len(values),
            mean=float(values.mean()),
            SD=float(values.std(ddof=1)),
            median=float(np.median(values)),
            Q1=float(np.quantile(values, 0.25)),
            Q3=float(np.quantile(values, 0.75)),
            minimum=float(values.min()),
            maximum=float(values.max()),
        ))
    descriptive_rows.append(dict(
        group="difference_pregnancy_minus_nonpregnancy",
        n=len(data),
        mean=difference,
        SD=None,
        median=None,
        Q1=ci_low,
        Q3=ci_high,
        minimum=None,
        maximum=None,
    ))
    descriptive = pd.DataFrame(descriptive_rows)

    logistic = _fit_logistic(data["MSE"], data["pregnancy"])
    mw = mannwhitneyu(positive, negative, alternative="two-sided")
    auc = float(roc_auc_score(data["pregnancy"], data["MSE"]))
    auc_low, auc_high = _bootstrap_auc(
        data["pregnancy"], data["MSE"], args.bootstrap, args.seed + 1
    )
    primary_rows = [
        dict(analysis="MSE_descriptive", term="pregnancy_minus_nonpregnancy",
             estimate=difference, CI_low=ci_low, CI_high=ci_high,
             p=float(mw.pvalue), n=len(data)),
        dict(analysis="logistic_MSE_only", term="MSE_per_SD",
             estimate=logistic["OR_per_SD"], CI_low=logistic["CI_low"],
             CI_high=logistic["CI_high"], p=logistic["p"], n=logistic["n"]),
    ]
    primary = pd.DataFrame(primary_rows)
    secondary = pd.DataFrame([dict(
        model="MSE_only",
        AUC=auc,
        AUC_low=auc_low,
        AUC_high=auc_high,
        n=len(data),
        positive=int(data["pregnancy"].sum()),
        negative=int((1 - data["pregnancy"]).sum()),
        clinical_AUC=None,
        clinical_plus_MSE_AUC=None,
        delta_AUC=None,
        note="No outcome-independent fixed clinical covariate join was available for this hold-out.",
    )])
    year_summary = (
        data.groupby(["acquisition_period", "acquisition_year", "pregnancy"])
        .agg(n=("case_id", "size"), MSE_median=("MSE", "median"))
        .reset_index()
    )
    source_summary = (
        data.groupby(["embryo_source", "pregnancy"])
        .agg(n=("case_id", "size"), MSE_median=("MSE", "median"))
        .reset_index()
    )
    primary.to_csv(args.output / "holdout_primary_analysis.csv",
                   index=False, encoding="utf-8-sig")
    descriptive.to_csv(args.output / "holdout_primary_descriptives.csv",
                       index=False, encoding="utf-8-sig")
    secondary.to_csv(args.output / "holdout_auc_secondary.csv",
                     index=False, encoding="utf-8-sig")
    year_summary.to_csv(args.output / "holdout_year_summary.csv",
                        index=False, encoding="utf-8-sig")
    source_summary.to_csv(args.output / "holdout_source_summary.csv",
                          index=False, encoding="utf-8-sig")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    axis.boxplot(
        [negative, positive],
        widths=0.55,
        showfliers=False,
    )
    axis.set_xticks([1, 2])
    axis.set_xticklabels(["Non-pregnancy", "Pregnancy"])
    rng = np.random.default_rng(args.seed + 2)
    for location, values in ((1, negative), (2, positive)):
        axis.scatter(
            location + rng.uniform(-0.10, 0.10, len(values)),
            values,
            s=24,
            alpha=0.75,
            color="#365f91" if location == 1 else "#b45f06",
        )
    axis.set_ylabel("Complete-record ER-MSE")
    axis.set_title("Short-duration internal hold-out")
    figure.tight_layout()
    figure.savefig(args.output / "mse_by_pregnancy.png", dpi=180)
    plt.close(figure)

    false_positive, true_positive, _ = roc_curve(
        data["pregnancy"], data["MSE"]
    )
    figure, axis = plt.subplots(figsize=(4.8, 4.4))
    axis.plot(false_positive, true_positive, color="#365f91", linewidth=2)
    axis.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    axis.set_xlabel("1 - specificity")
    axis.set_ylabel("Sensitivity")
    axis.set_title(f"MSE-only ROC (AUC={auc:.3f})")
    figure.tight_layout()
    figure.savefig(args.output / "mse_only_roc.png", dpi=180)
    plt.close(figure)

    if logistic["OR_per_SD"] > 1:
        if logistic["CI_low"] > 1:
            status = "directional_replication"
        else:
            status = "same_direction_but_imprecise"
    else:
        status = "no_replication"
    report = [
        "# 短视频独立患者级内部hold-out验证",
        "",
        "影像测量已先冻结并通过SHA256核验；本分析在冻结后读取妊娠结果。"
        "该队列只能称为 independent patient-level internal hold-out，"
        "不是external、temporal或prospective validation。",
        "",
        "## 主要结果",
        "",
        f"- 可分析且MSE可计算病例：{len(data)}；妊娠{len(positive)}例、"
        f"未妊娠{len(negative)}例。",
        "- 33例MSE中另1例来源为未分类视频，没有可用妊娠分组，"
        "因此未进入验证。",
        f"- MSE妊娠组均值{positive.mean():.6f}，未妊娠组{negative.mean():.6f}；"
        f"差值{positive.mean()-negative.mean():+.6f}，"
        f"bootstrap 95% CI [{ci_low:+.6f}, {ci_high:+.6f}]。",
        f"- MSE-only logistic：OR/SD={logistic['OR_per_SD']:.3f}，"
        f"95% CI [{logistic['CI_low']:.3f}, {logistic['CI_high']:.3f}]，"
        f"p={logistic['p']:.4f}。",
        f"- MSE-only AUC={auc:.3f}，bootstrap 95% CI "
        f"[{auc_low:.3f}, {auc_high:.3f}]（secondary）。",
        "- 分布图：`mse_by_pregnancy.png`；ROC：`mse_only_roc.png`。",
        f"- 判定：**{status}**。",
        "",
        "## 解释边界",
        "",
        "- 样本量小，置信区间宽度比点估计更重要。",
        "- 固定7项临床变量未在本hold-out中建立可审计的完整join，"
        "因此没有强行填入临床调整模型或clinical AUC；AUC仅报告MSE-only。",
        "- 不按cutoff、AUC或结果删病例，不修改任何影像参数。",
        "",
        "## 来源与年份",
        "",
    ]
    report.extend(
        f"- {row.acquisition_period} / pregnancy={int(row.pregnancy)}: "
        f"n={int(row.n)}, MSE median={row.MSE_median:.6f}"
        for row in year_summary.itertuples(index=False)
    )
    report += ["", "来源类型："]
    report.extend(
        f"- {row.embryo_source} / pregnancy={int(row.pregnancy)}: "
        f"n={int(row.n)}, MSE median={row.MSE_median:.6f}"
        for row in source_summary.itertuples(index=False)
    )
    (args.output / "REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    summary = dict(
        status=status,
        n=len(data),
        positive=len(positive),
        negative=len(negative),
        mean_difference=difference,
        mean_difference_CI=[ci_low, ci_high],
        logistic=logistic,
        AUC=auc,
        AUC_CI=[auc_low, auc_high],
        source_summary=source_summary.to_dict(orient="records"),
        frozen_measurement_sha256=sealed["frozen_measurement_sha256"],
        outcomes_read_after_measurement_freeze=True,
        clinical_adjustment_available=False,
        cutoff_selection=False,
        model_search=False,
    )
    save(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--measurement-csv", type=Path, required=True)
    parser.add_argument("--measurement-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261120)
    run(parser.parse_args())
