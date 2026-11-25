"""Outcome comparison after the Huang bridge Phase A protocol is frozen."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from analysis.mse_pregnancy_association.code.feature_association import fit_cluster_logit
from analysis.mse_pregnancy_association.code.frozen_feature_auc import (
    CLINICAL,
    EXAM,
    GROUP,
    OUTCOME,
    SEED,
    bootstrap_oof,
    digest,
    evaluate,
    save,
)

PHASE_A_SCRIPT_SHA256 = "E5FB580D9198DCA03CE53A71E8746D947C38A1A59B980C3CBC51C56D096E574E"
PHASE_A_RESULTS_SHA256 = "2DDEAB28F9CFF1C8570C017EBA2E9C66118CCCC8680C798742781171FACCD017"


def association(data, feature):
    use = data[[feature] + CLINICAL + [GROUP, OUTCOME, "case_id"]].dropna().reset_index(drop=True)
    z = (use[feature].to_numpy(float) - use[feature].mean()) / use[feature].std(ddof=1)
    clinical = use[CLINICAL].to_numpy(float)
    clinical = (clinical - clinical.mean(0)) / clinical.std(0)
    design = np.column_stack([np.ones(len(use)), z, clinical])
    fit = fit_cluster_logit(design, use[OUTCOME].to_numpy(int), use[GROUP].to_numpy(str))
    beta = float(fit["beta"][1])
    se = float(np.sqrt(fit["covariance"][1, 1]))
    return dict(
        feature=feature,
        n=len(use),
        patients=int(use[GROUP].nunique()),
        positive=int(use[OUTCOME].sum()),
        negative=int((1 - use[OUTCOME]).sum()),
        OR_per_SD=float(np.exp(beta)),
        CI_low=float(np.exp(beta - 1.959963984540054 * se)),
        CI_high=float(np.exp(beta + 1.959963984540054 * se)),
        p=float(2 * norm.sf(abs(beta / se))),
        feature_SD=float(use[feature].std(ddof=1)),
    )


def descriptives(data, branch):
    feature = f"{branch}_MSE"
    rows = []
    for outcome in (0, 1):
        values = data.loc[data[OUTCOME].eq(outcome), feature].dropna()
        rows.append(
            dict(
                branch=branch,
                outcome=outcome,
                outcome_label="clinical_pregnancy" if outcome else "non_pregnancy",
                n=len(values),
                mean=float(values.mean()),
                median=float(values.median()),
                SD=float(values.std(ddof=1)),
                Q1=float(values.quantile(0.25)),
                Q3=float(values.quantile(0.75)),
                pregnancy_minus_nonpregnancy=None,
            )
        )
    rows[0]["pregnancy_minus_nonpregnancy"] = float(rows[1]["mean"] - rows[0]["mean"])
    rows[1]["pregnancy_minus_nonpregnancy"] = rows[0]["pregnancy_minus_nonpregnancy"]
    return rows


def _mse_direction(rows):
    return float(rows[1]["mean"] - rows[0]["mean"])


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    phase_a_path = Path(__file__).with_name("huang_bridge_phase_a.py")
    phase_a_script_hash = digest(phase_a_path)
    phase_a_result_hash = digest(args.phase_a / "method_comparison.json")
    frozen_path = args.phase_a / "HUANG_BRIDGE_METHOD_FROZEN.md"
    if not frozen_path.exists():
        raise RuntimeError("HUANG_BRIDGE_METHOD_FROZEN.md is missing")
    if phase_a_script_hash.upper() != PHASE_A_SCRIPT_SHA256:
        raise RuntimeError("Phase A bridge code hash does not match the frozen protocol")
    if phase_a_result_hash.upper() != PHASE_A_RESULTS_SHA256:
        raise RuntimeError("Phase A bridge results hash does not match the frozen protocol")

    protocol = dict(
        phase_a_script_sha256=phase_a_script_hash,
        phase_a_results_sha256=phase_a_result_hash,
        frozen_protocol_sha256=digest(frozen_path),
        primary_comparison=(
            "Pregnancy MSE mean minus nonpregnancy MSE mean for "
            "A_2s aligned and B_Huang_like"
        ),
        association="fixed 7 clinical covariates; no variable selection or model search",
        auc="secondary; same patient-grouped folds as existing analysis",
        decomposition="post-hoc mechanistic comparison; no production feature replacement",
        outcomes_read_after_this_protocol=True,
    )
    save(args.output / "phase_b_protocol_before_outcome.json", protocol)

    bridge = pd.read_csv(args.phase_a / "case_level_bridge.csv", dtype={"case_id": str})
    outcome = pd.read_csv(args.cohort, dtype={GROUP: str, EXAM: str})
    outcome["case_id"] = outcome["case_id"].astype(str)
    keep = ["case_id", EXAM, GROUP, OUTCOME] + CLINICAL
    data = bridge.merge(
        outcome[keep],
        on="case_id",
        how="inner",
        validate="one_to_one",
    )
    if len(data) != len(bridge):
        raise ValueError("Outcome cohort does not cover all Phase A cases")
    data = data.rename(
        columns={
            "A_official_record_MSE_time_equal": "A_official_MSE",
            "A_2s_record_MSE_time_equal": "A_2s_MSE",
            "B_Huang_like_record_MSE_time_equal": "B_Huang_like_MSE",
        }
    )
    direction_rows = descriptives(data, "A_2s") + descriptives(data, "B_Huang_like")
    direction = pd.DataFrame(direction_rows)
    a_difference = _mse_direction(direction_rows[:2])
    b_difference = _mse_direction(direction_rows[2:])
    if a_difference > 0 and b_difference < 0:
        situation = 1
    elif a_difference > 0 and b_difference > 0:
        situation = 2
    elif a_difference <= 0 and b_difference <= 0:
        situation = "both_nonpositive"
    else:
        situation = "other"

    association_features = ["A_2s_MSE", "B_Huang_like_MSE"]
    decomposition_file = pd.read_csv(args.phase_a / "MSE_decomposition_comparison.csv")
    pivot = decomposition_file.pivot(index="case_id", columns="branch")
    decomposition = pd.DataFrame(index=pivot.index)
    for component in ("mean_bias_sq", "sd_scale_sq", "time_shape_term", "MSE_z", "CC"):
        for branch in ("A_2s", "B_Huang_like"):
            decomposition[f"{branch}_{component}"] = pivot[(component, branch)]
    decomposition = decomposition.reset_index()
    data = data.merge(decomposition, on="case_id", how="left", validate="one_to_one")
    association_features += [
        "A_2s_mean_bias_sq",
        "A_2s_time_shape_term",
        "A_2s_MSE_z",
        "B_Huang_like_mean_bias_sq",
        "B_Huang_like_time_shape_term",
        "B_Huang_like_MSE_z",
    ]
    associations = pd.DataFrame(
        [association(data, feature) for feature in association_features]
    )

    auc_models = dict(
        clinical=CLINICAL,
        MSE_2s=["A_2s_MSE"],
        clinical_MSE_2s=CLINICAL + ["A_2s_MSE"],
        MSE_Huang_like=["B_Huang_like_MSE"],
        clinical_MSE_Huang_like=CLINICAL + ["B_Huang_like_MSE"],
    )
    auc_metrics, auc_differences, auc_predictions = evaluate(
        "huang_bridge_auc",
        data,
        auc_models,
        args,
    )
    draws = bootstrap_oof(data, auc_predictions, args.bootstrap, SEED + 1000)
    direct_differences = []
    for added, reference in (
        ("MSE_Huang_like", "MSE_2s"),
        ("clinical_MSE_Huang_like", "clinical_MSE_2s"),
    ):
        distribution = draws[added] - draws[reference]
        low, high = np.quantile(distribution, [0.025, 0.975])
        direct_differences.append(
            dict(
                comparison=f"{added} minus {reference}",
                delta_AUC=float(
                    roc_auc_score(data[OUTCOME], auc_predictions[added])
                    - roc_auc_score(data[OUTCOME], auc_predictions[reference])
                ),
                CI_low=float(low),
                CI_high=float(high),
            )
        )
    direct_differences = pd.DataFrame(direct_differences)

    direction.to_csv(args.output / "outcome_comparison.csv", index=False, encoding="utf-8-sig")
    associations.to_csv(args.output / "adjusted_associations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(auc_metrics).to_csv(args.output / "AUC_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(auc_differences).to_csv(
        args.output / "AUC_paired_differences.csv", index=False, encoding="utf-8-sig"
    )
    direct_differences.to_csv(
        args.output / "AUC_direct_branch_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    data.to_csv(args.output / "phase_b_case_level.csv", index=False, encoding="utf-8-sig")

    summary = dict(
        situation=situation,
        A_2s_pregnancy_minus_nonpregnancy_MSE=a_difference,
        Huang_like_pregnancy_minus_nonpregnancy_MSE=b_difference,
        direction_rows=direction.to_dict(orient="records"),
        adjusted_associations=associations.to_dict(orient="records"),
        AUC_metrics=auc_metrics,
        AUC_differences=auc_differences,
        AUC_direct_branch_differences=direct_differences.to_dict(orient="records"),
        frozen_protocol_sha256=protocol["frozen_protocol_sha256"],
        no_model_or_feature_search=True,
        formal_features_changed=False,
    )
    save(args.output / "summary.json", summary)

    lines = [
        "# Huang桥接：Phase B结局比较",
        "",
        "Phase A方法和结果哈希已在读取结局前核对。",
        "",
        f"Situation：{situation}",
        "",
        "## MSE方向",
        "",
        "|分支|结局|n|均值|中位数|SD|IQR|",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in direction.itertuples(index=False):
        lines.append(
            f"|{row.branch}|{row.outcome_label}|{row.n}|{row.mean:.4f}|{row.median:.4f}|"
            f"{row.SD:.4f}|{row.Q1:.4f}–{row.Q3:.4f}|"
        )
    lines += [
        "",
        f"A 2s：妊娠−非妊娠MSE={a_difference:+.4f}。",
        f"Huang-like：妊娠−非妊娠MSE={b_difference:+.4f}。",
        "",
        "## 固定临床调整",
        "",
        "|特征|n|患者|OR/SD|95% CI|p|",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in associations.itertuples(index=False):
        lines.append(
            f"|{row.feature}|{row.n}|{row.patients}|{row.OR_per_SD:.3f}|"
            f"{row.CI_low:.3f}–{row.CI_high:.3f}|{row.p:.4g}|"
        )
    lines += [
        "",
        "## 次要AUC",
        "",
        "|模型|n|患者|折外AUC|95%区间|重复均值|",
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in auc_metrics:
        lines.append(
            f"|{row['model']}|{row['n']}|{row['patients']}|{row['primary_OOF_AUC']:.4f}|"
            f"{row['CI_low']:.4f}–{row['CI_high']:.4f}|{row['repeat_mean_AUC']:.4f}|"
        )
    lines += [
        "",
        "两分支直接配对差：",
        "",
        "|比较|ΔAUC|95%区间|",
        "|---|---:|---|",
    ]
    for row in direct_differences.itertuples(index=False):
        lines.append(
            f"|{row.comparison}|{row.delta_AUC:+.4f}|{row.CI_low:+.4f}–{row.CI_high:+.4f}|"
        )
    lines += [
        "",
        "## 判定",
        "",
        "Situation 1：2秒MSE妊娠更高、Huang-like妊娠更低，测量定义可解释方向冲突。",
        "Situation 2：两分支妊娠均更高，2秒时间尺度不足以解释与原文献的反向。",
        "本报告只按预设规则判定，不据AUC调整Phase A。",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    run(parser.parse_args())
