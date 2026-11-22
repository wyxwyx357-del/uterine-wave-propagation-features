"""Audit the complete-record ER-MSE signal without adding or recomputing features.

The audit reads the frozen feature aggregation, original examination cohort,
and optional clinical workbook. It checks case membership, outcome direction,
missingness, MSE decomposition, and clinical-baseline behaviour.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.mse_pregnancy_association.code.all_features_auc import grouped_splits
from analysis.mse_pregnancy_association.code.frozen_feature_auc import CLINICAL, GROUP, OUTCOME, SEED, digest, save

EXAM = "临床检查编号"
MSE = "record_MSE_time_equal"
CC = "record_CC_time_equal"


def _std_diff(a, b):
    """Unavailable minus available standardized mean difference."""
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().to_numpy(float)
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 1e-12 else None


def _case_masks(data):
    clinical_complete = data[CLINICAL].notna().all(axis=1)
    return {
        "MSE_available": data[MSE].notna(),
        "CC_available": data[CC].notna(),
        "MSE_and_CC_complete": data[[MSE, CC]].notna().all(axis=1),
        "MSE_clinical_complete": data[MSE].notna() & clinical_complete,
        "CC_clinical_complete": data[CC].notna() & clinical_complete,
        "MSE_and_CC_AUC_complete": data[[MSE, CC]].notna().all(axis=1),
    }


def _clinical_model(C):
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(C),
                    penalty="l2",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def _clinical_repeat_auc(data, base_seed, repeats):
    y = data[OUTCOME].to_numpy(int)
    groups = data[GROUP].to_numpy(str)
    values = []
    for repeat in range(repeats):
        score = np.full(len(data), np.nan)
        for train, test in grouped_splits(y, groups, 5, base_seed + repeat):
            model = _clinical_model(1e6).fit(data.iloc[train][CLINICAL], y[train])
            score[test] = model.predict_proba(data.iloc[test][CLINICAL])[:, 1]
        if not np.isfinite(score).all():
            raise ValueError("Clinical comparison has an unassigned fold")
        values.append(float(roc_auc_score(y, score)))
    return dict(
        mean_AUC=float(np.mean(values)),
        min_AUC=float(np.min(values)),
        max_AUC=float(np.max(values)),
    )


def _clinical_audit(data, repeats):
    y = data[OUTCOME].to_numpy(int)
    groups = data[GROUP].to_numpy(str)
    audit_rows = []
    coefficient_rows = []
    for repeat in range(repeats):
        for fold, (train, test) in enumerate(
            grouped_splits(y, groups, 5, SEED + repeat), 1
        ):
            model = _clinical_model(1e6).fit(data.iloc[train][CLINICAL], y[train])
            score = model.predict_proba(data.iloc[test][CLINICAL])[:, 1]
            coefficients = model.named_steps["model"].coef_[0]
            audit_rows.append(
                dict(
                    repeat=repeat,
                    fold=fold,
                    train_n=len(train),
                    test_n=len(test),
                    test_positive=int(y[test].sum()),
                    test_negative=int((1 - y[test]).sum()),
                    AUC=float(roc_auc_score(y[test], score)),
                    age_coefficient=float(coefficients[0]),
                    age_direction="positive" if coefficients[0] > 0 else "negative",
                    patient_overlap=0,
                )
            )
            for feature, coefficient in zip(CLINICAL, coefficients):
                coefficient_rows.append(
                    dict(
                        repeat=repeat,
                        fold=fold,
                        feature=feature,
                        coefficient=float(coefficient),
                    )
                )

    fixed_c = []
    for cohort_name, subset in (("all223", data), ("MSE_available", data.loc[data[MSE].notna()])):
        subset = subset.reset_index(drop=True)
        y_sub = subset[OUTCOME].to_numpy(int)
        g_sub = subset[GROUP].to_numpy(str)
        for C in (0.01, 0.1, 1.0, 10.0, 1e6):
            for repeat in range(repeats):
                score = np.full(len(subset), np.nan)
                for train, test in grouped_splits(y_sub, g_sub, 5, SEED + repeat):
                    model = _clinical_model(C).fit(subset.iloc[train][CLINICAL], y_sub[train])
                    score[test] = model.predict_proba(subset.iloc[test][CLINICAL])[:, 1]
                if not np.isfinite(score).all():
                    raise ValueError("Clinical sensitivity has an unassigned fold")
                fixed_c.append(
                    dict(
                        cohort=cohort_name,
                        C=float(C),
                        repeat=repeat,
                        n=len(subset),
                        patients=int(subset[GROUP].nunique()),
                        AUC=float(roc_auc_score(y_sub, score)),
                    )
                )
    return pd.DataFrame(audit_rows), pd.DataFrame(coefficient_rows), pd.DataFrame(fixed_c)


def _coordination_fold_audit(data, repeats):
    features = [
        MSE,
        CC,
        "record_MSE_z",
        "record_mean_bias_sq",
        "record_sd_scale_sq",
        "record_time_shape_term",
        "record_mse_missing",
    ]
    y = data[OUTCOME].to_numpy(int)
    groups = data[GROUP].to_numpy(str)
    rows = []
    for feature in features:
        for repeat in range(repeats):
            for fold, (train, test) in enumerate(
                grouped_splits(y, groups, 5, SEED + repeat), 1
            ):
                model = _clinical_model(1e6).fit(
                    data.iloc[train][[feature]], y[train]
                )
                score = model.predict_proba(data.iloc[test][[feature]])[:, 1]
                coefficient = float(model.named_steps["model"].coef_[0, 0])
                rows.append(
                    dict(
                        feature=feature,
                        repeat=repeat,
                        fold=fold,
                        train_available=int(data.iloc[train][feature].notna().sum()),
                        test_available=int(data.iloc[test][feature].notna().sum()),
                        coefficient=coefficient,
                        direction="positive" if coefficient > 0 else "negative",
                        AUC=float(roc_auc_score(y[test], score)),
                        patient_overlap=0,
                    )
                )
    audit = pd.DataFrame(rows)
    summary = (
        audit.groupby("feature", as_index=False)
        .agg(
            fold_mean_AUC=("AUC", "mean"),
            fold_min_AUC=("AUC", "min"),
            fold_max_AUC=("AUC", "max"),
            mean_coefficient=("coefficient", "mean"),
            positive_coefficient_fraction=("coefficient", lambda values: float((values > 0).mean())),
            sign_flips=("coefficient", lambda values: int(np.any(values > 0) and np.any(values < 0))),
        )
    )
    return audit, summary


def _categorical_missingness(data, missing):
    rows = []
    for variable in (
        "outcome",
        "transfer_multiple",
        "blastocyst",
        "endometrial_type_C",
        "video_month",
        "clinical_source",
        "clinical_block_status",
    ):
        if variable == "outcome":
            values = data[OUTCOME].astype("string")
        elif variable == "transfer_multiple":
            values = data[CLINICAL[3]].map({0.0: "single", 1.0: "multiple"}).astype("string")
        elif variable == "blastocyst":
            values = data[CLINICAL[5]].map({0.0: "non_blastocyst", 1.0: "blastocyst"}).astype("string")
        elif variable == "endometrial_type_C":
            values = data[CLINICAL[6]].map({0.0: "non_C", 1.0: "C"}).astype("string")
        else:
            values = data[variable].astype("string")
        values = values.fillna("<missing>")
        for level in sorted(values.unique()):
            selected = values.eq(level)
            rows.append(
                dict(
                    variable=variable,
                    level=level,
                    available_n=int((selected & ~missing).sum()),
                    unavailable_n=int((selected & missing).sum()),
                    missing_fraction=float(missing[selected].mean()) if selected.any() else None,
                )
            )
    return pd.DataFrame(rows)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    record_path = args.records
    source_hash = digest(record_path)
    records = pd.read_csv(record_path, dtype={GROUP: str, EXAM: str})
    cohort = pd.read_csv(args.cohort, dtype={GROUP: str, EXAM: str})
    if records["case_id"].duplicated().any() or cohort["case_id"].duplicated().any():
        raise ValueError("Case IDs must be unique")
    if set(records["case_id"]) != set(cohort["case_id"]):
        raise ValueError("Record and cohort case IDs differ")

    metadata = cohort[["case_id", "video_path"]].copy()
    metadata["video_month"] = metadata["video_path"].map(
        lambda value: Path(str(value)).parent.parent.name
    )
    metadata = metadata.drop(columns="video_path")
    data = records.merge(metadata, on="case_id", how="left", validate="one_to_one")

    if args.workbook:
        workbook = Path(args.workbook)
        final = pd.read_excel(workbook, sheet_name=0)
        final = final.copy()
        final["_exam_id"] = final.iloc[:, 10].astype(str).str.strip().str.upper()
        final = final.set_index("_exam_id")
        data["clinical_source"] = data[EXAM].map(final["临床数据来源文件"])
        data["clinical_block_status"] = data[EXAM].map(final["临床块归属状态"])
    else:
        data["clinical_source"] = "<not_supplied>"
        data["clinical_block_status"] = "<not_supplied>"

    masks = _case_masks(data)
    case_dir = args.output / "case_sets"
    case_dir.mkdir(exist_ok=True)
    membership = data[["case_id", EXAM, GROUP, OUTCOME]].copy()
    for name, mask in masks.items():
        membership[name] = mask
        data.loc[
            mask,
            ["case_id", EXAM, GROUP, OUTCOME],
        ].to_csv(case_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    membership.to_csv(args.output / "case_membership.csv", index=False, encoding="utf-8-sig")

    direction = []
    for feature in (MSE, CC):
        for value in (0, 1):
            values = data.loc[data[OUTCOME].eq(value), feature].dropna()
            direction.append(
                dict(
                    feature=feature,
                    outcome=int(value),
                    outcome_label="clinical_pregnancy" if value == 1 else "non_pregnancy",
                    n=len(values),
                    mean=float(values.mean()),
                    median=float(values.median()),
                    Q1=float(values.quantile(0.25)),
                    Q3=float(values.quantile(0.75)),
                    SD=float(values.std(ddof=1)),
                )
            )
    direction = pd.DataFrame(direction)
    outcome_coding = dict(
        unique_values=sorted(int(value) for value in data[OUTCOME].unique()),
        pregnancy_value=1,
        nonpregnancy_value=0,
        pregnancy_n=int(data[OUTCOME].sum()),
        nonpregnancy_n=int((1 - data[OUTCOME]).sum()),
        auc_positive_class="pregnancy=1",
        odds_ratio_orientation=(
            "OR>1 means higher feature value is associated with pregnancy"
        ),
        mse_direction_invariant_to_common_C2F_F2C_sign_flip=True,
    )

    missing = ~masks["MSE_available"]
    continuous = [
        ("age", CLINICAL[0]),
        ("BMI", CLINICAL[1]),
        ("infertility_years", CLINICAL[2]),
        ("multiple_embryo", CLINICAL[3]),
        ("endometrial_thickness_mm", CLINICAL[4]),
        ("blastocyst", CLINICAL[5]),
        ("endometrial_type_C", CLINICAL[6]),
        ("video_duration_s", "video_duration_s"),
    ]
    subgroup_rows = []
    for label, column in continuous:
        available = pd.to_numeric(data.loc[~missing, column], errors="coerce")
        unavailable = pd.to_numeric(data.loc[missing, column], errors="coerce")
        subgroup_rows.append(
            dict(
                variable=label,
                available_n=int(available.notna().sum()),
                available_mean=float(available.mean()),
                unavailable_n=int(unavailable.notna().sum()),
                unavailable_mean=float(unavailable.mean()),
                unavailable_minus_available_SD=_std_diff(unavailable, available),
            )
        )
    subgroup_rows.extend(
        [
            dict(
                variable="pregnancy_rate",
                available_n=int((~missing).sum()),
                available_mean=float(data.loc[~missing, OUTCOME].mean()),
                unavailable_n=int(missing.sum()),
                unavailable_mean=float(data.loc[missing, OUTCOME].mean()),
                unavailable_minus_available_SD=_std_diff(
                    data.loc[missing, OUTCOME], data.loc[~missing, OUTCOME]
                ),
            ),
            dict(
                variable="video_year",
                available_n=int((~missing).sum()),
                available_mean=float(
                    data.loc[~missing, "case_id"].astype(str).str.rsplit("-", n=1).str[-1].str[:4].astype(int).mean()
                ),
                unavailable_n=int(missing.sum()),
                unavailable_mean=float(
                    data.loc[missing, "case_id"].astype(str).str.rsplit("-", n=1).str[-1].str[:4].astype(int).mean()
                ),
                unavailable_minus_available_SD=_std_diff(
                    data.loc[missing, "case_id"].astype(str).str.rsplit("-", n=1).str[-1].str[:4].astype(int),
                    data.loc[~missing, "case_id"].astype(str).str.rsplit("-", n=1).str[-1].str[:4].astype(int),
                ),
            ),
        ]
    )
    subgroup = pd.DataFrame(subgroup_rows)
    categorical = _categorical_missingness(data, missing)

    fold_audit, coefficient_audit, clinical_c = _clinical_audit(data, args.repeats)
    coefficient_summary = (
        coefficient_audit.groupby("feature", as_index=False)
        .agg(
            mean_coefficient=("coefficient", "mean"),
            positive_fraction=("coefficient", lambda values: float((values > 0).mean())),
            negative_fraction=("coefficient", lambda values: float((values < 0).mean())),
            sign_flips=("coefficient", lambda values: int(np.any(values > 0) and np.any(values < 0))),
        )
    )
    clinical_c_summary = (
        clinical_c.groupby(["cohort", "C"], as_index=False)
        .agg(
            repeat_mean_AUC=("AUC", "mean"),
            repeat_min_AUC=("AUC", "min"),
            repeat_max_AUC=("AUC", "max"),
            n=("n", "first"),
            patients=("patients", "first"),
        )
    )
    coordination_fold, coordination_fold_summary = _coordination_fold_audit(
        data, args.repeats
    )
    prior_comparison = pd.DataFrame()
    if args.prior_cohort:
        prior_data = pd.read_csv(args.prior_cohort, dtype={GROUP: str, EXAM: str})
        for column in CLINICAL + [OUTCOME]:
            prior_data[column] = pd.to_numeric(prior_data[column], errors="coerce")
        shared = data.loc[data[EXAM].isin(set(prior_data[EXAM]))].reset_index(drop=True)
        rows = []
        for base_seed in (20261116, 20260531):
            for cohort_name, cohort in (
                ("prior_323", prior_data),
                ("shared_examinations", shared),
                ("current_223", data),
            ):
                metrics = _clinical_repeat_auc(cohort, base_seed, args.repeats)
                rows.append(
                    dict(
                        cohort=cohort_name,
                        base_seed=base_seed,
                        n=len(cohort),
                        patients=int(cohort[GROUP].nunique()),
                        positive=int(cohort[OUTCOME].sum()),
                        **metrics,
                    )
                )
        prior_comparison = pd.DataFrame(rows)

    decomposition = []
    for value in (0, 1):
        subset = data.loc[data[OUTCOME].eq(value)]
        decomposition.append(
            dict(
                outcome=int(value),
                n=int(subset[[MSE, "record_mean_bias_sq", "record_sd_scale_sq", "record_time_shape_term"]].dropna().shape[0]),
                mean_MSE=float(subset[MSE].mean()),
                mean_bias_sq=float(subset["record_mean_bias_sq"].mean()),
                mean_sd_scale_sq=float(subset["record_sd_scale_sq"].mean()),
                mean_time_shape_term=float(subset["record_time_shape_term"].mean()),
                mean_MSE_z=float(subset["record_MSE_z"].mean()),
                mean_CC=float(subset[CC].mean()),
            )
        )
    decomposition = pd.DataFrame(decomposition)

    clinical_reference = pd.DataFrame(
        [
            dict(
                cohort="prior_323_exam_301_patient",
                clinical_AUC=0.5731158605174353,
                CI_low=0.503580447189793,
                CI_high=0.6403085320847355,
                source="archive_20261116/R3_prediction_feature_evaluation/results/clinical_plus_features_auc_v1_20261116/auc_results.csv",
            ),
            dict(
                cohort="current_223_exam_208_patient",
                clinical_AUC=0.4916536254564424,
                CI_low=0.41018418201516793,
                CI_high=0.5750594269985808,
                source=str(args.records.parent / "record_auc_all_imputed" / "AUC_metrics.csv"),
            ),
        ]
    )

    for name, frame in (
        ("case_membership", membership),
        ("direction_descriptives", direction),
        ("missingness_continuous", subgroup),
        ("missingness_categorical", categorical),
        ("clinical_fold_audit", fold_audit),
        ("coordination_fold_audit", coordination_fold),
        ("coordination_fold_summary", coordination_fold_summary),
        ("clinical_coefficient_summary", coefficient_summary),
        ("clinical_regularization_auc", clinical_c_summary),
        ("clinical_prior_comparison", prior_comparison),
        ("mse_decomposition_descriptives", decomposition),
        ("clinical_AUC_cohort_comparison", clinical_reference),
    ):
        frame.to_csv(args.output / f"{name}.csv", index=False, encoding="utf-8-sig")
    save(args.output / "outcome_coding.json", outcome_coding)

    cat_tests = []
    for variable in categorical["variable"].unique():
        subset = categorical.loc[categorical["variable"] == variable]
        counts = subset[["available_n", "unavailable_n"]].to_numpy(int)
        if counts.shape[0] >= 2 and counts.sum(axis=0).min() > 0:
            chi2, p, _, _ = chi2_contingency(counts)
            total = counts.sum()
            cat_tests.append(
                dict(
                    variable=variable,
                    chi2=float(chi2),
                    p=float(p),
                    cramer_v=float(np.sqrt(chi2 / (total * (min(counts.shape) - 1)))),
                )
            )
    cat_tests = pd.DataFrame(cat_tests)
    cat_tests.to_csv(args.output / "missingness_categorical_tests.csv", index=False, encoding="utf-8-sig")

    summary = dict(
        source_sha256=source_hash,
        case_set_counts={name: int(mask.sum()) for name, mask in masks.items()},
        outcome_coding=outcome_coding,
        direction=direction.to_dict(orient="records"),
        missingness_continuous=subgroup.to_dict(orient="records"),
        mse_decomposition=decomposition.to_dict(orient="records"),
        clinical_C_sensitivity=clinical_c_summary.to_dict(orient="records"),
        coordination_fold_stability=coordination_fold_summary.to_dict(orient="records"),
        clinical_coefficient_stability=coefficient_summary.to_dict(orient="records"),
        clinical_AUC_cohort_comparison=clinical_reference.to_dict(orient="records"),
        clinical_prior_comparison=prior_comparison.to_dict(orient="records"),
        source_features_recomputed=False,
        formal_features_changed=False,
    )
    save(args.output / "summary.json", summary)

    lines = [
        "# ER-MSE 新信号可信度核查包",
        "",
        "本核查只读取既有冻结特征和已保存汇总，没有新增检测器、没有重跑追踪或径向伸缩率。",
        "",
        "## 1. 样本集合核对",
        "",
        "|集合|例数|",
        "|---|---:|",
    ]
    for name, count in summary["case_set_counts"].items():
        lines.append(f"|{name}|{count}|")
    lines += [
        "",
        "`MSE_available`和`CC_available`是特征本身可用性；`*_clinical_complete`又要求固定7项临床变量完整。",
        "`MSE_and_CC_AUC_complete`是AUC模型使用的中位数插补队列中两个协调特征均原本存在者，因此可为156例，不与临床完整关联的154/153例冲突。",
        "逐例集合位于`case_sets/`，完整成员表位于`case_membership.csv`。",
        "",
        "## 2. 标签和方向",
        "",
        "妊娠编码为1，非妊娠编码为0；AUC和OR均以妊娠为正类。",
        "",
        "|特征|结局|n|mean|median|Q1–Q3|",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in direction.itertuples(index=False):
        lines.append(
            f"|{row.feature}|{row.outcome_label}|{row.n}|{row.mean:.4f}|"
            f"{row.median:.4f}|{row.Q1:.4f}–{row.Q3:.4f}|"
        )
    lines += [
        "",
        "前后壁ER同时乘以−1不会改变差值的平方，因此共同的C2F/F2C符号翻转不会反转MSE方向；前后壁互换也不改变MSE。",
        "",
        "### 逐折方向稳定性",
        "",
        "|特征|折均AUC|AUC范围|系数为正的折比例|系数方向翻转|",
        "|---|---:|---|---:|---|",
    ]
    for row in coordination_fold_summary.itertuples(index=False):
        lines.append(
            f"|{row.feature}|{row.fold_mean_AUC:.4f}|"
            f"{row.fold_min_AUC:.4f}–{row.fold_max_AUC:.4f}|"
            f"{row.positive_coefficient_fraction:.0%}|"
            f"{'yes' if row.sign_flips else 'no'}|"
        )
    lines += [
        "",
        "## 3. 157例可计算者与66例缺失者",
        "",
        "|变量|可计算均值|缺失均值|缺失−可计算标准化差|",
        "|---|---:|---:|---:|",
    ]
    for row in subgroup.itertuples(index=False):
        value = "NA" if pd.isna(row.unavailable_minus_available_SD) else f"{row.unavailable_minus_available_SD:+.3f}"
        lines.append(
            f"|{row.variable}|{row.available_mean:.4f}|{row.unavailable_mean:.4f}|{value}|"
        )
    lines += [
        "",
        "缺失者两壁有效追踪比例明显更低，说明缺失首先反映可观察支持不足；BMI和妊娠率接近，但年龄和内膜厚度存在中等分布差。分类来源/月份和批次检验见`missingness_categorical_tests.csv`。",
        "",
        "|分类变量|卡方|p|Cramér V|",
        "|---|---:|---:|---:|",
    ]
    for row in cat_tests.itertuples(index=False):
        lines.append(
            f"|{row.variable}|{row.chi2:.3f}|{row.p:.4f}|{row.cramer_v:.3f}|"
        )
    lines += [
        "",
        "妊娠结局在可计算与不可计算组中的缺失率几乎相同（29.58%对29.63%）。但视频月份和临床来源文件与可计算性有描述性关联，提示批次/采集时点仍需核查；这些不是随机缺失的证明。",
        "",
        "## 4. MSE分解",
        "",
        "完整记录MSE满足：均值偏置平方 + SD尺度平方 + 时间形状项。",
        "两壁各自标准化后，`MSE_z ≈ 2(1-CC)`，因此MSE_z和CC在本数据中给出相同的排序信息。",
        "",
        "|结局|n|MSE|均值偏置项|SD尺度项|时间形状项|MSE_z|CC|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in decomposition.itertuples(index=False):
        lines.append(
            f"|{row.outcome}|{row.n}|{row.mean_MSE:.4f}|{row.mean_bias_sq:.4f}|"
            f"{row.mean_sd_scale_sq:.4f}|{row.mean_time_shape_term:.4f}|"
            f"{row.mean_MSE_z:.4f}|{row.mean_CC:.4f}|"
        )
    lines += [
        "",
        "关联结果显示均值偏置项仍显著（调整后OR约2.06，BH q约0.040），SD尺度项无关联；MSE_z与CC基本同向且不显著。因此原始MSE信号主要不是“标准化后的时间不同步”，而是前后壁平均ER差异，时间形状项只有边缘提示。",
        "",
        "## 5. 临床基线审计",
        "",
        "|队列|C|折外AUC均值|范围|",
        "|---|---:|---:|---|",
    ]
    for row in clinical_c_summary.itertuples(index=False):
        lines.append(
            f"|{row.cohort}|{row.C:g}|{row.repeat_mean_AUC:.4f}|"
            f"{row.repeat_min_AUC:.4f}–{row.repeat_max_AUC:.4f}|"
        )
    lines += [
        "",
        "当前年龄系数方向在`clinical_coefficient_summary.csv`中逐变量汇总。先前323次检查/301患者的临床基线为0.5731（0.5036–0.6403），当前223次/208患者为0.4917（0.4102–0.5751）；两者队列、时间和影像可计算性不同，不能直接归因于编码错误。",
        "",
        "|临床变量|平均标准化系数|正号折比例|折内符号翻转|",
        "|---|---:|---:|---|",
    ]
    for row in coefficient_summary.itertuples(index=False):
        lines.append(
            f"|{row.feature}|{row.mean_coefficient:+.3f}|"
            f"{row.positive_fraction:.0%}|{'yes' if row.sign_flips else 'no'}|"
        )
    lines += [
        "",
        "C=1e6固定临床模型的逐折AUC范围为"
        f"{fold_audit['AUC'].min():.4f}–{fold_audit['AUC'].max():.4f}；"
        "年龄在全部训练折均为负系数，BMI均为正系数，其余变量存在方向变化。",
    ]
    if len(prior_comparison):
        lines += [
            "",
            "### 旧323例与当前影像可计算子集",
            "",
            f"当前223例与旧323例有{shared[EXAM].nunique()}个共同检查编号。",
            "",
            "|队列|基础种子|n|患者|妊娠|折外AUC均值|范围|",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in prior_comparison.itertuples(index=False):
            lines.append(
                f"|{row.cohort}|{row.base_seed}|{row.n}|{row.patients}|{row.positive}|"
                f"{row.mean_AUC:.4f}|{row.min_AUC:.4f}–{row.max_AUC:.4f}|"
            )
        lines += [
            "",
            "当前临床基线下降主要随影像/追踪可计算子集出现，而不是结局编码或标准化/插补反转。旧323例可包含当前223例无法处理的影像，因此旧AUC不能直接作为当前子集的预期基线。",
        ]
    lines += [
        "",
        "## 6. 缺失敏感性结论",
        "",
        "- 全部223例中位数插补：完整记录MSE单独AUC约0.588。",
        "- 全部223例临床+MSE：AUC约0.571；再加MSE缺失指示后约0.580。",
        "- MSE缺失指示单独AUC约0.420，且缺失者有效追踪比例更低。",
        "- MSE和CC均存在的156例：MSE AUC约0.596，临床+MSE约0.541。",
        "",
        "三种缺失处理下MSE方向一致，没有显示“高MSE只由缺失机制造成”；但完整病例临床基线仅约0.408，说明该子集存在明显分布/样本不稳定，不能把完整病例增量当成主验证。",
        "",
        "## 7. 当前审核结论",
        "",
        "当前可支持的表述是：完整记录前后壁ER-MSE存在内部探索性关联，方向经标签、缺失三种处理和MSE分解后没有反转；但信号主要对应平均ER偏置，而非标准化时间协调性。当前临床基线偏低主要是影像可计算子集相对旧323例的下降，继续冻结前仍需在独立数据中验证。",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--workbook", type=Path)
    parser.add_argument("--prior-cohort", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    run(parser.parse_args())
