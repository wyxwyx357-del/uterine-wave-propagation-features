"""Post-hoc source audit of between-wall mean ER offset.

This does not define a new production feature or search a new model. It
describes the already-observed mean-ER term and checks its stability across
source, calendar month, and technical tracking covariates.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis.mse_pregnancy_association.code.feature_association import fit_cluster_logit
from analysis.mse_pregnancy_association.code.frozen_feature_auc import CLINICAL, GROUP, OUTCOME, digest, save

EXAM = "临床检查编号"
MSE = "record_MSE_time_equal"
MEAN_ANT = "record_ER_mean_anterior"
MEAN_POST = "record_ER_mean_posterior"
BIAS = "record_mean_bias_sq"


def add_offset_columns(data):
    data = data.copy()
    data["delta_ER"] = data[MEAN_ANT] - data[MEAN_POST]
    data["abs_delta_ER"] = data["delta_ER"].abs()
    data["mean_ER"] = (data[MEAN_ANT] + data[MEAN_POST]) / 2
    return data


def _technical_covariates(frame):
    columns = [
        "video_duration_s",
        "anterior_valid_fraction",
        "posterior_valid_fraction",
    ]
    values = frame[columns].to_numpy(float)
    return (values - values.mean(0)) / values.std(0)


def adjusted_association(data, feature, include_source):
    use = data[
        [feature, OUTCOME, GROUP, "video_duration_s",
         "anterior_valid_fraction", "posterior_valid_fraction", "source_file"]
    ].dropna().reset_index(drop=True)
    scale = float(use[feature].std(ddof=1))
    z = (use[feature].to_numpy(float) - use[feature].mean()) / scale
    technical = _technical_covariates(use)
    columns = [np.ones(len(use)), z, *technical.T]
    if include_source:
        source_levels = sorted(use["source_file"].unique())
        if len(source_levels) < 2:
            return None
        columns.append(use["source_file"].eq(source_levels[0]).to_numpy(float))
    design = np.column_stack(columns)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError(f"Technical design is rank deficient for {feature}")
    fit = fit_cluster_logit(
        design,
        use[OUTCOME].to_numpy(int),
        use[GROUP].to_numpy(str),
    )
    beta = float(fit["beta"][1])
    se = float(np.sqrt(fit["covariance"][1, 1]))
    return dict(
        feature=feature,
        n=len(use),
        patients=int(use[GROUP].nunique()),
        feature_SD=scale,
        OR_per_SD=float(np.exp(beta)),
        CI_low=float(np.exp(beta - 1.959963984540054 * se)),
        CI_high=float(np.exp(beta + 1.959963984540054 * se)),
        p=float(2 * norm.sf(abs(beta / se))),
        source_adjusted=bool(include_source),
        score_max=float(fit["score_max"]),
        optimizer_success=bool(fit["optimizer_success"]),
        optimizer_message=fit["optimizer_message"],
    )


def outcome_descriptives(data, features):
    rows = []
    for feature in features:
        for outcome in (0, 1):
            values = data.loc[data[OUTCOME].eq(outcome), feature].dropna()
            rows.append(
                dict(
                    feature=feature,
                    outcome=outcome,
                    n=len(values),
                    mean=float(values.mean()),
                    median=float(values.median()),
                    Q1=float(values.quantile(0.25)),
                    Q3=float(values.quantile(0.75)),
                    SD=float(values.std(ddof=1)),
                )
            )
    return pd.DataFrame(rows)


def stratified_effects(data, variable):
    rows = []
    available = data.loc[data[[MEAN_ANT, MEAN_POST]].notna().all(axis=1)]
    for level in sorted(available[variable].dropna().astype(str).unique()):
        subset = available.loc[available[variable].astype(str) == level]
        row = dict(
            stratum=level,
            available_n=len(subset),
            patients=int(subset[GROUP].nunique()),
            positive=int(subset[OUTCOME].sum()),
            negative=int((1 - subset[OUTCOME]).sum()),
        )
        for feature in ("delta_ER", "abs_delta_ER", BIAS, MSE):
            for outcome in (0, 1):
                row[f"{feature}_mean_outcome{outcome}"] = float(
                    subset.loc[subset[OUTCOME].eq(outcome), feature].mean()
                )
            row[f"{feature}_pregnancy_minus_nonpregnancy"] = float(
                row[f"{feature}_mean_outcome1"] - row[f"{feature}_mean_outcome0"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def technical_correlations(data):
    rows = []
    technical = [
        "video_duration_s",
        "anterior_valid_fraction",
        "posterior_valid_fraction",
    ]
    for feature in ("delta_ER", "abs_delta_ER", BIAS, MSE):
        use = data[[feature, *technical, "video_month"]].dropna().copy()
        use["month_index"] = (
            pd.to_datetime(use["video_month"] + "-01") - pd.Timestamp("2024-01-01")
        ).dt.days / 30.4375
        for covariate in technical + ["month_index"]:
            rows.append(
                dict(
                    feature=feature,
                    covariate=covariate,
                    n=len(use),
                    Pearson_r=float(use[feature].corr(use[covariate])),
                )
            )
    return pd.DataFrame(rows)


def leave_one_out(data, variable, features):
    rows = []
    for level in sorted(data[variable].dropna().astype(str).unique()):
        subset = data.loc[data[variable].astype(str) != level]
        for feature in features:
            result = adjusted_association(subset, feature, include_source=False)
            if result is not None:
                rows.append(dict(excluded=level, **result))
    return pd.DataFrame(rows)


def period_effects(data, features):
    rows = []
    for year in sorted(data["video_year"].unique()):
        subset = data.loc[data["video_year"].eq(year)]
        for feature in features:
            result = adjusted_association(subset, feature, include_source=False)
            if result is not None:
                rows.append(dict(period=str(int(year)), **result))
    return pd.DataFrame(rows)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash = digest(args.records)
    data = pd.read_csv(args.records, dtype={GROUP: str, EXAM: str})
    data = add_offset_columns(data)

    cohort = pd.read_csv(args.cohort, dtype={EXAM: str, GROUP: str})
    metadata = cohort[["case_id", "video_path", "anterior_valid_fraction",
                       "posterior_valid_fraction"]].copy()
    metadata["video_month"] = metadata["video_path"].map(
        lambda value: Path(str(value)).parent.parent.name
    )
    metadata = metadata.drop(columns="video_path")
    data = data.merge(metadata, on="case_id", how="left", validate="one_to_one",
                      suffixes=("", "_cohort"))

    workbook = pd.read_excel(args.workbook, sheet_name=0)
    workbook["_exam_id"] = workbook.iloc[:, 10].astype(str).str.strip().str.upper()
    workbook = workbook.set_index("_exam_id")
    data["source_file"] = data[EXAM].map(workbook["临床数据来源文件"])
    data["source_block_status"] = data[EXAM].map(workbook["临床块归属状态"])
    data["video_year"] = (
        data["video_month"].str.slice(0, 4).astype(int)
    )

    features = [
        MEAN_ANT,
        MEAN_POST,
        "delta_ER",
        "abs_delta_ER",
        BIAS,
        MSE,
    ]
    descriptive = outcome_descriptives(data, features)

    associations = []
    for feature in features:
        for source_adjusted in (False, True):
            result = adjusted_association(data, feature, source_adjusted)
            if result is not None:
                associations.append(result)
    associations = pd.DataFrame(associations)

    by_source = stratified_effects(data, "source_file")
    by_month = stratified_effects(data, "video_month")
    by_year = stratified_effects(data, "video_year")
    correlations = technical_correlations(data)
    correlations["abs_Pearson_r"] = correlations["Pearson_r"].abs()
    correlation_summary = (
        correlations.sort_values("abs_Pearson_r", ascending=False)
        .groupby("feature", as_index=False)
        .first()[["feature", "covariate", "Pearson_r", "abs_Pearson_r", "n"]]
    )
    source_year = (
        data.loc[data[MEAN_ANT].notna()]
        .groupby(["video_year", "source_file"], as_index=False)
        .agg(
            available_n=("case_id", "size"),
            patients=(GROUP, "nunique"),
            positive=(OUTCOME, "sum"),
            pregnancy_rate=(OUTCOME, "mean"),
            mean_delta_ER=("delta_ER", "mean"),
            mean_abs_delta_ER=("abs_delta_ER", "mean"),
            mean_bias_sq=(BIAS, "mean"),
            mean_MSE=(MSE, "mean"),
        )
    )
    leave_source = leave_one_out(
        data.loc[data["source_file"].notna()],
        "source_file",
        ["delta_ER", "abs_delta_ER", BIAS, MSE],
    )
    leave_month = leave_one_out(
        data.loc[data["video_month"].notna()],
        "video_month",
        ["delta_ER", "abs_delta_ER", BIAS, MSE],
    )
    by_period = period_effects(
        data, ["delta_ER", "abs_delta_ER", BIAS, MSE]
    )

    case_rows = data[
        [
            "case_id", EXAM, GROUP, OUTCOME, "video_month", "video_year",
            "source_file", MEAN_ANT, MEAN_POST, "delta_ER", "abs_delta_ER",
            BIAS, MSE, "video_duration_s", "anterior_valid_fraction",
            "posterior_valid_fraction",
        ]
    ]
    for name, frame in (
        ("case_mean_offset", case_rows),
        ("outcome_descriptives", descriptive),
        ("technical_adjusted_associations", associations),
        ("stratified_by_source", by_source),
        ("stratified_by_month", by_month),
        ("stratified_by_year", by_year),
        ("technical_correlations", correlations),
        ("technical_correlation_summary", correlation_summary),
        ("source_year_distribution", source_year),
        ("leave_one_source_out", leave_source),
        ("leave_one_month_out", leave_month),
        ("period_effects", by_period),
    ):
        frame.to_csv(args.output / f"{name}.csv", index=False, encoding="utf-8-sig")

    month_complete = by_month.dropna(
        subset=[
            "delta_ER_pregnancy_minus_nonpregnancy",
            "abs_delta_ER_pregnancy_minus_nonpregnancy",
            BIAS + "_pregnancy_minus_nonpregnancy",
            MSE + "_pregnancy_minus_nonpregnancy",
        ]
    )
    direction_summary = []
    for feature in ("delta_ER", "abs_delta_ER", BIAS, MSE):
        values = month_complete[f"{feature}_pregnancy_minus_nonpregnancy"]
        direction_summary.append(
            dict(
                feature=feature,
                strata_with_both_outcomes=len(values),
                positive_strata=int((values > 0).sum()),
                negative_strata=int((values < 0).sum()),
                exact_zero_strata=int((values == 0).sum()),
            )
        )
    direction_summary = pd.DataFrame(direction_summary)
    direction_summary.to_csv(args.output / "month_direction_summary.csv",
                             index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(args.output / "er_mean_offset_batch_audit.xlsx", engine="openpyxl") as writer:
        case_rows.to_excel(writer, sheet_name="case_offset", index=False)
        descriptive.to_excel(writer, sheet_name="outcomes", index=False)
        associations.to_excel(writer, sheet_name="adjusted_OR", index=False)
        by_source.to_excel(writer, sheet_name="source", index=False)
        by_month.to_excel(writer, sheet_name="month", index=False)
        leave_source.to_excel(writer, sheet_name="leave_source", index=False)
        leave_month.to_excel(writer, sheet_name="leave_month", index=False)
        by_period.to_excel(writer, sheet_name="period", index=False)

    if digest(args.records) != source_hash:
        raise RuntimeError("Source coordination table changed during audit")
    summary = dict(
        source_sha256=source_hash,
        available_exams=int(data[[MEAN_ANT, MEAN_POST]].notna().all(axis=1).sum()),
        outcome_descriptives=descriptive.to_dict(orient="records"),
        technical_associations=associations.to_dict(orient="records"),
        technical_correlation_summary=correlation_summary.to_dict(orient="records"),
        month_direction_summary=direction_summary.to_dict(orient="records"),
        leave_one_source_out=leave_source.to_dict(orient="records"),
        period_effects=by_period.to_dict(orient="records"),
        formal_features_changed=False,
        new_model_search=False,
    )
    save(args.output / "summary.json", summary)

    lines = [
        "# 前后壁平均ER偏置来源与批次稳定性审计",
        "",
        "这是对已发现MSE信号的post-hoc解释分析，不新增正式特征，不搜索新模型。",
        "",
        f"共同有效ER记录{summary['available_exams']}例。",
        "",
        "## 1. 哪一壁发生偏移",
        "",
        "|特征|非妊娠均值|妊娠均值|妊娠−非妊娠|",
        "|---|---:|---:|---:|",
    ]
    desc = descriptive.set_index(["feature", "outcome"])
    for feature in features:
        nonpreg = desc.loc[(feature, 0), "mean"]
        preg = desc.loc[(feature, 1), "mean"]
        lines.append(
            f"|{feature}|{nonpreg:.4f}|{preg:.4f}|{preg-nonpreg:+.4f}|"
        )
    lines += [
        "",
        "当前主要差异不是两壁同向平移：妊娠组后壁平均ER更高，而前壁平均ER略低，使带符号的`ant-post`由非妊娠正偏移转为妊娠负偏移，绝对偏置增大。",
        "",
        "## 2. 技术变量调整",
        "",
        "模型固定调整为视频时长、前壁有效比例、后壁有效比例；第二行再增加临床来源文件。没有调参或模型搜索。",
        "",
        "|特征|来源调整|n|OR/SD|95% CI|p|",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in associations.itertuples(index=False):
        lines.append(
            f"|{row.feature}|{'yes' if row.source_adjusted else 'no'}|{row.n}|"
            f"{row.OR_per_SD:.3f}|{row.CI_low:.3f}–{row.CI_high:.3f}|{row.p:.4g}|"
        )
    lines += [
        "",
        "技术相关矩阵见`technical_correlations.csv`。",
        "",
        "各偏置指标与已测技术变量的最大绝对相关："
        + "；".join(
            f"{row.feature}={row.Pearson_r:+.3f}（{row.covariate}）"
            for row in correlation_summary.itertuples(index=False)
        )
        + "。",
        "",
        "## 3. 来源与月份分层",
        "",
        "完整逐来源、逐月份表在`stratified_by_source.csv`和`stratified_by_month.csv`。月份中同时含两类结局的方向计数：",
        "",
        "|特征|有双结局月份数|正方向月份|负方向月份|零差值月份|",
        "|---|---:|---:|---:|---:|",
    ]
    for row in direction_summary.itertuples(index=False):
        lines.append(
            f"|{row.feature}|{row.strata_with_both_outcomes}|"
            f"{row.positive_strata}|{row.negative_strata}|{row.exact_zero_strata}|"
        )
    lines += [
        "",
        "### 来源与年度叠加",
        "",
        "|年份|来源|n|患者|妊娠|ΔER均值|绝对ΔER均值|均值偏置项均值|MSE均值|",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in source_year.itertuples(index=False):
        lines.append(
            f"|{row.video_year}|{row.source_file}|{row.available_n}|{row.patients}|"
            f"{row.pregnancy_rate:.0%}|{row.mean_delta_ER:+.3f}|"
            f"{row.mean_abs_delta_ER:.3f}|{row.mean_bias_sq:.3f}|{row.mean_MSE:.3f}|"
        )
    lines += [
        "",
        "来源与年份显著共线：2024年可计算记录全部来自文件2；2025年包含19例文件2和59例文件3。因此年度分段不能视为独立时间验证，年度差异也不能与来源差异分开。",
        "",
        "## 4. 留一来源、留一月与年度分段",
        "",
        "留一来源和逐月留一的固定技术调整OR保存在CSV中；年度分段结果：",
        "",
        "|时间段|特征|n|OR/SD|95% CI|",
        "|---|---|---:|---:|---|",
    ]
    for row in by_period.itertuples(index=False):
        lines.append(
            f"|{row.period}|{row.feature}|{row.n}|"
            f"{row.OR_per_SD:.3f}|{row.CI_low:.3f}–{row.CI_high:.3f}|"
        )
    lines += [
        "",
        "## 5. 解释边界",
        "",
        "- 本分析围绕已观察到的均值偏置进行解释，不把`mean_bias_sq`升级为正式主特征。",
        "- 分层样本可能较小，单个月份的方向不能单独作为稳定或不稳定证据。",
        "- 技术调整只能减少已测时长和有效比例差异，不能消除选择偏倚、未测批次因素或真实生理差异。",
        "- 当前信号在文件2较强、文件3较弱，且来源与年份共线；尚不能证明跨来源/跨时间稳定。",
        "- 只有外部新时间段/新患者重复验证后，才能决定未来预注册MSE还是两壁平均ER偏置。",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
