"""Patient-grouped screening of all available clinical variables.

The clinical workbook is linked to the already-authorized examination cohort.
LASSO variable selection, imputation, encoding and scaling are fitted inside
each outer training fold. The selected variables are then refitted with a
fixed L2 logistic model and evaluated on the untouched outer test fold.

This is an internal exploratory analysis, not external validation. Patient
records and predictions remain under local ``outputs/``.
"""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis.mse_pregnancy_association.code.all_features_auc import VIDEO, grouped_splits
from analysis.mse_pregnancy_association.code.frozen_feature_auc import (
    CLINICAL,
    EXAM,
    GROUP,
    OUTCOME,
    SEED,
    bootstrap_oof,
    digest,
    save,
)

EXAM_ID = "临床检查编号"
WB_EXAM_ID = "临床检查编号"

# Distinct fields from the clinical record. The 7-variable baseline remains a
# separate scope so that the effect of adding the other measurements is clear.
EXTRA_NUMERIC_SOURCE = {
    "infertility_months": "临床_不育年限(月)",
    "endometrial_volume_ml": "临床_容积(ml)",
    "doppler_SD": "临床_S/D（内膜血流）",
    "doppler_PI": "临床_PI（内膜血流）",
    "doppler_RI": "临床_RI（内膜血流）",
    "doppler_VI": "临床_VI（内膜血流）",
    "doppler_FI": "临床_FI（内膜血流）",
    "doppler_VFI": "临床_VFI（内膜血流）",
    "clinical_peristalsis_positive_cpm": "临床_蠕动（正向）",
    "clinical_peristalsis_negative_cpm": "临床_蠕动（反向）",
}

EXTRA_CATEGORICAL_SOURCE = {
    "uterine_position": "临床_子宫位置",
    "endometrial_type": "临床_分型",
}

MODELING_SOURCE = {
    "transfer_embryo_count_component": "移植胚胎数_解析",
    "transferred_embryo_stage": "胚胎阶段_解析",
}

ALL_CLINICAL = [
    "age",
    "bmi",
    "infertility_years",
    "infertility_months",
    "endometrial_thickness_mm",
    "endometrial_volume_ml",
    "doppler_SD",
    "doppler_PI",
    "doppler_RI",
    "doppler_VI",
    "doppler_FI",
    "doppler_VFI",
    "uterine_position",
    "endometrial_type",
    "clinical_peristalsis_positive_cpm",
    "clinical_peristalsis_positive_disordered",
    "clinical_peristalsis_negative_cpm",
    "transfer_embryo_count_component",
    "transferred_embryo_stage",
]

BASELINE = [
    "age",
    "bmi",
    "infertility_years",
    "transfer_embryo_count_multiple",
    "endometrial_thickness_mm",
    "blastocyst_transfer",
    "endometrial_type_C",
]

CATEGORICAL = {
    "uterine_position",
    "endometrial_type",
    "transferred_embryo_stage",
}

SCOPES = {
    "baseline7": BASELINE,
    "all_clinical": ALL_CLINICAL,
    "all_clinical_video": ALL_CLINICAL + VIDEO,
}

GRIDS = {
    "l1_logistic": {"model__C": [0.01, 0.1, 1.0, 10.0]},
    "lasso_then_logistic": {"selector__estimator__C": [0.01, 0.1, 1.0, 10.0]},
}


class InterceptOrLogistic(LogisticRegression):
    """Logistic model that returns the training prevalence for zero features."""

    def fit(self, X, y):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Expected a two-dimensional feature matrix")
        if X.shape[1]:
            return super().fit(X, y)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError("Both outcome classes are required")
        self.classes_ = classes
        self.coef_ = np.zeros((1, 0), float)
        self.intercept_ = np.array(
            [np.log(counts[1] / counts[0])],
            float,
        )
        self.n_features_in_ = 0
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        if self.coef_.shape[1] == 0:
            probability = 1.0 / (1.0 + np.exp(-self.intercept_[0]))
            return np.column_stack(
                [
                    np.full(len(X), 1.0 - probability),
                    np.full(len(X), probability),
                ]
            )
        return super().predict_proba(X)

    def decision_function(self, X):
        X = np.asarray(X)
        if self.coef_.shape[1] == 0:
            return np.full(len(X), self.intercept_[0])
        return super().decision_function(X)


def _clean_id(series):
    return series.astype(str).str.strip().str.upper()


def _categorical(series):
    return series.astype("string").fillna("<missing>").astype(str)


def _coerce_numeric(audit, data, field, output):
    values = data[field]
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & numeric.isna()
    audit.append(
        dict(
            output_feature=output,
            source_field=field,
            role="included_numeric",
            nonmissing=int(numeric.notna().sum()),
            unique_nonmissing=int(numeric.nunique(dropna=True)),
            nonnumeric_or_nonnumeric_text=int(invalid.sum()),
            reason="numeric clinical measurement; nonnumeric clinical text retained in audit only",
        )
    )
    return numeric


def load_clinical_features(previous, workbook):
    cohort_path = previous / "analysis_cohort.csv"
    if not cohort_path.exists():
        raise FileNotFoundError(cohort_path)
    data = pd.read_csv(cohort_path, dtype={EXAM_ID: str, GROUP: str})
    data[EXAM_ID] = _clean_id(data[EXAM_ID])
    workbook = Path(workbook)
    sheets = pd.ExcelFile(workbook).sheet_names
    if len(sheets) < 2:
        raise ValueError("Clinical workbook must contain the final cohort and modeling-input sheets")

    final = pd.read_excel(workbook, sheet_name=0)
    modeling = pd.read_excel(workbook, sheet_name=1)

    # The final sheet contains the same examination ID in two blocks. Use the
    # first block for the one-to-one link.
    final = final.copy()
    final["_exam_id"] = _clean_id(final.iloc[:, 10])
    if final["_exam_id"].duplicated().any():
        raise ValueError("Final clinical sheet has duplicate examination IDs")
    final = final.set_index("_exam_id")

    modeling = modeling.copy()
    modeling[WB_EXAM_ID] = _clean_id(modeling[WB_EXAM_ID])
    if modeling[WB_EXAM_ID].duplicated().any():
        raise ValueError("Modeling-input sheet has duplicate examination IDs")
    modeling = modeling.set_index(WB_EXAM_ID)

    missing_final = sorted(set(data[EXAM_ID]) - set(final.index))
    missing_model = sorted(set(data[EXAM_ID]) - set(modeling.index))
    if missing_final or missing_model:
        raise ValueError(
            f"Clinical workbook lacks cohort examinations: final={len(missing_final)}, "
            f"modeling={len(missing_model)}"
        )

    audit = []
    numeric = pd.DataFrame(index=pd.Index(data[EXAM_ID], name=EXAM_ID))
    baseline_source = {
        "age": CLINICAL[0],
        "bmi": CLINICAL[1],
        "infertility_years": CLINICAL[2],
        "transfer_embryo_count_multiple": CLINICAL[3],
        "endometrial_thickness_mm": CLINICAL[4],
        "blastocyst_transfer": CLINICAL[5],
        "endometrial_type_C": CLINICAL[6],
    }
    cohort_indexed = data.set_index(EXAM_ID)
    for output, field in baseline_source.items():
        numeric[output] = _coerce_numeric(audit, cohort_indexed, field, output)
        audit[-1]["role"] = "included_baseline_and_all_clinical"

    final_subset = final.loc[data[EXAM_ID]]
    for output, field in EXTRA_NUMERIC_SOURCE.items():
        numeric[output] = _coerce_numeric(audit, final_subset, field, output)

    positive_text = final_subset["临床_蠕动（正向）"]
    numeric["clinical_peristalsis_positive_disordered"] = (
        positive_text.astype("string").str.strip().eq("杂乱无序").astype(float)
    )
    audit.append(
        dict(
            output_feature="clinical_peristalsis_positive_disordered",
            source_field="临床_蠕动（正向）",
            role="included_categorical_indicator",
            nonmissing=int(positive_text.notna().sum()),
            unique_nonmissing=int(positive_text.nunique(dropna=True)),
            nonnumeric_or_nonnumeric_text=int(
                (positive_text.notna() & pd.to_numeric(positive_text, errors="coerce").isna()).sum()
            ),
            reason="preserves the recorded disordered category instead of coercing it to missing",
        )
    )

    categorical = pd.DataFrame(index=pd.Index(data[EXAM_ID], name=EXAM_ID))
    for output, field in EXTRA_CATEGORICAL_SOURCE.items():
        categorical[output] = _categorical(final_subset[field])
        audit.append(
            dict(
                output_feature=output,
                source_field=field,
                role="included_categorical",
                nonmissing=int(final_subset[field].notna().sum()),
                unique_nonmissing=int(final_subset[field].nunique(dropna=True)),
                nonnumeric_or_nonnumeric_text=0,
                reason="clinical category; missing remains a separate one-hot category",
            )
        )

    model_subset = modeling.loc[data[EXAM_ID]]
    numeric["transfer_embryo_count_component"] = pd.to_numeric(
        model_subset[MODELING_SOURCE["transfer_embryo_count_component"]],
        errors="coerce",
    )
    audit.append(
        dict(
            output_feature="transfer_embryo_count_component",
            source_field=MODELING_SOURCE["transfer_embryo_count_component"],
            role="included_numeric",
            nonmissing=int(numeric["transfer_embryo_count_component"].notna().sum()),
            unique_nonmissing=int(numeric["transfer_embryo_count_component"].nunique()),
            nonnumeric_or_nonnumeric_text=0,
            reason="parsed transferred-embryo count from the existing modeling-input sheet",
        )
    )
    categorical["transferred_embryo_stage"] = _categorical(
        model_subset[MODELING_SOURCE["transferred_embryo_stage"]]
    )
    audit.append(
        dict(
            output_feature="transferred_embryo_stage",
            source_field=MODELING_SOURCE["transferred_embryo_stage"],
            role="included_categorical",
            nonmissing=int(categorical["transferred_embryo_stage"].notna().sum()),
            unique_nonmissing=int(categorical["transferred_embryo_stage"].nunique()),
            nonnumeric_or_nonnumeric_text=0,
            reason="parsed transferred-embryo stage from the existing modeling-input sheet",
        )
    )

    features = pd.concat([numeric, categorical], axis=1).reset_index()
    for column in VIDEO:
        features[column] = data[column].to_numpy()
    if len(features) != len(data) or not features[EXAM_ID].equals(data[EXAM_ID]):
        raise ValueError("Clinical feature rows do not preserve examination order")
    if features[EXAM_ID].duplicated().any():
        raise ValueError("Clinical feature table is not one row per examination")

    excluded = [
        ("文件1行/匹配状态/匹配依据", "linkage or audit metadata"),
        ("患者姓名/拼音/检查时间/路径/目录标识", "identifier, date, free text or path"),
        ("视频帧数/帧率/时长/视频胚胎类别", "video acquisition metadata, not a clinical predictor"),
        ("视频目录妊娠结局/生化妊娠/临床妊娠", "target or outcome leakage"),
        ("临床_手术结局", "only one nonmissing category in the linked cohort; no estimable contrast"),
        ("临床_胚胎情况", "free-text compound field; parsed count and stage are included"),
        ("临床_描述/临床_诊断", "free text without a prespecified coding scheme"),
        ("临床数据来源文件/来源行/块归属状态/核对依据/复核状态", "source-management metadata"),
    ]
    excluded_df = pd.DataFrame(excluded, columns=["fields", "exclusion_reason"])
    return data, features, pd.DataFrame(audit), excluded_df


def make_preprocessor(columns):
    numeric = [c for c in columns if c not in CATEGORICAL]
    categorical = [c for c in columns if c in CATEGORICAL]
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def make_pipeline(kind, columns):
    preprocess = make_preprocessor(columns)
    if kind == "l1_logistic":
        model = LogisticRegression(
            C=1.0,
            penalty="l1",
            solver="liblinear",
            max_iter=5000,
            random_state=SEED,
        )
        return Pipeline([("preprocess", preprocess), ("model", model)])
    if kind == "lasso_then_logistic":
        selector_model = LogisticRegression(
            C=1.0,
            penalty="l1",
            solver="liblinear",
            max_iter=5000,
            random_state=SEED,
        )
        model = InterceptOrLogistic(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            random_state=SEED,
        )
        return Pipeline(
            [
                ("preprocess", preprocess),
                ("selector", SelectFromModel(selector_model, threshold=1e-12)),
                ("model", model),
            ]
        )
    raise ValueError(kind)


def transformed_source_name(name):
    prefix, _, raw = name.partition("__")
    if prefix == "num":
        return raw
    if prefix == "cat":
        for field in sorted(CATEGORICAL, key=len, reverse=True):
            if raw == field or raw.startswith(field + "_"):
                return field
    return raw


def fitted_selection(model):
    preprocess = model.named_steps["preprocess"]
    names = list(preprocess.get_feature_names_out())
    if "selector" in model.named_steps:
        selected = np.asarray(model.named_steps["selector"].get_support(), bool)
        coefficients = np.asarray(model.named_steps["model"].coef_[0], float)
    else:
        coefficients_all = np.asarray(model.named_steps["model"].coef_[0], float)
        selected = np.abs(coefficients_all) > 1e-12
        coefficients = coefficients_all[selected]
    selected_names = [n for n, keep in zip(names, selected) if keep]
    rows = [
        dict(
            transformed_feature=name,
            source_feature=transformed_source_name(name),
            coefficient=float(coef),
        )
        for name, coef in zip(selected_names, coefficients)
    ]
    if len(rows) != len(selected_names):
        raise ValueError("Selected feature and coefficient lengths differ")
    return rows


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    data, features, feature_audit, excluded = load_clinical_features(
        args.previous, args.workbook
    )
    source_hashes = {
        str((args.previous / "analysis_cohort.csv").resolve()): digest(
            args.previous / "analysis_cohort.csv"
        ),
        str(Path(args.workbook).resolve()): digest(Path(args.workbook)),
    }
    protocol = dict(
        examinations=len(data),
        patients=int(data[GROUP].nunique()),
        outcome=OUTCOME,
        baseline=BASELINE,
        all_clinical=ALL_CLINICAL,
        video=VIDEO,
        scopes=SCOPES,
        models={
            "l1_logistic": "L1 Logistic directly screens through its training-fold coefficients",
            "lasso_then_logistic": "training-fold L1 Logistic selects features, then fixed L2 Logistic refits",
        },
        grids=GRIDS,
        outer_folds=args.folds,
        inner_folds=args.inner_folds,
        repeats=args.repeats,
        first_seed=SEED,
        patient_grouping="StratifiedGroupKFold at every outer and inner split",
        preprocessing="median imputation and scaling fitted only inside each training fold",
        categorical_encoding="one-hot fitted only inside each training fold; missing is a category",
        selection_location="inside the outer training fold and inside every inner tuning split",
        primary_metric="first-prespecified-seed outer-fold ROC AUC",
        bootstrap=args.bootstrap,
        uncertainty="patient-cluster bootstrap of fixed first-repeat OOF scores; no refit",
        excluded_fields=excluded.to_dict(orient="records"),
        source_sha256=source_hashes,
        frozen_features_or_tracking_changed=False,
        interpretation="internal exploratory variable screening, not external validation or causal inference",
        python=platform.python_version(),
        sklearn=sklearn.__version__,
        pandas=pd.__version__,
    )
    save(args.output / "protocol_before_fit.json", protocol)

    y = data[OUTCOME].to_numpy(int)
    groups = data[GROUP].to_numpy(str)
    metrics = []
    fold_audit = []
    selection_rows = []
    coefficient_rows = []
    grid_rows = []
    first_scores = None
    first_assignment = None

    for repeat in range(args.repeats):
        outer = grouped_splits(y, groups, args.folds, SEED + repeat)
        predictions = {
            f"{kind}_{scope}": np.full(len(data), np.nan)
            for kind in GRIDS
            for scope in SCOPES
        }
        assignment = np.full(len(data), -1, int)
        for fold, (train, test) in enumerate(outer, 1):
            assignment[test] = fold
            inner = grouped_splits(
                y[train],
                groups[train],
                args.inner_folds,
                SEED + 1000 + repeat * args.folds + fold,
            )
            for kind, grid in GRIDS.items():
                for scope, columns in SCOPES.items():
                    search = GridSearchCV(
                        make_pipeline(kind, columns),
                        grid,
                        cv=inner,
                        scoring="roc_auc",
                        n_jobs=1,
                        refit=True,
                        error_score="raise",
                    )
                    search.fit(features.iloc[train][columns], y[train])
                    model = search.best_estimator_
                    probability = model.predict_proba(features.iloc[test][columns])[:, 1]
                    predictions[f"{kind}_{scope}"][test] = probability
                    selection = fitted_selection(model)
                    selected_sources = {row["source_feature"] for row in selection}
                    for column in columns:
                        selection_rows.append(
                            dict(
                                repeat=repeat,
                                fold=fold,
                                model=kind,
                                scope=scope,
                                feature=column,
                                selected=column in selected_sources,
                            )
                        )
                    for row in selection:
                        coefficient_rows.append(
                            dict(repeat=repeat, fold=fold, model=kind, scope=scope, **row)
                        )
                    for params, score in zip(
                        search.cv_results_["params"],
                        search.cv_results_["mean_test_score"],
                    ):
                        grid_rows.append(
                            dict(
                                repeat=repeat,
                                fold=fold,
                                model=kind,
                                scope=scope,
                                parameters=json.dumps(params, sort_keys=True),
                                inner_AUC=float(score),
                            )
                        )
                    fold_audit.append(
                        dict(
                            repeat=repeat,
                            fold=fold,
                            model=kind,
                            scope=scope,
                            train_n=len(train),
                            test_n=len(test),
                            patient_overlap=0,
                            outer_test_AUC=float(roc_auc_score(y[test], probability)),
                            selected_transformed_features=len(selection),
                            selected_source_features=len(selected_sources),
                            chosen_parameters=json.dumps(search.best_params_, sort_keys=True),
                        )
                    )
        if (assignment < 0).any():
            raise ValueError("Some outer-fold examinations were not assigned")
        for name, score in predictions.items():
            if not np.isfinite(score).all():
                raise ValueError(f"Non-finite out-of-fold score: {name}")
            metrics.append(
                dict(
                    repeat=repeat,
                    seed=SEED + repeat,
                    model_scope=name,
                    AUC=float(roc_auc_score(y, score)),
                )
            )
        if repeat == 0:
            first_scores = predictions
            first_assignment = assignment

    if first_scores is None or first_assignment is None:
        raise ValueError("At least one repeat is required")

    draws = bootstrap_oof(data, first_scores, args.bootstrap, SEED + 1000)
    repeated = pd.DataFrame(metrics)
    summary_rows = []
    for kind in GRIDS:
        for scope in SCOPES:
            name = f"{kind}_{scope}"
            low, high = np.quantile(draws[name], [0.025, 0.975])
            values = repeated.loc[repeated["model_scope"] == name, "AUC"]
            summary_rows.append(
                dict(
                    model=kind,
                    scope=scope,
                    n=len(data),
                    patients=int(data[GROUP].nunique()),
                    positive=int(y.sum()),
                    negative=int((1 - y).sum()),
                    OOF_AUC=float(roc_auc_score(y, first_scores[name])),
                    CI_low=float(low),
                    CI_high=float(high),
                    repeat_mean_AUC=float(values.mean()),
                    repeat_min_AUC=float(values.min()),
                    repeat_max_AUC=float(values.max()),
                )
            )
    result = pd.DataFrame(summary_rows)

    differences = []
    pairs = [
        ("all_clinical", "baseline7"),
        ("all_clinical_video", "all_clinical"),
    ]
    for kind in GRIDS:
        for comparison, reference in pairs:
            added = f"{kind}_{comparison}"
            baseline = f"{kind}_{reference}"
            delta = draws[added] - draws[baseline]
            low, high = np.quantile(delta, [0.025, 0.975])
            current = result.loc[
                (result["model"] == kind) & (result["scope"] == comparison)
            ].iloc[0]
            base = result.loc[
                (result["model"] == kind) & (result["scope"] == reference)
            ].iloc[0]
            differences.append(
                dict(
                    model=kind,
                    comparison=comparison,
                    reference=reference,
                    delta_AUC=float(current["OOF_AUC"] - base["OOF_AUC"]),
                    CI_low=float(low),
                    CI_high=float(high),
                    repeat_mean_delta_AUC=float(
                        current["repeat_mean_AUC"] - base["repeat_mean_AUC"]
                    ),
                )
            )
    differences = pd.DataFrame(differences)

    selection = pd.DataFrame(selection_rows)
    stability = (
        selection.groupby(["model", "scope", "feature"], as_index=False)
        .agg(selection_fraction=("selected", "mean"), selected_fits=("selected", "sum"))
        .sort_values(["model", "scope", "selection_fraction"], ascending=[True, True, False])
    )
    stability["eligible_fits"] = args.repeats * args.folds
    coefficients = pd.DataFrame(coefficient_rows)
    grids = pd.DataFrame(grid_rows)
    audit = pd.DataFrame(fold_audit)
    availability = pd.DataFrame(
        [
            dict(
                scope=scope,
                feature=column,
                available=int(features[column].notna().sum()),
                missing=int(features[column].isna().sum()),
                unique=int(features[column].nunique(dropna=True)),
                type="categorical" if column in CATEGORICAL else "numeric",
            )
            for scope, columns in SCOPES.items()
            for column in columns
        ]
    )
    oof = data[["case_id", EXAM_ID, GROUP, OUTCOME]].copy()
    oof["fold"] = first_assignment
    for column, values in first_scores.items():
        oof[column] = values

    for name, frame in [
        ("clinical_feature_audit", feature_audit),
        ("excluded_clinical_fields", excluded),
        ("feature_availability", availability),
        ("selection_stability", stability),
        ("fold_selection", selection),
        ("selected_feature_coefficients", coefficients),
        ("inner_grid_scores", grids),
        ("fold_audit", audit),
        ("repeat_metrics", repeated),
        ("AUC_comparison", result),
        ("paired_AUC_differences", differences),
        ("out_of_fold_scores", oof),
    ]:
        frame.to_csv(args.output / f"{name}.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(args.output / "clinical_variable_screening.xlsx", engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="AUC", index=False)
        differences.to_excel(writer, sheet_name="paired_delta", index=False)
        stability.to_excel(writer, sheet_name="selection_stability", index=False)
        availability.to_excel(writer, sheet_name="availability", index=False)
        audit.to_excel(writer, sheet_name="fold_audit", index=False)
        oof.to_excel(writer, sheet_name="OOF", index=False)

    if any(digest(Path(path)) != value for path, value in source_hashes.items()):
        raise RuntimeError("Source cohort or clinical workbook changed during screening")

    summary = dict(
        examinations=len(data),
        patients=int(data[GROUP].nunique()),
        positive=int(y.sum()),
        negative=int((1 - y).sum()),
        all_clinical_features=len(ALL_CLINICAL),
        video_features=len(VIDEO),
        auc_comparison=result.to_dict(orient="records"),
        paired_differences=differences.to_dict(orient="records"),
        source_unchanged=True,
        frozen_tracking_or_features_changed=False,
        clinical_outcome_used_only_to_fit_and_evaluate_pregnancy_model=True,
        external_validation=False,
    )
    save(args.output / "summary.json", summary)

    lines = [
        "# 全部临床变量的分层变量筛选",
        "",
        f"{len(data)}次检查、{data[GROUP].nunique()}名患者；妊娠{y.sum()}、非妊娠{int((1-y).sum())}。",
        f"全部临床字段共{len(ALL_CLINICAL)}项；另与既往7项临床基线、6项视频特征分别比较。",
        "L1/LASSO筛选、缺失填补、编码和标准化都只在训练折内拟合；外层和内层均按患者分组。",
        "`lasso_then_logistic`先在训练折用L1 Logistic选变量，再用固定L2 Logistic在选中的变量上重新拟合，测试折不参与筛选或调参。",
        "结果用于内部探索，不是外部验证；没有根据结局修改追踪、ROI、0.6、活动计数或视频特征公式。",
        "",
        "## 变量范围",
        "",
        "- 全部临床变量：年龄、BMI、不孕年限（年和月）、内膜厚度、内膜容积、内膜血流S/D、PI、RI、VI、FI、VFI、子宫位置、内膜分型、临床正/反向蠕动记录、移植胚胎数和胚胎阶段；手术结局在当前队列仅一类非缺失值，记录后排除。",
        "- 自由文本诊断、描述、姓名、日期、路径、结局和管理字段不作为预测变量；`临床_胚胎情况`由已解析的胚胎数和阶段代替。",
        "- 视频特征为既有的双壁幅度、显著传播活动次数、次数/有效分钟、单block ER CC和ER MSE。",
        "",
        "## 模型结果",
        "",
        "|模型|变量范围|折外AUC|95%区间|重复均值|",
        "|---|---|---:|---|---:|",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"|{row.model}|{row.scope}|{row.OOF_AUC:.4f}|"
            f"{row.CI_low:.4f}–{row.CI_high:.4f}|{row.repeat_mean_AUC:.4f}|"
        )
    lines += ["", "## 配对增益", ""]
    for row in differences.itertuples(index=False):
        lines.append(
            f"- {row.model}: {row.comparison}−{row.reference}，"
            f"ΔAUC={row.delta_AUC:+.4f}，95%区间{row.CI_low:+.4f}–{row.CI_high:+.4f}。"
        )
    lines += [
        "",
        "## 训练折变量筛选稳定性",
        "",
        "以下列出`lasso_then_logistic`各范围选择频率最高的10项；完整20项及两个模型见`selection_stability.csv`。",
    ]
    for scope in ("all_clinical", "all_clinical_video"):
        subset = stability.loc[
            (stability["model"] == "lasso_then_logistic")
            & (stability["scope"] == scope)
        ].head(10)
        fold_subset = audit.loc[
            (audit["model"] == "lasso_then_logistic")
            & (audit["scope"] == scope)
        ]
        lines += [
            "",
            f"### {scope}",
            "",
            f"每个训练折选出原始变量数中位数{fold_subset['selected_source_features'].median():.0f}，"
            f"范围{fold_subset['selected_source_features'].min()}–"
            f"{fold_subset['selected_source_features'].max()}；"
            f"未选出任何变量的折占{fold_subset['selected_source_features'].eq(0).mean():.1%}。",
            "",
            "|变量|选择频率|",
            "|---|---:|",
        ]
        for row in subset.itertuples(index=False):
            lines.append(f"|{row.feature}|{row.selection_fraction:.0%}|")
    lines += [
        "",
        "筛选频率只描述训练折中的稳定性，不在同一数据上再按最高频率挑选变量后重报AUC。",
        "次数与频率、胚胎数和阶段、内膜分型分类与C型字段高度相关；L1可能只保留其中一个，这不等于另一个没有临床意义。",
        "妊娠结局为临床妊娠，未包含活产、周期取消或移植决策信息；不能据此作因果或个体预测解释。",
        "",
        "ER-MSE的文献方向与当前数据不同，见 `docs/ER_MSE_LITERATURE_CHECK.md`。",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result.to_string(index=False), flush=True)
    print(differences.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    run(parser.parse_args())
