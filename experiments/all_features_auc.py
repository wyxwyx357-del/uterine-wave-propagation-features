"""Nested patient-grouped comparison of Logistic, L1 Logistic and RBF SVM.

Only existing examination-level feature scalars are used. No window averaging,
outcome-based feature extraction, patient QC predictors or global selection.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.frozen_feature_auc import (CLINICAL, AMPLITUDE, COORD, GROUP,
    OUTCOME, SEED, digest, save, bootstrap_oof)
from experiments.batch_wave_counts import COUNT, RATE

VIDEO = AMPLITUDE+[COUNT, RATE]+COORD
GRIDS = {'logistic': None, 'lasso': {'model__C': [.01, .1, 1., 10.]},
         'svm': {'model__C': [.1, 1., 10.], 'model__gamma': ['scale', .01, .1]}}


def pipeline(kind):
    if kind == 'svm':
        model = SVC(kernel='rbf', probability=False, class_weight=None)
    else:
        model = LogisticRegression(C=1e6 if kind=='logistic' else 1.,
            penalty='l1' if kind=='lasso' else 'l2', solver='liblinear',
            max_iter=5000, random_state=SEED)
    return Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('scale', StandardScaler()), ('model', model)])


def grouped_splits(y, groups, folds, seed):
    splits = list(StratifiedGroupKFold(n_splits=folds, shuffle=True,
                  random_state=seed).split(np.zeros(len(y)), y, groups))
    for a, b in splits:
        assert not set(groups[a]) & set(groups[b])
        if len(np.unique(y[a])) != 2 or len(np.unique(y[b])) != 2:
            raise ValueError('A prespecified fold lacks both outcome classes')
    return splits


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.previous/'analysis_cohort.csv'
    before = digest(source)
    data = pd.read_csv(source, dtype={GROUP: str})
    protocol = dict(video_features=VIDEO, clinical_features=CLINICAL,
        excluded='window/block ER,RMS,energy fraction; raw spectral power; QC; speed; direction',
        coordination='only existing single-block CC/MSE, other blocks remain missing',
        grids=GRIDS, outer_folds=args.folds, inner_folds=args.inner_folds,
        repeats=args.repeats, first_seed=SEED, bootstrap=args.bootstrap,
        tuning='inner patient-grouped ROC AUC only; imputation and scaling inside pipeline',
        comparisons='each model: same clinical-only baseline vs clinical plus all six video features',
        feature_selection='L1 coefficients fitted only within training; no preselection',
        SVM_score='decision function, not calibrated pregnancy probability',
        uncertainty='cluster bootstrap of fixed first-repeat OOF scores; no bootstrap refit',
        source_sha256=before, algorithms_or_thresholds_changed=False,
        interpretation='internal exploratory comparison; no winning-model external validation')
    save(args.output/'protocol_before_fit.json', protocol)
    y = data[OUTCOME].to_numpy(int)
    groups = data[GROUP].to_numpy(str)
    columns = {'clinical': CLINICAL, 'all': CLINICAL+VIDEO}
    metrics = []
    audit = []
    coefficients = []
    grid_rows = []
    primary = {}
    first_assignment = None
    for repeat in range(args.repeats):
        outer = grouped_splits(y, groups, args.folds, SEED+repeat)
        scores = {f'{kind}_{scope}': np.full(len(y), np.nan)
                  for kind in GRIDS for scope in columns}
        assignment = np.full(len(y), -1, int)
        for fold, (train, test) in enumerate(outer, 1):
            assignment[test] = fold
            inner = grouped_splits(y[train], groups[train], args.inner_folds,
                                   SEED+1000+repeat*args.folds+fold)
            for kind, grid in GRIDS.items():
                for scope, names in columns.items():
                    model = pipeline(kind)
                    if grid is not None:
                        search = GridSearchCV(model, grid, cv=inner, scoring='roc_auc',
                                              n_jobs=1, refit=True, error_score='raise')
                        search.fit(data.iloc[train][names], y[train])
                        model = search.best_estimator_
                        best = search.best_params_
                        for params, value in zip(search.cv_results_['params'],
                                                 search.cv_results_['mean_test_score']):
                            grid_rows.append(dict(repeat=repeat, fold=fold, model=kind,
                                scope=scope, parameters=json.dumps(params), inner_AUC=float(value)))
                    else:
                        model.fit(data.iloc[train][names], y[train])
                        best = {'C': 1e6}
                    score = (model.decision_function(data.iloc[test][names]) if kind=='svm'
                             else model.predict_proba(data.iloc[test][names])[:, 1])
                    scores[f'{kind}_{scope}'][test] = score
                    audit.append(dict(repeat=repeat, fold=fold, model=kind, scope=scope,
                        train_n=len(train), test_n=len(test), patient_overlap=0,
                        inner_patient_overlap=0, outer_test_AUC=float(roc_auc_score(y[test], score)),
                        chosen_parameters=json.dumps(best)))
                    if kind=='lasso':
                        for name, coef in zip(names, model.named_steps['model'].coef_[0]):
                            coefficients.append(dict(repeat=repeat, fold=fold, scope=scope,
                                                     feature=name, coefficient=float(coef)))
        assert (assignment > 0).all()
        for name, score in scores.items():
            assert np.isfinite(score).all()
            metrics.append(dict(repeat=repeat, model=name, AUC=float(roc_auc_score(y, score))))
        if repeat==0:
            primary = scores
            first_assignment = assignment
    prior = pd.read_csv(args.previous/'wave_frequency_main'/'out_of_fold_predictions.csv')
    if data['case_id'].tolist() != prior['case_id'].tolist():
        raise ValueError('Prior paired cohort order changed')
    np.testing.assert_array_equal(first_assignment, prior['fold'])
    np.testing.assert_allclose(primary['logistic_clinical'], prior['clinical'], atol=1e-10, rtol=1e-10)
    draws = bootstrap_oof(data, primary, args.bootstrap, SEED+1000)
    repeated = pd.DataFrame(metrics)
    summary = []
    for kind in GRIDS:
        row = dict(model=kind, n=len(data), patients=int(data[GROUP].nunique()))
        for scope in columns:
            name = f'{kind}_{scope}'
            lo, hi = np.quantile(draws[name], [.025, .975])
            row[scope+'_OOF_AUC'] = float(roc_auc_score(y, primary[name]))
            row[scope+'_CI_low'] = float(lo)
            row[scope+'_CI_high'] = float(hi)
            row[scope+'_repeat_mean'] = float(repeated.loc[repeated['model']==name, 'AUC'].mean())
        lo, hi = np.quantile(draws[f'{kind}_all']-draws[f'{kind}_clinical'], [.025, .975])
        row.update(delta_AUC=row['all_OOF_AUC']-row['clinical_OOF_AUC'],
                   delta_CI_low=float(lo), delta_CI_high=float(hi))
        summary.append(row)
    results = pd.DataFrame(summary)
    coeff = pd.DataFrame(coefficients)
    coeff['selected'] = coeff['coefficient'].abs() > 1e-12
    stability = coeff.groupby(['scope','feature'], as_index=False).agg(
        selection_fraction=('selected','mean'), mean_coefficient=('coefficient','mean'))
    # Descriptive across training folds, not another round of variable selection.
    available = data[VIDEO].notna().sum().rename('available').rename_axis('feature').reset_index()
    oof = data[['case_id', GROUP, OUTCOME]].copy()
    oof['fold'] = first_assignment
    for name, score in primary.items():
        oof[name] = score
    oof.to_csv(args.output/'out_of_fold_scores.csv', index=False, encoding='utf-8-sig')
    audit_df = pd.DataFrame(audit)
    grids_df = pd.DataFrame(grid_rows)
    for name, frame in [('AUC_comparison', results), ('repeat_metrics', repeated),
        ('fold_audit', audit_df), ('inner_grid_scores', grids_df),
        ('L1_training_coefficients', coeff), ('L1_selection_stability', stability),
        ('feature_availability', available)]:
        frame.to_csv(args.output/(name+'.csv'), index=False, encoding='utf-8-sig')
    with pd.ExcelWriter(args.output/'all_features_LASSO_SVM_AUC.xlsx', engine='openpyxl') as writer:
        results.to_excel(writer, sheet_name='AUC', index=False)
        repeated.to_excel(writer, sheet_name='repeats', index=False)
        available.to_excel(writer, sheet_name='availability', index=False)
        stability.to_excel(writer, sheet_name='L1_stability', index=False)
        audit_df.to_excel(writer, sheet_name='fold_audit', index=False)
        oof.to_excel(writer, sheet_name='OOF', index=False)
    if digest(source) != before:
        raise RuntimeError('Source features changed during modeling')
    save(args.output/'summary.json', dict(rows=summary, source_unchanged=True,
        previous_clinical_baseline_reproduced=True, frozen_algorithms_changed=False))
    lines = ['# 全部现有患者级特征：Logistic / L1 / SVM', '',
        f'{len(data)}次检查、{data[GROUP].nunique()}名患者；7项临床变量+6项视频变量。',
        '视频变量：前壁幅度、后壁幅度、显著传播活动次数、次数/有效分钟、既有单block ER-CC、ER-MSE。',
        f'CC/MSE共有{int(data[COORD].notna().all(axis=1).sum())}次既有单block可用，其他记缺失；没有平均多block。窗口ER/RMS/energy fraction没有冻结患者聚合，不输入。QC不作为生理预测变量。',
        '计数与频率相关，因此保留两者同时输入并由训练折L1收缩；未按结局预筛特征。L1是分类Logistic惩罚，不是连续结局Lasso回归。',
        f'外层{args.folds}折按患者分组，重复{args.repeats}个预定种子；内层{args.inner_folds}折按患者分组调参。插补、标准化、L1选择全部在训练折内完成。',
        '每个模型都与同算法临床基线比较，不跨模型基线宣称增益；主表使用第一个预定种子，重复均值单列，不挑最好一轮。',
        'SVM使用RBF核和decision_function，分数不是校准妊娠概率。折外分数的95%区间为固定预测的患者聚类bootstrap，不重训，未覆盖全部训练不确定性。',
        '', '|模型|临床AUC|临床+全部视频AUC|差值|95%差值区间|', '|---|---:|---:|---:|---|']
    for r in summary:
        lines.append(f'|{r["model"]}|{r["clinical_OOF_AUC"]:.4f}|{r["all_OOF_AUC"]:.4f}|{r["delta_AUC"]:+.4f}|{r["delta_CI_low"]:+.4f}–{r["delta_CI_high"]:+.4f}|')
    lines += ['', '训练折非零系数频率仅为稳定性描述，不能在同一数据上据此选变量再报一次AUC。',
              '这是内部探索性比较，不因换模型而消除活动计数误差或形成独立外部验证。', '',
              '方法参考：https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html',
              'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html',
              'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(results.to_string(index=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--inner-folds', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--bootstrap', type=int, default=1000)
    run(parser.parse_args())
