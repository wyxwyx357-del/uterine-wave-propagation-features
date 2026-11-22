"""Final authorized static-method exploration: static-only block Elastic Net.

Clinical/dynamic covariates remain in every model with L2, never L1 selection.
No final full-data model or frequency-selected refit is produced.
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from analysis.pregnancy_prediction.code import block_elasticnet as en
from analysis.pregnancy_prediction.code.static_dynamic_pca_models import NUMERIC, STAGE, DYNAMIC
from analysis.mse_pregnancy_association.code.static_dynamic_pca_test import data, sha, save
from analysis.pregnancy_prediction.code.frozen_foldwise_auc import evaluate_frozen_scores

ROOT = Path(os.environ.get('DATA_ROOT', 'data'))
OLD = ROOT / 'outputs/static_dynamic_pca_nested_20261121'
STATIC = ROOT / 'outputs/static_endometrium_radiomics_v3_manual2_20261120'
OUT = ROOT / 'outputs/static_elasticnet_final_exploration_20261122'
MODELS = ['Clinical', 'Clinical_Static_EN', 'Clinical_Dynamic', 'Clinical_Dynamic_Static_EN']
AUGMENTED = ['Clinical_Static_EN', 'Clinical_Dynamic_Static_EN']
SEED, B = 20261122, 10000
_DATA = None


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig')


def preprocess(train, validation, columns, use_dynamic):
    matrices = []
    clinical_imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(train[NUMERIC])
    clinical_scale = StandardScaler().fit(clinical_imputer.transform(train[NUMERIC]))
    stage = SimpleImputer(strategy='most_frequent', keep_empty_features=True).fit(train[[STAGE]])
    static_imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(train[columns])
    static_scale = StandardScaler().fit(static_imputer.transform(train[columns]))
    np.testing.assert_allclose(clinical_imputer.statistics_, np.nanmedian(train[NUMERIC], axis=0))
    np.testing.assert_allclose(static_imputer.statistics_, np.nanmedian(train[columns], axis=0))
    np.testing.assert_allclose(static_scale.mean_, static_imputer.transform(train[columns]).mean(axis=0))
    dynamic_scale = StandardScaler().fit(train[DYNAMIC]) if use_dynamic else None
    for frame in [train, validation]:
        blocks = [clinical_scale.transform(clinical_imputer.transform(frame[NUMERIC])), stage.transform(frame[[STAGE]])]
        if use_dynamic:
            blocks.append(dynamic_scale.transform(frame[DYNAMIC]))
        blocks.append(static_scale.transform(static_imputer.transform(frame[columns])))
        matrices.append(np.column_stack(blocks))
    state = dict(n_fit_samples=len(train), clinical_medians=clinical_imputer.statistics_.tolist(),
        clinical_means=clinical_scale.mean_.tolist(), clinical_scales=clinical_scale.scale_.tolist(),
        stage_mode=stage.statistics_.tolist(), static_medians=static_imputer.statistics_.tolist(),
        static_means=static_scale.mean_.tolist(), static_scales=static_scale.scale_.tolist(),
        dynamic_means=dynamic_scale.mean_.tolist() if use_dynamic else [],
        dynamic_scales=dynamic_scale.scale_.tolist() if use_dynamic else [])
    n_base = 11 if use_dynamic else 6
    assert matrices[0].shape[1] == matrices[1].shape[1] == n_base + 101
    assert all(np.isfinite(matrix).all() for matrix in matrices)
    return matrices[0], matrices[1], state


def fit_job(job):
    global _DATA
    if _DATA is None:
        _DATA = data()
    frame, columns = _DATA
    train, test = frame.iloc[job['train']], frame.iloc[job['test']]
    assert not set(train.patient_id) & set(test.patient_id)
    assignment = np.asarray(job['inner_assignment'], int)
    results = []
    with threadpool_limits(limits=1):
        for model in AUGMENTED:
            dynamic = model == 'Clinical_Dynamic_Static_EN'
            n_base = 11 if dynamic else 6
            scores = {(alpha, lam): [] for alpha in en.ALPHAS for lam in en.LAMBDAS}
            diagnostics, inner_preprocessing = [], []
            for inner_fold in range(1, 6):
                a, b = np.flatnonzero(assignment != inner_fold), np.flatnonzero(assignment == inner_fold)
                fit_train, fit_validation = train.iloc[a], train.iloc[b]
                assert not set(fit_train.patient_id) & set(fit_validation.patient_id)
                assert fit_train.outcome.nunique() == fit_validation.outcome.nunique() == 2
                X, V, state = preprocess(fit_train, fit_validation, columns, dynamic)
                y, vy = fit_train.outcome.to_numpy(int), fit_validation.outcome.to_numpy(int)
                inner_preprocessing.append(dict(inner_fold=inner_fold, train_indices=a.tolist(), validation_indices=b.tolist(),
                    patient_overlap=0, preprocessing=state))
                for alpha in en.ALPHAS:
                    warm = None
                    for lam in en.LAMBDAS:
                        fitted = en.fit(X, y, n_base, alpha, lam, warm)
                        eta = fitted['intercept'] + V @ np.asarray(fitted['coef'])
                        loss = float(np.mean(np.logaddexp(0, eta) - vy * eta))
                        scores[(alpha, lam)].append(loss)
                        diagnostics.append(dict(inner_fold=inner_fold, alpha=alpha, lambda_sum_loss=lam,
                            validation_logloss=loss, selected_static=fitted['selected_static'],
                            iterations=fitted['iterations'], KKT_residual=fitted['KKT_residual'],
                            lbfgsb_initial_KKT=fitted['lbfgsb_initial_KKT'],
                            proximal_polish_iterations=fitted['proximal_polish_iterations']))
                        warm = fitted
            tuning = [dict(alpha=alpha, lambda_sum_loss=lam, C_reference=1 / lam,
                mean_inner_logloss=float(np.mean(values)), SD_inner_logloss=float(np.std(values)))
                for (alpha, lam), values in scores.items()]
            best = min(tuning, key=lambda row: (row['mean_inner_logloss'], -row['lambda_sum_loss'], -row['alpha']))
            X, T, state = preprocess(train, test, columns, dynamic)
            fitted = en.fit(X, train.outcome.to_numpy(int), n_base, best['alpha'], best['lambda_sum_loss'])
            probability = en.predict(fitted, T)
            assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
            static_coef = np.asarray(fitted['coef'][n_base:])
            assert len(static_coef) == 101
            results.append(dict(model=model, **fitted, predicted_probability=probability.tolist(),
                n_fixed_base=n_base, fixed_base_names=NUMERIC + [STAGE] + (DYNAMIC if dynamic else []),
                static_names=columns, static_selected=(static_coef != 0).tolist(),
                preprocessing=state, inner_tuning=tuning, inner_fit_diagnostics=diagnostics,
                inner_preprocessing=inner_preprocessing, classes=[0, 1]))
    save(OUT / 'jobs' / (job['key'] + '.json'), dict(**job, results=results))
    return job['key']


def prepare():
    OUT.mkdir(exist_ok=False)
    (OUT / 'jobs').mkdir()
    with threadpool_limits(limits=1):
        numerical = en.verification()
    frame, columns = data()
    jobs = json.loads((OLD / 'FIT_JOBS_LOCAL.json').read_text(encoding='utf-8'))
    old_protocol = json.loads((OLD / 'PROTOCOL_BEFORE_FIT.json').read_text(encoding='utf-8'))
    assert len(jobs) == 50 and frame.patient_id.nunique() == 174 and len(frame) == 180
    assert old_protocol['static_columns'] == columns
    source_files = [OLD / 'FIT_JOBS_LOCAL.json', OLD / 'PROTOCOL_BEFORE_FIT.json', OLD / '03_OOF_PREDICTIONS_LOCAL.csv',
        OLD / '05_SELECTED_PARAMETERS.csv', STATIC / 'STATIC_FEATURES_JOINT180.csv', STATIC / 'STATIC_FEATURE_INVENTORY.csv',
        Path(next(path for path in old_protocol['source_hashes'] if Path(path).name == 'MASTER_CLINICAL_SOURCE_ALIGNED_311.csv'))]
    hashes = {str(path): sha(path) for path in source_files}
    assert all(sha(Path(path)) == checksum for path, checksum in old_protocol['source_hashes'].items())
    for job in jobs:
        assert not set(frame.iloc[job['train']].patient_id) & set(frame.iloc[job['test']].patient_id)
    protocol = dict(analysis_name='exploratory supervised feature-selection analysis',
        version='final_static_ElasticNet_20261122', n_exams=180, n_unique_patients=174,
        repeated_patient_groups=6, pregnancy_n=117, non_pregnancy_n=63,
        clinical_fixed_names=NUMERIC + [STAGE], dynamic_fixed_names=DYNAMIC, static_names=columns,
        only_new_method='Elastic Net, only static block subject to L1 selection',
        objective='sum(logloss)+lambda/2*||base||²+lambda*(1-alpha)/2*||static||²+lambda*alpha*||static||1; intercept free',
        base_handling='clinical6 and dynamic5 always included, original Ridge-style L2 scale; never L1-selected',
        alpha_grid=en.ALPHAS, lambda_sum_loss_grid=en.LAMBDAS, C_reference='C=1/lambda; original six-value C grid',
        lambda_normalization='mean-loss solver divides complete objective by training_n; no change in optimum',
        preprocessing='inner/outer training-only clinical median,zscore; stage mode/unscaled; dynamic zscore; static median,zscore',
        missing_inputs='same complete180 cohort, no new row deletion or imputation from full data',
        selection='lowest mean inner5fold logloss; exact ties stronger lambda then larger alpha',
        outer_inner_maps='reuse exact previous global patient-grouped outer5,inner5,10 repeats',
        no_outer_test_selection=True, zero_selected_static_allowed=True,
        support='exact nonzero fitted static coefficient; no frequency cutoff or final subset refit',
        baseline_controls='reuse saved Clinical and Clinical_Dynamic OOF, identical patient/fold maps; no new baseline method',
        main_comparisons=[['Clinical_Static_EN', 'Clinical'], ['Clinical_Dynamic_Static_EN', 'Clinical_Dynamic']],
        evaluation='within-fold positive-negative-pair-weighted AUC per repeat, then mean10',
        bootstrap=dict(iterations=B, seed=SEED, unit='patient', scope='paired fixed OOF conditional95%; not full training-refit CI'),
        selection_frequencies='all101 each of two scenarios, selected outer fits/50; overlapping training samples, descriptive only',
        numerical_implementation=dict(method='L-BFGS-B nonnegative variable split, same convex EN objective',
            options=en.OPTIONS, KKT_tolerance=en.KKT_TOLERANCE, checks=numerical,
            warning='numerical convergence tolerance is not a clinical feature-validity cutoff'),
        STOP_RULE='After this one EN analysis, stop static method searches regardless of results. No LASSO/RF/SVM/newPCA/binWidth search.',
        no_final_full180_fit_or_frequency_selected_refit=True,
        source_hashes=hashes, script_hashes={p.name:sha(p) for p in [Path(__file__), Path(en.__file__)]},
        scope='prior180 data already explored; not independent validation; no superiority search',
        references=['https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression',
                    'https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html'])
    save(OUT / 'PROTOCOL_BEFORE_FIT.json', protocol)
    save(OUT / 'STOP_RULE.json', dict(status='final_static_exploration_in_progress', rule=protocol['STOP_RULE'],
        final_model_refit_forbidden=True, results_must_be_retained=True))
    csv('01_COHORT_LOCAL.csv', frame[['case_id', 'exam_id', 'patient_id', 'outcome']])
    return frame, columns, jobs, protocol


def resume_after_numerical_correction():
    """Resume frozen fits after a documented solver-only accuracy correction."""
    protocol = json.loads((OUT / 'PROTOCOL_BEFORE_FIT.json').read_text(encoding='utf-8'))
    assert all(sha(Path(path)) == checksum for path, checksum in protocol['source_hashes'].items())
    assert protocol['alpha_grid'] == en.ALPHAS and protocol['lambda_sum_loss_grid'] == en.LAMBDAS
    assert not (OUT / '08_MODEL_PERFORMANCE.csv').exists()
    frame, columns = data()
    assert protocol['static_names'] == columns and len(frame) == protocol['n_exams']
    with threadpool_limits(limits=1):
        checks = en.verification()
    if 'numerical_correction' not in protocol:
        assert not list((OUT / 'jobs').glob('*.json'))
        protocol['numerical_correction'] = dict(
            reason='L-BFGS-B relative objective stopping occurred before frozen KKT accuracy was reached',
            original_attempt='PROTOCOL_INITIAL_NUMERICAL_ATTEMPT.json',
            diagnostic='NUMERICAL_DIAGNOSTIC_INITIAL.json',
            change='Same-objective accelerated proximal polishing, static-only soft threshold',
            objective_grid_tolerance_features_splits_unchanged=True,
            no_EN_outer_predictions_completed_or_performance_inspected=True)
    protocol['numerical_implementation'].update(
        method='L-BFGS-B variable split plus same-objective accelerated proximal polishing if needed', checks=checks)
    protocol['script_hashes'] = {p.name: sha(p) for p in [Path(__file__), Path(en.__file__)]}
    save(OUT / 'PROTOCOL_BEFORE_FIT.json', protocol)
    jobs = json.loads((OLD / 'FIT_JOBS_LOCAL.json').read_text(encoding='utf-8'))
    return frame, columns, jobs, protocol


def collect(frame, columns, jobs, protocol):
    prior = pd.read_csv(OLD / '03_OOF_PREDICTIONS_LOCAL.csv', dtype={'exam_id': str, 'patient_id': str})
    controls = prior[prior.model.isin(['Clinical', 'Clinical_Dynamic'])].copy()
    predictions, choices, coefs, tuning, numerical = [], [], [], [], []
    for job in jobs:
        result = json.loads((OUT / 'jobs' / (job['key'] + '.json')).read_text(encoding='utf-8'))
        for fit in result['results']:
            model = fit['model']
            choices.append(dict(model=model, repeat=job['repeat'], fold=job['fold'], alpha=fit['alpha'],
                lambda_sum_loss=fit['lambda_sum_loss'], C_reference=1 / fit['lambda_sum_loss'],
                n_static_selected=fit['selected_static'], n_fixed_base=fit['n_fixed_base'],
                outer_fit_KKT=fit['KKT_residual'], outer_fit_iterations=fit['iterations']))
            for index, probability in zip(job['test'], fit['predicted_probability']):
                row = frame.iloc[index]
                predictions.append(dict(model=model, repeat=job['repeat'], fold=job['fold'],
                    exam_id=row.exam_id, patient_id=row.patient_id, outcome=int(row.outcome), predicted_probability=probability))
            for j, (name, coefficient) in enumerate(zip(fit['fixed_base_names'] + columns, fit['coef'])):
                static = j >= fit['n_fixed_base']
                coefs.append(dict(model=model, repeat=job['repeat'], fold=job['fold'], feature_name=name,
                    coefficient=coefficient, intercept=fit['intercept'], role='static_EN' if static else 'fixed_base_L2',
                    selected=bool(coefficient != 0) if static else True))
            tuning.extend(dict(model=model, repeat=job['repeat'], fold=job['fold'], **row) for row in fit['inner_tuning'])
            numerical.extend(dict(model=model, repeat=job['repeat'], fold=job['fold'], **row) for row in fit['inner_fit_diagnostics'])
    pred = pd.concat([controls, pd.DataFrame(predictions)], ignore_index=True)
    assert len(pred) == 4 * 180 * 10 and not pred.duplicated(['model', 'repeat', 'exam_id']).any()
    assert pred.groupby(['model', 'exam_id']).size().eq(10).all()
    wide = pred.pivot(index=['repeat', 'exam_id', 'patient_id', 'outcome', 'fold'], columns='model', values='predicted_probability').reset_index()
    assert len(wide) == 1800 and not wide[MODELS].isna().any().any()
    assert wide.groupby(['repeat', 'patient_id']).fold.nunique().eq(1).all()
    coefficient = pd.DataFrame(coefs)
    static_coef = coefficient[coefficient.role.eq('static_EN')]
    assert len(static_coef) == 2 * 50 * 101
    frequency = static_coef.groupby(['model', 'feature_name'], sort=True).agg(
        selected_outer_folds=('selected', 'sum'), evaluated_outer_folds=('selected', 'size'),
        mean_coefficient=('coefficient', 'mean'), median_coefficient=('coefficient', 'median'),
        positive_coefficient_folds=('coefficient', lambda v:int((v > 0).sum())),
        negative_coefficient_folds=('coefficient', lambda v:int((v < 0).sum()))).reset_index()
    assert frequency.evaluated_outer_folds.eq(50).all() and len(frequency) == 202
    frequency['selection_frequency'] = frequency.selected_outer_folds / frequency.evaluated_outer_folds
    frequency = frequency.sort_values(['model', 'selection_frequency', 'feature_name'], ascending=[True, False, True])
    csv('02_OOF_PREDICTIONS_LOCAL.csv', pred)
    csv('03_OUTER_SELECTED_PARAMETERS.csv', choices)
    csv('04_ALL_COEFFICIENTS.csv', coefficient)
    csv('05_STATIC_SELECTION_FREQUENCIES.csv', frequency)
    csv('06_INNER_TUNING.csv', tuning)
    csv('07_INNER_NUMERICAL_AUDIT.csv', numerical)
    save(OUT / 'FIT_CHECKS.json', dict(all100_outer_EN_fits_converged=True,
        all9000_inner_EN_fits_converged=len(numerical) == 9000,
        max_inner_KKT=float(pd.DataFrame(numerical).KKT_residual.max()),
        max_outer_KKT=float(pd.DataFrame(choices).outer_fit_KKT.max()),
        fixed_base_features_never_L1_selected=True, all101_static_fields_frequency_reported=True,
        no_full_data_selection_or_scaling=True, all_model_case_patient_label_fold_pairs_identical=True))
    return pred, wide, pd.DataFrame(choices), frequency


def metrics(pred, wide, choices, frequency, protocol):
    result = evaluate_frozen_scores(wide, MODELS, iterations=B, seed=SEED)
    ci = np.quantile(result['draws'], [.025, .975], axis=0)
    performance = []
    for i, model in enumerate(MODELS):
        p, y = wide[model].to_numpy(), wide.outcome.to_numpy(int)
        p = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
        performance.append(dict(model=model, n_exams=180, n_unique_patients=174, repeated_patient_groups=6,
            pregnancy_n=117, non_pregnancy_n=63, AUC=float(result['points'][i]), AUC_CI_low=float(ci[0, i]),
            AUC_CI_high=float(ci[1, i]), Brier=float(np.mean((p - y) ** 2)),
            logloss=float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p))),
            scope='exploratory supervised feature-selection analysis; fixed OOF conditional interval'))
    perf = pd.DataFrame(performance)
    contrasts = []
    for augmented, baseline in protocol['main_comparisons']:
        i, j = MODELS.index(augmented), MODELS.index(baseline)
        delta = result['draws'][:, i] - result['draws'][:, j]
        low, high = np.quantile(delta, [.025, .975])
        contrasts.append(dict(augmented=augmented, baseline=baseline, n_exams=180, n_unique_patients=174,
            Delta_AUC=float(result['points'][i] - result['points'][j]), CI_low=float(low), CI_high=float(high),
            crosses_zero=bool(low <= 0 <= high), Delta_Brier=float(perf.Brier.iloc[i] - perf.Brier.iloc[j]),
            Delta_logloss=float(perf.logloss.iloc[i] - perf.logloss.iloc[j]),
            interpretation='exploratory marginal95% fixed OOF; no full training-refit or independent validation claim'))
    contrast = pd.DataFrame(contrasts)
    csv('08_MODEL_PERFORMANCE.csv', perf)
    csv('09_PAIRED_INCREMENTS.csv', contrast)
    csv('10_EACH_OUTER_FOLD.csv', result['folds'])
    csv('11_EACH_REPEAT.csv', result['repeats'])
    csv('12_BOOTSTRAP_DRAWS.csv', pd.DataFrame(result['draws'], columns=MODELS))
    counts = choices.groupby('model').n_static_selected.agg(['count', 'min', 'median', 'max']).reset_index()
    counts['zero_selected_folds'] = [int(choices.loc[choices.model.eq(m), 'n_static_selected'].eq(0).sum()) for m in counts.model]
    csv('13_SELECTION_COUNTS.csv', counts)
    assert all(sha(Path(path)) == value for path, value in protocol['source_hashes'].items())
    instructions = pd.DataFrame([
        ('定位', 'exploratory supervised feature-selection analysis；不是独立验证'),
        ('唯一新方法', '只对静态101项施加ElasticNet的L1+L2；临床6/动态5固定保留，仅L2'),
        ('参数', 'alpha0.1/0.5/0.9；lambda为原C网格倒数。所有填补/标准化/选择在训练折内'),
        ('选择频率', '两场景各50个outer训练模型，全部101项均报告；重叠训练集下仅描述频次'),
        ('零选择', '允许某折静态全部系数为0；不强制补足固定数量'),
        ('区间', '10000次患者聚类固定OOF条件区间，不含完整重拟合/调参/分折不确定性'),
        ('停止规则', '本次完成后停止静态方法搜索，不再因结果换LASSO/RF/SVM/PCA/binWidth'),
        ('禁止二次拟合', '不按入选频率选出最终子集，再在这180例重训并报告确认性AUC'),
    ], columns=['项目', '说明'])
    workbook = OUT / '静态ElasticNet_最后补充探索.xlsx'
    with pd.ExcelWriter(workbook, engine='openpyxl') as writer:
        for sheet, frame in [('使用说明', instructions), ('四组性能', perf), ('两组配对增量', contrast),
            ('101项入选频率', frequency), ('各折选择数量', choices), ('选择数量汇总', counts),
            ('各折系数', pd.read_csv(OUT / '04_ALL_COEFFICIENTS.csv')), ('每折性能', result['folds']),
            ('每轮性能', result['repeats'])]:
            frame.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            ws.freeze_panes, ws.auto_filter.ref = 'A2', ws.dimensions
        writer.sheets['使用说明'].column_dimensions['A'].width = 20
        writer.sheets['使用说明'].column_dimensions['B'].width = 115
    restored = pd.read_excel(workbook, sheet_name='四组性能')
    np.testing.assert_allclose(restored[['AUC', 'Brier', 'logloss']], perf[['AUC', 'Brier', 'logloss']], rtol=1e-12)
    def md(frame):
        return '\n'.join(['|' + '|'.join(frame.columns) + '|', '|' + '|'.join(['---'] * len(frame.columns)) + '|'] +
                         ['|' + '|'.join(map(str, row)) + '|' for row in frame.itertuples(index=False, name=None)])
    report = f'''# 静态Elastic Net：最后一次补充探索

分析定位为**exploratory supervised feature-selection analysis**。同180次检查/174名患者，117妊娠/63非妊娠，6重复组；沿用原patient-grouped outer5/inner5、10轮。该队列已被多轮探索，不能称独立验证。

## 预先冻结的唯一方法

临床6项和动态5项始终进入模型，沿用原Ridge尺度的L2处理，不受L1清零。只有静态101项加入Elastic Net的L1+L2。sum-loss目标为：logloss_sum + λ/2||base||² + λ(1−α)/2||static||² + λα||static||1，截距不惩罚。λ=1/C，与原C网格对应；α={en.ALPHAS}，λ={en.LAMBDAS}。因此是分块Elastic Net，而不是对整张表做普通LASSO或把临床一起筛掉。

缺失填补和标准化均在每个inner训练集拟合，分别比较18组α/λ，以5个inner验证集平均log-loss选择；确切并列优先更强λ再更大α。outer test不参与选择。outer训练全体再按选定参数拟合，只预测该outer test。无PCA、无binWidth/特征定义变化、无全数据相关性筛选。允许零静态入选。

采用非负变量分解的同一凸EN目标，SciPy L-BFGS-B计算；解析梯度、零静态与Ridge对照、全部变量为静态的EN与sklearn合成数据对照通过。患者数据每个inner/outer拟合均检查KKT残差≤1e-5，此为数值求解精度，不是排除病例/特征的测量学门槛。两个临床/动态对照直接复用冻结OOF，病例与fold逐行完全配对。

## 性能与配对增量

{md(perf[['model', 'AUC', 'AUC_CI_low', 'AUC_CI_high', 'Brier', 'logloss']].round(4))}

{md(contrast[['augmented', 'baseline', 'Delta_AUC', 'CI_low', 'CI_high', 'crosses_zero', 'Delta_Brier', 'Delta_logloss']].round(4))}

所有AUC为fold内正负病例对加权后平均10轮。95%区间来自{B}次患者聚类配对固定OOF bootstrap，seed={SEED}；不是完整流程重拟合区间，未包含此前探索/多重比较的选择影响。Brier/log-loss为同样检查/重复轮次的平均。

## 选择稳定性

{md(counts)}

05_STATIC_SELECTION_FREQUENCIES.csv及Excel对两个场景分别保留全部101项的入选次数/50、系数方向和全部outer系数。训练集彼此重叠，这只是内部重采样的选择频率，不是独立复现率。没有按频率阈值生成最终特征子集，也未在全部180例重训最终模型。

## 停止规则已执行

本次无论正负，均为静态方法搜索的最后一次补充。结果完整保留，停止静态模型/筛选方法搜索：不再转LASSO、RF、SVM、改变PCA维数或binWidth。后续方向为测量学验证与未参与开发的患者评价。像素尺度、设备/增益和人工重复分割等测量限制仍在。

[Elastic Net逻辑回归定义](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)；[数值求解器](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)。
'''
    (OUT / 'REPORT.md').write_text(report, encoding='utf-8')
    save(OUT / 'CHECKS_FINAL.json', dict(source_files_unchanged=True, same180_exams174_patients_all4_models=True,
        all_inner_outer_groups_disjoint=True, all_preprocessing_training_only=True,
        alpha_lambda_inner_only=True, only_static_L1_selection=True,
        clinical6_dynamic5_forced_kept=True, zero_selected_static_allowed=True,
        selection_frequency_all101_each_scenario=True, no_final_subset_refit=True,
        controls_reused_exactly=True, workbook_numeric_readback_matches=True,
        static_search_stop_rule_completed=True, no_independent_validation_claim=True))
    save(OUT / 'STOP_RULE.json', dict(status='completed_static_search_closed', rule=protocol['STOP_RULE'],
        no_further_static_method_search=True, no_final_frequency_selected_refit=True,
        results_retained=True, next_work='measurement validation and new untouched patients'))
    print(perf[['model', 'AUC', 'AUC_CI_low', 'AUC_CI_high', 'Brier', 'logloss']].round(6).to_string(index=False))
    print(contrast[['augmented', 'baseline', 'Delta_AUC', 'CI_low', 'CI_high', 'crosses_zero']].round(6).to_string(index=False))
    print(counts.to_string(index=False))
    print('Static method search CLOSED under the user stop rule.')


def run(resume=False):
    frame, columns, jobs, protocol = resume_after_numerical_correction() if resume else prepare()
    pending = [job for job in jobs if not (OUT / 'jobs' / (job['key'] + '.json')).exists()]
    already = len(jobs) - len(pending)
    save(OUT / 'progress.json', dict(stage='fitting', completed=already, total=50))
    # First real outer task validates numerical convergence before queueing the rest;
    # no outer performance is inspected until all frozen tasks finish.
    if pending:
        fit_job(pending[0])
        already += 1
        save(OUT / 'progress.json', dict(stage='fitting', completed=already, total=50))
        print(f'Completed{already}/50 outer tasks; frozen block-EN solver converged.', flush=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fit_job, job) for job in pending[1:]]
        for completed, future in enumerate(as_completed(futures), already + 1):
            future.result()
            save(OUT / 'progress.json', dict(stage='fitting', completed=completed, total=50))
            if completed % 5 == 0:
                print(f'Completed{completed}/50 outer tasks (two EN models each)', flush=True)
    pred, wide, choices, frequency = collect(frame, columns, jobs, protocol)
    save(OUT / 'progress.json', dict(stage='paired_conditional_bootstrap'))
    metrics(pred, wide, choices, frequency, protocol)
    save(OUT / 'progress.json', dict(stage='complete', static_search_closed=True))


if __name__ == '__main__':
    import sys
    run(resume='--resume-numerical-correction' in sys.argv)
