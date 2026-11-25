"""First static/PCA versus dynamic paired exploratory nested-CV test, common180."""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits
import sklearn

from analysis.pregnancy_prediction.code.static_dynamic_pca_models import (
    CLINICAL, NUMERIC, STAGE, DYNAMIC, MODELS, C_GRID, K_GRID,
    has_static, has_dynamic, pipeline, grid, training_isolation_check)
from analysis.pregnancy_prediction.code.frozen_foldwise_auc import evaluate_frozen_scores

ROOT = Path(os.environ.get('DATA_ROOT', 'data'))
OLD = ROOT / 'outputs/auc_first_pass_extended6_globalfolds_20261120'
STATIC = ROOT / 'outputs/static_endometrium_radiomics_v3_manual2_20261120'
MASTER = ROOT / 'outputs/extended6_increment_protocol_20261120/MASTER_CLINICAL_SOURCE_ALIGNED_311.csv'
OUT = ROOT / 'outputs/static_dynamic_pca_nested_20261121'
META = ['case_id', 'exam_id', 'patient_id', 'outcome', 'model_identity_eligible']
BOOTSTRAP_N, BOOTSTRAP_SEED = 10000, 20261121
_DATA = None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig')


def data():
    names = pd.read_csv(STATIC / 'STATIC_FEATURE_INVENTORY.csv', usecols=['feature_name']).feature_name.tolist()
    assert len(names) == 101 and len(set(names)) == 101
    clinical = pd.read_csv(MASTER, usecols=META + CLINICAL + DYNAMIC,
                           dtype={'exam_id': str, 'patient_id': str})
    static = pd.read_csv(STATIC / 'STATIC_FEATURES_JOINT180.csv',
        usecols=['exam_id', 'patient_id'] + names, dtype={'exam_id': str, 'patient_id': str})
    merged = clinical.merge(static, on='exam_id', how='inner', suffixes=('', '_static'), validate='one_to_one')
    assert merged.patient_id.eq(merged.patient_id_static).all()
    merged = merged.drop(columns='patient_id_static').reset_index(drop=True)
    assert len(merged) == 180 and merged.patient_id.nunique() == 174
    assert merged.model_identity_eligible.eq(True).all() and set(merged.outcome) == {0, 1}
    assert not merged[CLINICAL + DYNAMIC + names].isna().any().any()
    assert np.isfinite(merged[CLINICAL + DYNAMIC + names].to_numpy(float)).all()
    assert set(merged[STAGE].unique()) <= {0, 1}
    return merged, names


def fit_job(job):
    global _DATA
    if _DATA is None:
        _DATA = data()
    frame, static_names = _DATA
    tr, te = frame.iloc[job['train']], frame.iloc[job['test']]
    assert not set(tr.patient_id) & set(te.patient_id)
    assignment = np.asarray(job['inner_assignment'], int)
    splits, audit = [], []
    for fold in range(1, 6):
        a, b = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
        assert not set(tr.iloc[a].patient_id) & set(tr.iloc[b].patient_id)
        assert tr.iloc[a].outcome.nunique() == tr.iloc[b].outcome.nunique() == 2
        splits.append((a, b))
        audit.append(dict(inner_fold=fold, train_n=len(a), validation_n=len(b), patient_overlap=0))
    outputs = []
    with threadpool_limits(limits=1):
        for name in MODELS:
            columns = NUMERIC + [STAGE] + (DYNAMIC if has_dynamic(name) else []) + (static_names if has_static(name) else [])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                search = GridSearchCV(pipeline(name, static_names), grid(name), scoring='neg_log_loss',
                    cv=splits, n_jobs=1, refit=True, error_score='raise', return_train_score=False)
                search.fit(tr[columns], tr.outcome.to_numpy(int))
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise RuntimeError(f'{name}: convergence warning; no automatic solver/grid substitution')
            fitted = search.best_estimator_
            model, preprocess = fitted.named_steps['model'], fitted.named_steps['preprocess']
            np.testing.assert_array_equal(model.classes_, [0, 1])
            probability = fitted.predict_proba(te[columns])[:, 1]
            z = preprocess.transform(te[columns])
            np.testing.assert_allclose(probability, expit(z @ model.coef_[0] + model.intercept_[0]), atol=1e-12)
            clinical = preprocess.named_transformers_['clinical_numeric']
            imputer, scale = clinical.named_steps['imputer'], clinical.named_steps['scale']
            np.testing.assert_allclose(imputer.statistics_, np.nanmedian(tr[NUMERIC], axis=0))
            filled = imputer.transform(tr[NUMERIC])
            np.testing.assert_allclose(scale.mean_, filled.mean(axis=0))
            parameters = dict(clinical_medians=imputer.statistics_.tolist(), clinical_means=scale.mean_.tolist(),
                clinical_scales=scale.scale_.tolist(), stage_mode=preprocess.named_transformers_['stage'].statistics_.tolist())
            transformed_names = NUMERIC + [STAGE]
            k = 0
            if has_dynamic(name):
                scaler = preprocess.named_transformers_['imaging']
                np.testing.assert_allclose(scaler.mean_, tr[DYNAMIC].mean().to_numpy())
                parameters['dynamic_means'], parameters['dynamic_scales'] = scaler.mean_.tolist(), scaler.scale_.tolist()
                transformed_names += DYNAMIC
            if has_static(name):
                block = preprocess.named_transformers_['static']
                input_scale, pca, score_scale = (block.named_steps[x] for x in ['input_scale', 'pca', 'score_scale'])
                np.testing.assert_allclose(input_scale.mean_, tr[static_names].mean().to_numpy())
                pca_input = input_scale.transform(tr[static_names])
                assert pca.n_samples_ == len(tr)
                np.testing.assert_allclose(pca.mean_, pca_input.mean(axis=0), atol=1e-12)
                train_scores = pca.transform(pca_input)
                np.testing.assert_allclose(score_scale.mean_, train_scores.mean(axis=0), atol=1e-12)
                np.testing.assert_allclose(score_scale.scale_, train_scores.std(axis=0), atol=1e-12)
                k = int(pca.n_components_)
                parameters['static'] = dict(feature_order=static_names,
                    input_means=input_scale.mean_.tolist(), input_scales=input_scale.scale_.tolist(),
                    pca_mean=pca.mean_.tolist(), components=pca.components_.tolist(),
                    explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
                    score_means=score_scale.mean_.tolist(), score_scales=score_scale.scale_.tolist(), n_fit_samples=len(tr))
                transformed_names += [f'static_PC{i + 1}' for i in range(k)]
            assert z.shape[1] == len(transformed_names)
            outputs.append(dict(model=name, predicted_probability=probability.tolist(), selected_C=float(search.best_params_['model__C']),
                selected_static_components=k, predictor_df=len(transformed_names),
                transformed_feature_order=transformed_names, coefficients=model.coef_[0].tolist(),
                intercept=float(model.intercept_[0]), classes=[0, 1], preprocessing=parameters,
                inner_scores=[dict(C=float(p['model__C']), static_components=int(p.get('preprocess__static__pca__n_components', 0)),
                    mean_negative_log_loss=float(v), SD=float(sd))
                    for p, v, sd in zip(search.cv_results_['params'], search.cv_results_['mean_test_score'], search.cv_results_['std_test_score'])],
                converged=True, transformations_training_only_checked=True))
    save(OUT / 'jobs' / (job['key'] + '.json'), dict(**job, results=outputs, inner_audit=audit))
    return job['key']


def prepare():
    OUT.mkdir(exist_ok=False)
    (OUT / 'jobs').mkdir()
    isolation = training_isolation_check()
    frame, names = data()
    outer_file, inner_file = OLD / '02a_GLOBAL_PATIENT_FOLDS.csv', OLD / '02b_GLOBAL_INNER_VALIDATION_FOLDS.csv'
    outer = pd.read_csv(outer_file, dtype={'exam_id': str, 'patient_id': str})
    inner = pd.read_csv(inner_file, dtype={'exam_id': str, 'patient_id': str})
    sources = [MASTER, STATIC / 'STATIC_FEATURES_JOINT180.csv', STATIC / 'STATIC_FEATURE_INVENTORY.csv',
               outer_file, inner_file, OLD / '03_OOF_PREDICTIONS.csv', OLD / '04_MODEL_PERFORMANCE.csv', OLD / '05_INCREMENTAL_PERFORMANCE.csv']
    hashes = {str(path): sha(path) for path in sources}
    jobs, assignments = [], []
    for repeat in range(10):
        mapping = outer[outer['repeat'].eq(repeat)].set_index('exam_id').fold
        assigned = frame.exam_id.map(mapping)
        assert assigned.notna().all()
        assert frame.assign(fold=assigned).groupby('patient_id').fold.nunique().eq(1).all()
        for fold in range(1, 6):
            train, test = frame.index[assigned.ne(fold)].tolist(), frame.index[assigned.eq(fold)].tolist()
            assert not set(frame.loc[train].patient_id) & set(frame.loc[test].patient_id)
            imap = inner[inner['repeat'].eq(repeat) & inner.outer_fold.eq(fold)].set_index('exam_id').inner_fold
            ia = frame.loc[train, 'exam_id'].map(imap)
            assert ia.notna().all()
            jobs.append(dict(key=f'r{repeat:02d}_f{fold}', repeat=repeat, fold=fold,
                             train=train, test=test, inner_assignment=ia.astype(int).tolist()))
        assignments.extend(dict(repeat=repeat, fold=int(assigned.iloc[i]), exam_id=row.exam_id,
            patient_id=row.patient_id, outcome=int(row.outcome)) for i, row in frame.iterrows())
    protocol = dict(version='static_dynamic_PCA_common180_v1_20261121',
        n_exams=180, n_unique_patients=174, repeated_patient_groups=int(frame.patient_id.value_counts().gt(1).sum()),
        pregnancy_n=int(frame.outcome.sum()), non_pregnancy_n=int(frame.outcome.eq(0).sum()),
        clinical=CLINICAL, dynamic=DYNAMIC, static_columns=names, raw_static_count=101,
        models=MODELS, C_grid=C_GRID, static_component_grid=K_GRID,
        classifier='Ridge logistic L2,lbfgs,max_iter10000; intercept unpenalized',
        preprocessing='train median/zscore clinical numeric; stage train mode/unscaled; dynamic train zscore; static train zscore->PCA full/no whiten->train PC zscore',
        tuning='mean5inner-fold neg_log_loss; choose C and static k only inside training',
        exact_ties='smaller k then smaller C; no outer-score method/feature selection',
        component_grid_reason='small predefined1/2/3/5 complexity comparison, not full-data90/95% variance counts',
        outer_folds=5, inner_folds=5, repeats=10, original_global_maps_reused=True,
        evaluation='within-repeat positive-negative pair-weighted fold AUC; average10 repeats',
        paired_contrasts=[['Clinical_Static', 'Clinical'], ['Clinical_Static_Dynamic', 'Clinical_Dynamic'],
                          ['Clinical_Dynamic', 'Clinical'], ['Clinical_Static_Dynamic', 'Clinical_Static'],
                          ['Clinical_Static_Dynamic', 'Clinical']],
        bootstrap=dict(iterations=BOOTSTRAP_N, seed=BOOTSTRAP_SEED, unit='patient',
            scope='same multiplicities for4 models/all repeats; fixed OOF percentile95%, not full refit CI'),
        clinical_age_nonlinearity='not changed; prior limited baseline-development variants not substituted',
        source_hashes=hashes, synthetic_isolation_checks=isolation, sklearn_version=sklearn.__version__,
        script_hashes={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('static_dynamic_pca_models.py'),
                                               Path(__file__).with_name('frozen_foldwise_auc.py')]},
        no_full_data_audit_PCA_reused=True, all101_static_fields_retained=True, no_old_R2_loaded=True,
        no_static_extraction_tracking_or_QC_threshold_change=True,
        scope='exploratory internal model development after prior outcome access; not independent validation',
        references=['https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html',
                    'https://scikit-learn.org/stable/common_pitfalls.html'])
    save(OUT / 'PROTOCOL_BEFORE_FIT.json', protocol)
    save(OUT / 'FIT_JOBS_LOCAL.json', jobs)
    csv('01_COHORT_LOCAL.csv', frame[META + CLINICAL + DYNAMIC])
    csv('02_FOLD_ASSIGNMENTS_LOCAL.csv', assignments)
    return frame, jobs, protocol


def collect(frame, jobs, protocol):
    predictions, tuning, choices, inner_audits, coefficients = [], [], [], [], []
    for job in jobs:
        saved = json.loads((OUT / 'jobs' / (job['key'] + '.json')).read_text(encoding='utf-8'))
        for result in saved['results']:
            model = result['model']
            choices.append(dict(repeat=job['repeat'], fold=job['fold'], model=model,
                C=result['selected_C'], static_components=result['selected_static_components'], predictor_df=result['predictor_df']))
            for index, probability in zip(job['test'], result['predicted_probability']):
                row = frame.iloc[index]
                predictions.append(dict(model=model, repeat=job['repeat'], fold=job['fold'], exam_id=row.exam_id,
                    patient_id=row.patient_id, outcome=int(row.outcome), predicted_probability=probability))
            tuning.extend(dict(repeat=job['repeat'], fold=job['fold'], model=model, **score) for score in result['inner_scores'])
            coefficients.extend(dict(repeat=job['repeat'], fold=job['fold'], model=model, feature=name,
                coefficient=value, intercept=result['intercept']) for name, value in zip(result['transformed_feature_order'], result['coefficients']))
        inner_audits.extend(dict(repeat=job['repeat'], outer_fold=job['fold'], **row) for row in saved['inner_audit'])
    pred = pd.DataFrame(predictions)
    assert len(pred) == 180 * 10 * 4
    assert not pred.duplicated(['model', 'repeat', 'exam_id']).any()
    assert pred.groupby(['model', 'exam_id']).size().eq(10).all()
    csv('03_OOF_PREDICTIONS_LOCAL.csv', pred)
    csv('04_INNER_TUNING.csv', tuning)
    csv('05_SELECTED_PARAMETERS.csv', choices)
    csv('06_INNER_SPLIT_AUDIT.csv', inner_audits)
    csv('07_COEFFICIENTS.csv', coefficients)
    # Independently reproduce both original control procedures on identical cases.
    old = pd.read_csv(OLD / '03_OOF_PREDICTIONS.csv', usecols=['analysis_set', 'role', 'exam_id', 'repeat', 'fold', 'predicted_probability'],
                      dtype={'exam_id': str})
    reference = old[old.analysis_set.eq('IMG_ACT__clinical_complete_case')]
    errors = {}
    for model, role in [('Clinical', 'baseline'), ('Clinical_Dynamic', 'augmented')]:
        paired = pred[pred.model.eq(model)].merge(reference[reference.role.eq(role)],
            on=['exam_id', 'repeat', 'fold'], validate='one_to_one', suffixes=('_new', '_old'))
        assert len(paired) == 1800
        np.testing.assert_allclose(paired.predicted_probability_new, paired.predicted_probability_old, atol=1e-12, rtol=1e-12)
        errors[model] = float((paired.predicted_probability_new - paired.predicted_probability_old).abs().max())
    assert all(sha(Path(path)) == value for path, value in protocol['source_hashes'].items())
    save(OUT / 'FIT_CHECKS.json', dict(all200_outer_fits_converged=True, all_inner_outer_patient_overlaps_zero=True,
        PCA_and_all_scalers_training_only_checked=True, all_prediction_columns_are_P1=True,
        both_original_controls_reproduced=True, control_max_probability_errors=errors,
        source_files_unchanged=True, no_full_data_PCA_used=True))
    return pred


def metrics(pred, protocol):
    wide = pred.pivot(index=['repeat', 'exam_id', 'patient_id', 'outcome', 'fold'], columns='model', values='predicted_probability').reset_index()
    result = evaluate_frozen_scores(wide, MODELS, iterations=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    auc_ci = np.quantile(result['draws'], [.025, .975], axis=0)
    performance = []
    for j, model in enumerate(MODELS):
        p, y = wide[model].to_numpy(), wide.outcome.to_numpy(int)
        clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
        performance.append(dict(model=model, n_exams=180, n_unique_patients=174, repeated_patient_groups=6,
            pregnancy_n=117, non_pregnancy_n=63, AUC=float(result['points'][j]),
            AUC_CI_low=float(auc_ci[0, j]), AUC_CI_high=float(auc_ci[1, j]),
            Brier=float(np.mean((p - y) ** 2)), logloss=float(np.mean(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped))),
            interval_scope='fixed OOF conditional; not full-refit uncertainty'))
    comparisons = []
    for augmented, baseline in protocol['paired_contrasts']:
        i, j = MODELS.index(augmented), MODELS.index(baseline)
        draws = result['draws'][:, i] - result['draws'][:, j]
        lo, hi = np.quantile(draws, [.025, .975])
        comparisons.append(dict(augmented=augmented, baseline=baseline, n_exams=180, n_unique_patients=174,
            Delta_AUC=float(result['points'][i] - result['points'][j]), CI_low=float(lo), CI_high=float(hi),
            crosses_zero=bool(lo <= 0 <= hi), interval_scope='exploratory marginal95%, no multiplicity correction; fixed OOF'))
    csv('08_MODEL_PERFORMANCE.csv', performance)
    csv('09_PAIRED_INCREMENTS.csv', comparisons)
    csv('10_EACH_OUTER_FOLD.csv', result['folds'])
    csv('11_EACH_REPEAT.csv', result['repeats'])
    csv('12_BOOTSTRAP_DRAWS.csv', pd.DataFrame(result['draws'], columns=MODELS))
    selected = pd.read_csv(OUT / '05_SELECTED_PARAMETERS.csv')
    frequency = selected.groupby(['model', 'static_components', 'C']).size().rename('outer_folds').reset_index()
    csv('13_PARAMETER_FREQUENCIES.csv', frequency)
    assert all(sha(Path(path)) == value for path, value in protocol['source_hashes'].items())
    save(OUT / 'CHECKS_FINAL.json', dict(all_source_files_unchanged=True, same180_exams174_patients_all4_models=True,
        same_frozen_global_outer_inner_splits=True, all200_outer_fits_converged=True,
        all_transformations_and_PCA_fitted_training_only=True, C_and_k_selected_inner_logloss_only=True,
        both_original_controls_reproduced_to1e_12=True, no_old_R2_or_new_feature_definition=True,
        no_patient_specific_parameters=True, no_full_data_PCA_or_outcome_feature_selection=True,
        paired_patient_bootstrap_all_models_all_repeats=True, bootstrap_iterations=BOOTSTRAP_N,
        metrics_engine_independently_checked_against_sklearn=True,
        no_outer_result_based_model_selection=True, no_independent_validation_claim=True))
    def md(d):
        return '\n'.join(['|' + '|'.join(d.columns) + '|', '|' + '|'.join(['---'] * len(d.columns)) + '|'] +
                         ['|' + '|'.join(map(str, row)) + '|' for row in d.itertuples(index=False, name=None)])
    perf, contrasts = pd.DataFrame(performance), pd.DataFrame(comparisons)
    report = f'''# 静态PCA与动态特征：共同180次检查配对测试

用户同意质量审查后的下一轮配对预测测试。当前为探索性内部开发，不是未接触数据上的独立验证。全部4模型使用同180次检查、174名患者（6个重复患者组，117妊娠/63非妊娠），同全局外5内5患者折及10轮重复。

## 拟合前固定的方案

临床沿用Extended6线性Ridge，保留年龄、实际胚胎数、囊胚、BMI、不孕总年限和厚度；不同时改为此前试过的年龄非线性模型。动态为MSE、CC、前后壁幅度和显著传播活动频率5项，定义不变。静态101项不按本数据结局筛选、不按相关阈值删除，训练折标准化后做PCA，再对训练成分标准化；候选k={K_GRID}，C={C_GRID}。k和C只按5个inner fold平均log-loss选择，确切并列优先更小k再更小C。候选k是小规模复杂度试验，不是直接采用全数据8/11成分的描述结果。

所有填补、标准化、PCA均在Pipeline中，每个inner训练折重新拟合；outer最终仅拟合其训练集。没有复用全数据质量审查的PCA。已用合成极端测试集值验证变换不会被测试数据改变，逐outer核对均值、样本数、PCA均值和成分尺度。临床及临床+动态两个对照的全部1800条预测各自逐条重现旧结果，容差1e-12。

## 结果

{md(perf[['model', 'n_exams', 'n_unique_patients', 'AUC', 'AUC_CI_low', 'AUC_CI_high', 'Brier', 'logloss']].round(4))}

预先列出的配对增量：

{md(contrasts[['augmented', 'baseline', 'Delta_AUC', 'CI_low', 'CI_high', 'crosses_zero']].round(4))}

## 评价与限制

AUC仅在同fold内比较病例，每轮按正负病例对数加权，然后平均10轮。区间为{BOOTSTRAP_N}次患者聚类配对bootstrap（seed={BOOTSTRAP_SEED}），同一患者各周期、四模型及10轮一起抽样；是固定OOF条件区间，不含完整重拟合、调参及重新分折的变动，也未校正此前多轮开发及本轮多组比较。没有按本轮最高外层AUC升级最终模型。

即使PCA减少参数，也不能保证180例足以稳定估计；所有101静态特征仍是探索候选，原生像素尺度、采集灰度/增益和分割重复性限制没有消失。显著传播活动频率也不是已确认的真实生理波次数。本轮没有重算图像、改ROI/特征或追踪，也未调整标签和结局政策。源表既有不确定标签的核查限制保留。

完整静态单独建模可用更多病例，但本轮只回答同批180例上的配对增量，不能与其他队列AUC混比。每outer选择的k/C、预处理均值、PCA载荷及系数均保留本地以便复核；单个PC不等于一个有独立生理含义的原始特征。源数据及旧结果哈希未变。

[PCA官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)；[训练折内处理](https://scikit-learn.org/stable/common_pitfalls.html)。
'''
    (OUT / 'REPORT.md').write_text(report, encoding='utf-8')
    workbook = OUT / '静态动态配对测试_180次检查.xlsx'
    with pd.ExcelWriter(workbook, engine='openpyxl') as writer:
        for name, table in [('四组性能', perf), ('配对增量', contrasts), ('参数选择', selected),
                            ('参数频次', frequency), ('每折性能', result['folds']), ('每轮性能', result['repeats'])]:
            table.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes, ws.auto_filter.ref = 'A2', ws.dimensions
    print(perf[['model', 'AUC', 'AUC_CI_low', 'AUC_CI_high', 'Brier', 'logloss']].round(6).to_string(index=False))
    print(contrasts[['augmented', 'baseline', 'Delta_AUC', 'CI_low', 'CI_high', 'crosses_zero']].round(6).to_string(index=False))


def run():
    frame, jobs, protocol = prepare()
    save(OUT / 'progress.json', dict(stage='fitting', completed=0, total=len(jobs)))
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fit_job, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            save(OUT / 'progress.json', dict(stage='fitting', completed=completed, total=len(jobs)))
            if completed % 5 == 0:
                print(f'Completed {completed}/{len(jobs)} outer tasks (four models each)', flush=True)
    pred = collect(frame, jobs, protocol)
    save(OUT / 'progress.json', dict(stage='paired_conditional_bootstrap'))
    metrics(pred, protocol)
    save(OUT / 'progress.json', dict(stage='complete', outer_tasks=len(jobs), model_outer_fits=200))


if __name__ == '__main__':
    run()
