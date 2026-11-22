"""Frozen clinical-baseline completion and sole MSE incremental re-evaluation."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
import sklearn

from analysis.pregnancy_prediction.code.clinical9_data_audit import (ROOT, OUT, BOOK, MASTER, OLD, LIMITED, HISTORICAL,
    CORE, EXTRA, MSE, sha, save, csv)
from analysis.pregnancy_prediction.code.clinical9_models import (GRID, FULL_MODELS, MSE_MODELS, specification, pipeline,
    preprocessing_state, reconstruct)
from analysis.pregnancy_prediction.code.frozen_foldwise_auc import evaluate_frozen_scores

SEED, B = 20261120, 10000
_DATA = None


def read_data():
    fields = ['exam_id', 'patient_id', 'outcome', 'model_identity_eligible', 'cohort', MSE] + CORE + EXTRA + ['uterine_position_clean']
    return pd.read_csv(OUT / '04_MASTER_CLINICAL9_LOCAL_311.csv', usecols=fields,
                       dtype={'exam_id': str, 'patient_id': str})


def fit_job(job):
    global _DATA
    if _DATA is None:
        _DATA = read_data()
    train, test = _DATA.iloc[job['train']], _DATA.iloc[job['test']]
    assert not set(train.patient_id) & set(test.patient_id)
    assignment = np.asarray(job['inner_assignment'])
    models = FULL_MODELS if job['cohort'] == 'Full306' else MSE_MODELS
    results = []
    with threadpool_limits(limits=1):
        for model in models:
            _, names = specification(model)
            if 'MSE' in model:
                assert train[MSE].notna().all() and test[MSE].notna().all()
            scores = {C: [] for C in GRID}
            inner_states = []
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                for fold in range(1, 6):
                    a, b = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
                    tr, va = train.iloc[a], train.iloc[b]
                    assert not set(tr.patient_id) & set(va.patient_id)
                    assert tr.outcome.nunique() == va.outcome.nunique() == 2
                    first_state = None
                    for C in GRID:
                        fitted = pipeline(model, C).fit(tr[names], tr.outcome.to_numpy(int))
                        state, feature_names = preprocessing_state(fitted, tr, model)
                        if first_state is None:
                            first_state = state
                        else:
                            assert state == first_state
                        p = fitted.predict_proba(va[names])[:, 1]
                        # Same scorer, folds, equal-fold averaging and C tie order as historical GridSearchCV.
                        scores[C].append(float(log_loss(va.outcome, p, labels=[0, 1])))
                    inner_states.append(dict(inner_fold=fold, train_indices=a.tolist(), validation_indices=b.tolist(),
                        patient_overlap=0, preprocessing=first_state))
                best = min(GRID, key=lambda C: np.mean(scores[C]))
                fitted = pipeline(model, best).fit(train[names], train.outcome.to_numpy(int))
                state, feature_names = preprocessing_state(fitted, train, model)
                p = fitted.predict_proba(test[names])[:, 1]
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise RuntimeError('Frozen Ridge convergence failure; no substitution/grid alteration')
            coefficients = fitted.named_steps['model'].coef_[0]
            intercept = float(fitted.named_steps['model'].intercept_[0])
            np.testing.assert_allclose(p, expit(reconstruct(test, state, feature_names) @ coefficients + intercept), atol=1e-12)
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            results.append(dict(model=model, selected_C=best, predictions=p.tolist(), feature_names=feature_names,
                coefficients=coefficients.tolist(), intercept=intercept, preprocessing=state,
                inner_preprocessing=inner_states, classes=[0, 1], converged=True,
                inner_scores=[dict(C=C, mean_logloss=float(np.mean(scores[C])), fold_logloss=scores[C]) for C in GRID]))
    save(OUT / 'jobs' / (job['key'] + '.json'), dict(**job, results=results))
    return job['key']


def prepare():
    assert not (OUT / 'PROTOCOL_BEFORE_FIT.json').exists()
    audit = json.loads((OUT / 'DATA_AUDIT_BEFORE_FIT.json').read_text(encoding='utf-8'))
    assert all(sha(Path(p)) == h for p, h in audit['source_hashes'].items())
    d = read_data()
    original = pd.read_csv(MASTER, usecols=['exam_id', 'patient_id', 'outcome', MSE] + CORE,
                           dtype={'exam_id': str, 'patient_id': str})
    assert d[['exam_id', 'patient_id']].equals(original[['exam_id', 'patient_id']])
    np.testing.assert_allclose(d[CORE + [MSE, 'outcome']], original[CORE + [MSE, 'outcome']], equal_nan=True)
    assert not np.isinf(d[CORE + EXTRA + [MSE]].to_numpy(float)).any()
    eligible = d.model_identity_eligible.eq(True) & d.outcome.isin([0, 1])
    masks = {'Full306': eligible, 'MSE185': eligible & d[MSE].notna()}
    assert masks['Full306'].sum() == 306 and masks['MSE185'].sum() == 185
    outer = pd.read_csv(OLD / '02a_GLOBAL_PATIENT_FOLDS.csv', dtype={'exam_id': str, 'patient_id': str})
    inner = pd.read_csv(OLD / '02b_GLOBAL_INNER_VALIDATION_FOLDS.csv', dtype={'exam_id': str, 'patient_id': str})
    jobs, balances = [], []
    for cohort, mask in masks.items():
        sub = d[mask]
        for repeat in range(10):
            mapping = outer[outer['repeat'].eq(repeat)].set_index('exam_id')
            assert sub.exam_id.map(mapping.patient_id).eq(sub.patient_id).all()
            folds = sub.exam_id.map(mapping.fold)
            assert folds.notna().all()
            for fold in range(1, 6):
                tr, te = sub.index[folds.ne(fold)].tolist(), sub.index[folds.eq(fold)].tolist()
                train, test = d.iloc[tr], d.iloc[te]
                assert not set(train.patient_id) & set(test.patient_id)
                imap = inner[inner['repeat'].eq(repeat) & inner.outer_fold.eq(fold)].set_index('exam_id')
                ia = train.exam_id.map(imap.inner_fold)
                assert ia.notna().all() and train.exam_id.map(imap.patient_id).eq(train.patient_id).all()
                assert train.outcome.nunique() == test.outcome.nunique() == 2
                for f in range(1, 6):
                    a, b = train.loc[ia.ne(f)], train.loc[ia.eq(f)]
                    assert not set(a.patient_id) & set(b.patient_id)
                    assert a.outcome.nunique() == b.outcome.nunique() == 2
                jobs.append(dict(key=f'{cohort}_r{repeat:02d}_f{fold}', cohort=cohort, repeat=repeat,
                                 fold=fold, train=tr, test=te, inner_assignment=ia.astype(int).tolist()))
                balances.append(dict(cohort=cohort, repeat=repeat, fold=fold, train_n=len(train), test_n=len(test),
                    train_patients=train.patient_id.nunique(), test_patients=test.patient_id.nunique(),
                    test_repeated_patient_groups=int(test.patient_id.value_counts().gt(1).sum()),
                    pregnancy_n=int(test.outcome.eq(1).sum()), non_pregnancy_n=int(test.outcome.eq(0).sum()), patient_overlap=0))
    csv('11_OUTER_FOLD_BALANCE_BEFORE_FIT.csv', balances)
    paths = [BOOK, MASTER, OUT / '04_MASTER_CLINICAL9_LOCAL_311.csv', OUT / 'DATA_AUDIT_BEFORE_FIT.json',
        OLD / '02a_GLOBAL_PATIENT_FOLDS.csv', OLD / '02b_GLOBAL_INNER_VALIDATION_FOLDS.csv',
        OLD / '03_OOF_PREDICTIONS.csv', LIMITED / '04_OOF_PREDICTIONS_LOCAL.csv',
        HISTORICAL / '01_MODULE_PAIRED_RESULTS.csv', ROOT / 'outputs/static_elasticnet_final_exploration_20261122/STOP_RULE.json']
    protocol = dict(version='Clinical9_frozen_MSE_only_20261122', frozen_UTC=datetime.now(timezone.utc).isoformat(),
        outcome='clinical pregnancy1; nonpregnancy0; original source model handling of dash unchanged and flagged',
        data_audit='DATA_AUDIT_REPORT_BEFORE_FIT.md and01–11 tables exist before any fit',
        clinical6=CORE, clinical9=CORE + EXTRA, numeric9_linear_only=True, clinical9_predictor_df=9,
        pattern=audit['pattern_encoding'], days=audit['timing'], volume=audit['volume'],
        cohorts={'Full306': {'exams':306, 'patients':288}, 'MSE185': {'exams':185, 'patients':179}},
        models={'Full306':FULL_MODELS, 'MSE185':MSE_MODELS}, MSE_definition_unchanged=MSE,
        main_comparison='Clinical9_MSE minus Clinical9 on same185 cases/folds',
        baseline_comparison='Clinical9 minus Clinical6 on same306 cases/folds',
        historical_comparison='reuse unchanged Clinical6 and Clinical6_MSE OOF on original185',
        position_sensitivity='Clinical9_MSE_Position minus Clinical9_Position; latter is required paired adjustment control, not an extra candidate search',
        position_encoding='nominal, fixed anterior reference, posterior/horizontal2 indicator df; training mode if missing',
        position_secondary_comparison='difference of MSE increments with vs without position, paired fixed OOF',
        outer5_inner5_repeats10=True, exact_old_global_fold_maps=True, no_seed_search=True,
        algorithm='Ridge Logistic L2 lbfgs max_iter10000 tol1e-4, original settings',
        C_grid=GRID, selection='mean inner5 logloss minimum; exact tie first ascending C',
        preprocessing='numeric training median/zscore; stage and pureC training mode/unscaled; MSE observed only/training zscore; nominal position training mode/onehot',
        inner_preprocessing_audit='save state each inner split and verify every C uses same training-only state',
        no_complete_case_deletion=True, no_blood_predictor=True, no_static_or_other_dynamic=True,
        evaluation='AUC within-fold positive-negative-pair weighted per repeat then mean10; same frozen evaluator',
        bootstrap={'n':B, 'seed':SEED, 'unit':'patient clusters', 'scope':'paired fixed OOF conditional95%, no full workflow refit'},
        scope='further internal exploratory development on already explored cases, not independent validation or causal proof',
        stop_rule='No Clinical9 recoding/variable removal/seed or evaluation changes after results; no new imaging or static searches.',
        source_hashes={str(p): sha(p) for p in paths},
        script_hashes={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('clinical9_models.py'),
            Path(__file__).with_name('clinical9_data_audit.py'), Path(__file__).with_name('frozen_foldwise_auc.py')]},
        sklearn_version=sklearn.__version__)
    (OUT / 'jobs').mkdir()
    save(OUT / 'FIT_JOBS_LOCAL.json', jobs)
    save(OUT / 'PROTOCOL_BEFORE_FIT.json', protocol)
    save(OUT / 'STOP_RULE.json', dict(status='frozen_analysis_in_progress', rule=protocol['stop_rule']))
    return d, jobs, protocol


def collect(d, jobs, protocol):
    rows, tuning, choices, coefficients = [], [], [], []
    for job in jobs:
        result = json.loads((OUT / 'jobs' / (job['key'] + '.json')).read_text(encoding='utf-8'))
        for fit in result['results']:
            for index, p in zip(job['test'], fit['predictions']):
                row = d.iloc[index]
                rows.append(dict(cohort=job['cohort'], model=fit['model'], repeat=job['repeat'], fold=job['fold'],
                    exam_id=row.exam_id, patient_id=row.patient_id, outcome=int(row.outcome), predicted_probability=p))
            choices.append(dict(cohort=job['cohort'], model=fit['model'], repeat=job['repeat'], fold=job['fold'],
                                selected_C=fit['selected_C'], predictor_df=len(fit['feature_names'])))
            tuning.extend(dict(cohort=job['cohort'], model=fit['model'], repeat=job['repeat'], fold=job['fold'],
                C=s['C'], mean_logloss=s['mean_logloss']) for s in fit['inner_scores'])
            coefficients.extend(dict(cohort=job['cohort'], model=fit['model'], repeat=job['repeat'], fold=job['fold'],
                feature=f, coefficient=c, intercept=fit['intercept']) for f, c in zip(fit['feature_names'], fit['coefficients']))
    pred = pd.DataFrame(rows)
    assert not pred.duplicated(['cohort', 'model', 'repeat', 'exam_id']).any()
    assert pred.groupby(['cohort', 'model', 'exam_id']).size().eq(10).all()
    # Verify unchanged Clinical6 on all306 against the archived fixed linear Ridge procedure.
    prior = pd.read_csv(LIMITED / '04_OOF_PREDICTIONS_LOCAL.csv', dtype={'exam_id':str, 'patient_id':str})
    prior = prior[prior.cohort.eq('full_clinical') & prior.method.eq('ridge_linear')]
    current = pred[pred.cohort.eq('Full306') & pred.model.eq('Clinical6')]
    paired = current.merge(prior, on=['exam_id', 'patient_id', 'outcome', 'repeat', 'fold'], validate='one_to_one', suffixes=('_new', '_old'))
    assert len(paired) == 3060
    np.testing.assert_allclose(paired.predicted_probability_new, paired.predicted_probability_old, atol=1e-12, rtol=1e-12)
    historical = pd.read_csv(OLD / '03_OOF_PREDICTIONS.csv', dtype={'exam_id':str, 'patient_id':str})
    historical = historical[historical.analysis_set.eq('MSE__clinical_fold_imputation')].copy()
    historical['cohort'] = 'MSE185'
    historical['model'] = historical.role.map({'baseline':'Clinical6_historical', 'augmented':'Clinical6_MSE_historical'})
    assert historical.model.notna().all()
    pred = pd.concat([pred, historical[pred.columns]], ignore_index=True)
    csv('12_OOF_PREDICTIONS_LOCAL.csv', pred)
    csv('13_C_SELECTION.csv', choices)
    csv('14_INNER_TUNING.csv', tuning)
    csv('15_COEFFICIENTS.csv', coefficients)
    save(OUT / 'FIT_CHECKS.json', dict(all300_outer_and9000_inner_fits_converged=True,
        Clinical6_full306_historical_prediction_max_error=float(abs(paired.predicted_probability_new - paired.predicted_probability_old).max()),
        old_clinical6_control_reproduced=True, original_MSE_models_reused_unchanged=True,
        no_patient_leakage=True, clinical9_model_definition_unchanged=True))
    return pred


def metrics(pred, protocol):
    performance, contrasts, fold_rows, repeat_rows = [], [], [], []
    for cohort in ['Full306', 'MSE185']:
        q = pred[pred.cohort.eq(cohort)]
        models = FULL_MODELS if cohort == 'Full306' else ['Clinical6_historical', 'Clinical6_MSE_historical'] + MSE_MODELS
        wide = q.pivot(index=['repeat', 'exam_id', 'patient_id', 'outcome', 'fold'], columns='model', values='predicted_probability').reset_index()
        assert wide[models].notna().all().all() and wide.groupby(['repeat', 'patient_id']).fold.nunique().eq(1).all()
        result = evaluate_frozen_scores(wide, models, iterations=B, seed=SEED)
        ci = np.quantile(result['draws'], [.025, .975], axis=0)
        for j, model in enumerate(models):
            p, y = wide[model].to_numpy(), wide.outcome.to_numpy()
            p = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
            performance.append(dict(cohort=cohort, model=model, n_exams=result['n_exams'],
                n_unique_patients=result['n_unique_patients'], repeated_patient_groups=result['repeated_patient_groups'],
                pregnancy_n=result['pregnancy_n'], non_pregnancy_n=result['non_pregnancy_n'],
                AUC=float(result['points'][j]), CI_low=float(ci[0,j]), CI_high=float(ci[1,j]),
                Brier=float(np.mean((p-y)**2)), logloss=float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))))
        pairs = [('Clinical9','Clinical6','baseline_completion')] if cohort == 'Full306' else [
            ('Clinical6_MSE_historical','Clinical6_historical','historical_MSE'),
            ('Clinical9_MSE','Clinical9','primary_MSE'),
            ('Clinical9_MSE_Position','Clinical9_Position','position_adjusted_MSE'),
            ('Clinical9_MSE_Position','Clinical9_MSE','adding_position_to_MSE_model')]
        for augmented, baseline, role in pairs:
            i, j = models.index(augmented), models.index(baseline)
            delta = result['draws'][:, i] - result['draws'][:, j]
            lo, hi = np.quantile(delta, [.025,.975])
            current = pd.DataFrame(performance).set_index(['cohort','model'])
            contrasts.append(dict(cohort=cohort, role=role, augmented=augmented, baseline=baseline,
                n_exams=result['n_exams'], n_unique_patients=result['n_unique_patients'],
                Delta_AUC=float(result['points'][i]-result['points'][j]), CI_low=float(lo), CI_high=float(hi),
                crosses_zero=bool(lo <= 0 <= hi),
                Delta_Brier=float(current.loc[(cohort,augmented),'Brier']-current.loc[(cohort,baseline),'Brier']),
                Delta_logloss=float(current.loc[(cohort,augmented),'logloss']-current.loc[(cohort,baseline),'logloss'])))
        if cohort == 'MSE185':
            def increment(values, augmented, baseline):
                return values[..., models.index(augmented)] - values[..., models.index(baseline)]
            draws = increment(result['draws'],'Clinical9_MSE_Position','Clinical9_Position') - increment(result['draws'],'Clinical9_MSE','Clinical9')
            value = increment(result['points'],'Clinical9_MSE_Position','Clinical9_Position') - increment(result['points'],'Clinical9_MSE','Clinical9')
            lo, hi = np.quantile(draws,[.025,.975])
            csv('19_POSITION_EFFECT_ON_MSE_INCREMENT.csv', [dict(adjusted_minus_unadjusted_MSE_increment=float(value),
                CI_low=float(lo), CI_high=float(hi), crosses_zero=bool(lo <= 0 <= hi), scope='fixed OOF paired conditional')])
        csv('bootstrap_' + cohort + '.csv', pd.DataFrame(result['draws'], columns=models))
        fold_rows.append(result['folds'].assign(cohort=cohort))
        repeat_rows.append(result['repeats'].assign(cohort=cohort))
    csv('16_MODEL_PERFORMANCE.csv', performance)
    csv('17_PAIRED_COMPARISONS.csv', contrasts)
    csv('18a_EACH_OUTER_FOLD.csv', pd.concat(fold_rows, ignore_index=True))
    csv('18b_EACH_REPEAT.csv', pd.concat(repeat_rows, ignore_index=True))
    assert all(sha(Path(p)) == h for p, h in protocol['source_hashes'].items())
    save(OUT / 'CHECKS_FINAL.json', dict(sources_historical_and_static_stop_files_unchanged=True,
        all300_outer_Ridge_fits_converged=True, all9000_inner_Ridge_fits_converged=True,
        same_cases_and_folds_within_each_comparison=True, P1_clinical_pregnancy_verified=True,
        all_patient_group_checks_pass=True, inner_training_only_preprocessing_verified=True,
        Clinical6_history_reproduced=True, all_missing_retained_for_training_only_imputation=True,
        no_model_variable_or_encoding_search=True, clinical9_definition_frozen=True,
        fixed_OOF_bootstrap_only=True, no_independent_or_causal_validation_claim=True))
    save(OUT / 'STOP_RULE.json', dict(status='completed_clinical9_and_MSE_analysis_closed', rule=protocol['stop_rule'],
        next_work='measurement validation and untouched new patient evaluation', no_new_search_authorized=True))
    print(pd.DataFrame(performance).round(6).to_string(index=False), flush=True)
    print(pd.DataFrame(contrasts).round(6).to_string(index=False), flush=True)


def run():
    d, jobs, protocol = prepare()
    save(OUT / 'progress.json', dict(stage='fitting', completed=0, total=100))
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fit_job, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            save(OUT / 'progress.json', dict(stage='fitting', completed=completed, total=100))
            if completed % 10 == 0:
                print(f'Completed{completed}/100 frozen outer tasks', flush=True)
    pred = collect(d, jobs, protocol)
    save(OUT / 'progress.json', dict(stage='paired_fixed_OOF_bootstrap'))
    metrics(pred, protocol)
    save(OUT / 'progress.json', dict(stage='metrics_complete_report_pending'))


if __name__ == '__main__':
    run()
