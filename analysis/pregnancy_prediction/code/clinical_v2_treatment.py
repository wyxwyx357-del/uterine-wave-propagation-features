"""Clinical v2: freeze outcome-blind coding first, then one prescribed Ridge analysis."""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss, brier_score_loss
from threadpoolctl import threadpool_limits
import sklearn

from analysis.pregnancy_prediction.code.clinical_v2_models import (
    BASE, NUM9, MSE, PREP, ICSI, PREP_MAP, METHOD_MAP, PREP_LEVELS,
    GRID, MODELS, inputs, pipeline, preprocessing_state, reconstruct,
)
from analysis.pregnancy_prediction.code.frozen_foldwise_auc import evaluate_frozen_scores

ROOT = Path(os.environ.get('DATA_ROOT', 'data'))
OUT = ROOT / 'outputs/clinical_baseline_v2_treatment_context_20261123'
HISTORY = ROOT / 'outputs/clinical9_frozen_mse_20261122'
SUPPLEMENT = ROOT / 'outputs/clinical_information_supplement_audit_20261123'
CANDIDATES = SUPPLEMENT / '11_CANDIDATE_VALUES_AUDIT_ONLY_LOCAL.csv'
MASTER = HISTORY / '04_MASTER_CLINICAL9_LOCAL_311.csv'
B, SEED = 10000, 20261120
_DATA = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv(name, frame):
    pd.DataFrame(frame).to_csv(OUT / name, index=False, encoding='utf-8-sig')


def frozen_sources():
    files = [MASTER, CANDIDATES, HISTORY / 'FIT_JOBS_LOCAL.json',
             HISTORY / '12_OOF_PREDICTIONS_LOCAL.csv',
             ROOT / 'experiments/clinical9_models.py', ROOT / 'experiments/clinical9_data_audit.py',
             ROOT / 'experiments/frozen_foldwise_auc.py',
             SUPPLEMENT / 'TREATMENT_CONTEXT_EVIDENCE_ADDENDUM.md']
    for directory in ['clinical9_frozen_mse_20261122', 'clinical11_PI_VFI_mse_sensitivity_20261122',
                      'enhanced_clinical_v1_20261122', 'tiny_mlp_model_form_sensitivity_20261122',
                      'static_elasticnet_final_exploration_20261122', 'static_dynamic_pca_nested_20261121']:
        base = ROOT / 'outputs' / directory
        files += sorted(base.glob('*PERFORMANCE.csv')) + sorted(base.glob('*PAIRED_COMPARISONS.csv'))
        files += sorted(base.glob('STOP_RULE.json'))
    return {str(p): sha(p) for p in files}


def verify_sources(protocol):
    assert all(sha(p) == h for p, h in protocol['source_hashes'].items())
    assert all(sha(p) == h for p, h in protocol['script_hashes'].items())


def freeze():
    """This stage never opens outcomes, MSE values, predictions or performance tables."""
    assert not OUT.exists(), 'Do not overwrite a prior v2 analysis.'
    OUT.mkdir(parents=True)
    fields = ['exam_id', 'patient_id', 'in_existing306', 'in_MSE185', '治疗方式', '治疗方案', '周期类型',
              'fertilization_method_source', 'fertilization_method_known_before_scan',
              'fertilization_timing_flag', 'recorded_downregulated_HRT', 'source_provenance_json']
    d = pd.read_csv(CANDIDATES, usecols=fields, dtype={'exam_id': str, 'patient_id': str})
    assert len(d) == 358 and not d.exam_id.duplicated().any()
    assert set(d['治疗方式'].dropna()) <= set(PREP_MAP) | {'其它'}
    assert set(d.fertilization_method_known_before_scan.dropna()) <= set(METHOD_MAP)
    d[PREP] = d['治疗方式'].map(PREP_MAP)
    d[ICSI] = d.fertilization_method_known_before_scan.map(METHOD_MAP)
    stimulation = d[PREP].eq('fresh_cycle_stimulation')
    induction = d[PREP].eq('FET_ovulation_induction')
    assert d.loc[stimulation, '周期类型'].eq('新鲜周期').all()
    assert d.loc[induction, '周期类型'].eq('复苏周期').all()
    uncertain = d.fertilization_timing_flag.eq('fresh_same_day_or_undated_actual_fertilization_unconfirmed')
    assert uncertain.sum() == 3 and d.loc[uncertain, ICSI].isna().all()
    csv('01_TREATMENT_VALUES_OUTCOME_BLIND_LOCAL_358.csv', d)
    rows, summaries = [], []
    for cohort, mask in [('Source358', np.ones(len(d), bool)), ('Full306', d.in_existing306), ('MSE185', d.in_MSE185)]:
        part = d.loc[mask]
        for variable in ['治疗方式', PREP, 'fertilization_method_source', ICSI]:
            for level, n in part[variable].fillna('MISSING').value_counts().items():
                rows.append(dict(cohort=cohort, variable=variable, level=level, n=int(n)))
        for variable in [PREP, ICSI]:
            summaries.append(dict(cohort=cohort, variable=variable, n_exams=len(part),
                n_patients=part.patient_id.nunique(), repeated_patient_groups=int(part.patient_id.value_counts().gt(1).sum()),
                available=int(part[variable].notna().sum()), missing=int(part[variable].isna().sum()),
                missing_rate=float(part[variable].isna().mean())))
    csv('02_TREATMENT_DISTRIBUTIONS_BEFORE_OUTCOME.csv', rows)
    csv('03_TREATMENT_AVAILABILITY_BEFORE_OUTCOME.csv', summaries)
    protocol = dict(version='Clinical_baseline_v2_treatment_context_20261123',
        frozen_UTC=datetime.now(timezone.utc).isoformat(),
        coding_frozen_before_loading_outcomes_predictions_or_performance=True,
        clinical9_unchanged=BASE, preparation_raw_mapping=PREP_MAP,
        preparation_levels=PREP_LEVELS, preparation_reference='HRT_based',
        preparation_rationale='Merge HRT and downregulated HRT by shared HRT mechanism only; keep natural, FET induction and fresh stimulation distinct. Preserve all raw labels.',
        other_or_unevaluable_preparation='missing; never infer from subsequent outcomes',
        ICSI_mapping=METHOD_MAP, ICSI_definition='IVF-only vs any ICSI including mixed and Rescue; not pure IVF vs pure ICSI',
        ICSI_timing_uncertain_exam_ids=sorted(d.loc[uncertain, 'exam_id'].tolist()),
        time_limitation='Current treatment context precedes scan by care process; original field-entry timestamps unavailable. Three actual fertilization timings unconfirmed remain raw missing.',
        predictor_constructs=11, predictor_columns=13, predictor_columns_with_MSE=14,
        cohorts={'Full306': {'exams':306,'patients':288}, 'MSE185': {'exams':185,'patients':179}},
        models=MODELS, MSE=MSE, MSE_definition_unchanged=True,
        clinical9_controls='Reuse archived identical-cohort OOF predictions; never refit or overwrite history.',
        outer5_inner5_repeats10=True, exact_fold_source=str(HISTORY / 'FIT_JOBS_LOCAL.json'),
        ridge='L2 logistic, lbfgs, max_iter10000, tol1e-4', C_grid=GRID,
        tuning='Equal-inner-fold mean logloss minimum; exact tie chooses first ascending C, unchanged.',
        preprocessing='Frozen Clinical9: numeric training median/zscore; stage/pureC training mode unscaled; MSE observed-only/training zscore. New preparation training mode/fixed schema onehot, HRT reference, unscaled; new ICSI training mode/unscaled. No missingness indicator or row deletion.',
        categorical_schema='Clinically predefined levels fixed here; training mode learned separately for every inner and outer fit, no test-dependent level selection.',
        evaluation='Within-test-fold AUC, n_positive*n_negative weighted each repeat, average10; paired on identical exams/folds/repeats.',
        probability_scores='Exam-average Brier/logloss per repeat then average10; each outer fold additionally reported.',
        bootstrap={'n':B,'seed':SEED,'unit':'patient cluster','scope':'fixed OOF conditional95%, not complete workflow uncertainty'},
        primary_contrasts=['Full306: Clinical_v2 minus Clinical9', 'MSE185: Clinical_v2_MSE minus Clinical_v2'],
        secondary_contrasts=['MSE185: Clinical_v2 minus Clinical9', 'MSE185: Clinical_v2_MSE minus Clinical9_MSE',
                             'MSE185: change in MSE increment after treatment-context addition'],
        excluded='All other new variables, fresh/frozen as separate predictor, blood variables, static and other dynamic features, classifier/seed/encoding search.',
        interpretation='Already explored internal development cohort; no independent validation, causal or treatment-mechanism independence claims.',
        stop_rule='After one prespecified Clinical v2 run, freeze regardless of result; no single-variable versions, regrouping, variable/model/seed search.',
        source_hashes=frozen_sources(),
        script_hashes={str(p):sha(p) for p in [Path(__file__), Path(__file__).with_name('clinical_v2_models.py')]},
        sklearn_version=sklearn.__version__)
    save('PROTOCOL_BEFORE_OUTCOME_AND_FIT.json', protocol)
    save('STOP_RULE.json', dict(status='coding_frozen_no_fit_yet', rule=protocol['stop_rule']))
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


def read_data():
    fields = ['exam_id','patient_id','outcome','model_identity_eligible','cohort', MSE] + BASE
    d = pd.read_csv(MASTER, usecols=fields, dtype={'exam_id':str,'patient_id':str})
    new = pd.read_csv(OUT / '01_TREATMENT_VALUES_OUTCOME_BLIND_LOCAL_358.csv',
        usecols=['exam_id','patient_id',PREP,ICSI,'fertilization_timing_flag'], dtype={'exam_id':str,'patient_id':str})
    d = d.merge(new, on=['exam_id','patient_id'], how='left', validate='one_to_one', sort=False)
    assert len(d) == 311
    return d


def prepare():
    protocol = json.loads((OUT / 'PROTOCOL_BEFORE_OUTCOME_AND_FIT.json').read_text(encoding='utf-8'))
    verify_sources(protocol)
    d = read_data()
    jobs = json.loads((HISTORY / 'FIT_JOBS_LOCAL.json').read_text(encoding='utf-8'))
    eligible = d.model_identity_eligible.eq(True) & d.outcome.isin([0,1])
    masks = {'Full306':eligible, 'MSE185':eligible & d[MSE].notna()}
    counts, balances = [], []
    for cohort, mask in masks.items():
        sub = d[mask]
        assert len(sub) == protocol['cohorts'][cohort]['exams']
        assert sub.patient_id.nunique() == protocol['cohorts'][cohort]['patients']
        counts.append(dict(cohort=cohort,n_exams=len(sub),n_unique_patients=sub.patient_id.nunique(),
            repeated_patient_groups=int(sub.patient_id.value_counts().gt(1).sum()),
            pregnancy_n=int(sub.outcome.eq(1).sum()),non_pregnancy_n=int(sub.outcome.eq(0).sum()),
            preparation_missing=int(sub[PREP].isna().sum()),ICSI_missing=int(sub[ICSI].isna().sum())))
        for repeat in range(10):
            relevant = [j for j in jobs if j['cohort']==cohort and j['repeat']==repeat]
            assert sorted(i for j in relevant for i in j['test']) == sub.index.tolist()
    for job in jobs:
        train,test = d.iloc[job['train']],d.iloc[job['test']]
        assert not set(train.patient_id) & set(test.patient_id)
        assert train.outcome.nunique() == test.outcome.nunique() == 2
        assignment = np.asarray(job['inner_assignment'])
        for fold in range(1,6):
            tr,va=train.iloc[assignment!=fold],train.iloc[assignment==fold]
            assert not set(tr.patient_id) & set(va.patient_id)
            assert tr.outcome.nunique()==va.outcome.nunique()==2
        balances.append(dict(cohort=job['cohort'],repeat=job['repeat'],fold=job['fold'],
            train_n=len(train),test_n=len(test),train_patients=train.patient_id.nunique(),
            test_patients=test.patient_id.nunique(),pregnancy_n=int(test.outcome.eq(1).sum()),
            non_pregnancy_n=int(test.outcome.eq(0).sum()),patient_overlap=0))
    csv('04_MASTER_CLINICAL_V2_LOCAL_311.csv',d)
    csv('05_COHORT_COUNTS_BEFORE_FIT.csv',counts)
    csv('06_OUTER_FOLD_BALANCE_BEFORE_FIT.csv',balances)
    save('FIT_JOBS_LOCAL.json', jobs)
    (OUT / 'jobs').mkdir(exist_ok=True)
    return d,jobs,protocol


def fit_job(job):
    global _DATA
    if _DATA is None:
        _DATA=read_data()
    train,test = _DATA.iloc[job['train']],_DATA.iloc[job['test']]
    assignment=np.asarray(job['inner_assignment'])
    results=[]
    with threadpool_limits(limits=1):
        for model in MODELS[job['cohort']]:
            names=inputs(model)
            if model.endswith('_MSE'):
                assert train[MSE].notna().all() and test[MSE].notna().all()
            scores={C:[] for C in GRID}
            inner_states=[]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                for fold in range(1,6):
                    tr=train.iloc[assignment!=fold]; va=train.iloc[assignment==fold]
                    first=None
                    for C in GRID:
                        fitted=pipeline(model,C).fit(tr[names],tr.outcome.to_numpy(int))
                        state,feature_names=preprocessing_state(fitted,tr,model)
                        if first is None:
                            first=state
                        else:
                            assert state==first
                        probabilities=fitted.predict_proba(va[names])[:,1]
                        scores[C].append(float(log_loss(va.outcome,probabilities,labels=[0,1])))
                    inner_states.append(dict(inner_fold=fold,preprocessing=first))
                best=min(GRID,key=lambda C:np.mean(scores[C]))
                fitted=pipeline(model,best).fit(train[names],train.outcome.to_numpy(int))
                state,feature_names=preprocessing_state(fitted,train,model)
                p=fitted.predict_proba(test[names])[:,1]
                if any(issubclass(w.category,ConvergenceWarning) for w in caught):
                    raise RuntimeError('Ridge convergence failure; no model or grid substitution.')
            coefficients=fitted.named_steps['model'].coef_[0]
            intercept=float(fitted.named_steps['model'].intercept_[0])
            np.testing.assert_allclose(p,expit(reconstruct(test,state,feature_names)@coefficients+intercept),atol=1e-12)
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            results.append(dict(model=model,selected_C=best,predictions=p.tolist(),feature_names=feature_names,
                coefficients=coefficients.tolist(),intercept=intercept,preprocessing=state,
                inner_preprocessing=inner_states,classes=[0,1],converged=True,
                inner_scores=[dict(C=C,mean_logloss=float(np.mean(scores[C])),fold_logloss=scores[C]) for C in GRID]))
    save('jobs/'+job['key']+'.json',dict(**job,results=results))
    return job['key']


def collect(d,jobs):
    rows,choices,tuning,coefs=[],[],[],[]
    for job in jobs:
        result=json.loads((OUT/'jobs'/(job['key']+'.json')).read_text(encoding='utf-8'))
        for fit in result['results']:
            meta={k:job[k] for k in ['cohort','repeat','fold']}; meta['model']=fit['model']
            for index,p in zip(job['test'],fit['predictions']):
                r=d.iloc[index]
                rows.append(dict(**meta,exam_id=r.exam_id,patient_id=r.patient_id,outcome=int(r.outcome),predicted_probability=p))
            choices.append(dict(**meta,selected_C=fit['selected_C'],predictor_df=len(fit['feature_names'])))
            for s in fit['inner_scores']:
                tuning.append(dict(**meta,C=s['C'],mean_logloss=s['mean_logloss']))
            coefs += [dict(**meta,feature=f,coefficient=c,intercept=fit['intercept']) for f,c in zip(fit['feature_names'],fit['coefficients'])]
    new=pd.DataFrame(rows)
    history=pd.read_csv(HISTORY/'12_OOF_PREDICTIONS_LOCAL.csv',dtype={'exam_id':str,'patient_id':str})
    controls=history[(history.cohort.eq('Full306')&history.model.eq('Clinical9')) |
        (history.cohort.eq('MSE185')&history.model.isin(['Clinical9','Clinical9_MSE']))]
    pred=pd.concat([controls[new.columns],new],ignore_index=True)
    assert not pred.duplicated(['cohort','model','repeat','exam_id']).any()
    assert pred.groupby(['cohort','model','exam_id']).size().eq(10).all()
    csv('07_OOF_PREDICTIONS_LOCAL.csv',pred)
    csv('08_C_SELECTION.csv',choices); csv('09_INNER_TUNING.csv',tuning); csv('10_COEFFICIENTS.csv',coefs)
    return pred


def metrics(pred):
    performance,contrasts,folds,repeats,repeat_delta,fold_delta=[],[],[],[],[],[]
    for cohort in ['Full306','MSE185']:
        q=pred[pred.cohort.eq(cohort)]
        models=['Clinical9','Clinical_v2'] if cohort=='Full306' else ['Clinical9','Clinical9_MSE','Clinical_v2','Clinical_v2_MSE']
        wide=q.pivot(index=['repeat','exam_id','patient_id','outcome','fold'],columns='model',values='predicted_probability').reset_index()
        assert wide[models].notna().all().all()
        result=evaluate_frozen_scores(wide,models,iterations=B,seed=SEED)
        ci=np.quantile(result['draws'],[.025,.975],axis=0)
        for j,model in enumerate(models):
            y,p=wide.outcome.to_numpy(),wide[model].to_numpy()
            performance.append(dict(cohort=cohort,model=model,n_exams=result['n_exams'],
                n_unique_patients=result['n_unique_patients'],repeated_patient_groups=result['repeated_patient_groups'],
                pregnancy_n=result['pregnancy_n'],non_pregnancy_n=result['non_pregnancy_n'],
                AUC=float(result['points'][j]),CI_low=float(ci[0,j]),CI_high=float(ci[1,j]),
                Brier=float(brier_score_loss(y,p)),logloss=float(log_loss(y,p,labels=[0,1]))))
        contrasts_to_run=[('Clinical_v2','Clinical9','clinical_baseline_extension')]
        if cohort=='MSE185':
            contrasts_to_run += [('Clinical_v2_MSE','Clinical_v2','primary_v2_MSE_increment'),
                                 ('Clinical_v2_MSE','Clinical9_MSE','combined_absolute_AUC_change')]
        lookup=pd.DataFrame(performance).set_index(['cohort','model'])
        fold_table=result['folds']; repeat_table=result['repeats']
        for augmented,baseline,role in contrasts_to_run:
            i,j=models.index(augmented),models.index(baseline)
            lo,hi=np.quantile(result['draws'][:,i]-result['draws'][:,j],[.025,.975])
            contrasts.append(dict(cohort=cohort,role=role,augmented=augmented,baseline=baseline,
                n_exams=result['n_exams'],n_unique_patients=result['n_unique_patients'],
                Delta_AUC=float(result['points'][i]-result['points'][j]),CI_low=float(lo),CI_high=float(hi),
                crosses_zero=bool(lo<=0<=hi),
                Delta_Brier=float(lookup.loc[(cohort,augmented),'Brier']-lookup.loc[(cohort,baseline),'Brier']),
                Delta_logloss=float(lookup.loc[(cohort,augmented),'logloss']-lookup.loc[(cohort,baseline),'logloss'])))
            for repeat in range(10):
                a=repeat_table[(repeat_table['repeat']==repeat)&repeat_table.model.eq(augmented)].iloc[0]
                b=repeat_table[(repeat_table['repeat']==repeat)&repeat_table.model.eq(baseline)].iloc[0]
                repeat_delta.append(dict(cohort=cohort,role=role,repeat=repeat,Delta_AUC=float(a.weighted_fold_AUC-b.weighted_fold_AUC)))
            for (repeat,fold),f in fold_table.groupby(['repeat','fold']):
                f=f.set_index('model')
                fold_delta.append(dict(cohort=cohort,role=role,repeat=int(repeat),fold=int(fold),
                    Delta_AUC=float(f.loc[augmented,'AUC']-f.loc[baseline,'AUC']),pair_weight=int(f.loc[baseline,'pair_weight'])))
        if cohort=='MSE185':
            old_inc=result['draws'][:,1]-result['draws'][:,0]
            new_inc=result['draws'][:,3]-result['draws'][:,2]
            lo,hi=np.quantile(new_inc-old_inc,[.025,.975])
            point=(result['points'][3]-result['points'][2])-(result['points'][1]-result['points'][0])
            csv('15_CHANGE_IN_MSE_INCREMENT.csv',[dict(v2_minus_clinical9_MSE_increment=float(point),CI_low=float(lo),CI_high=float(hi),crosses_zero=bool(lo<=0<=hi))])
        csv('bootstrap_'+cohort+'.csv',pd.DataFrame(result['draws'],columns=models))
        for index,row in fold_table.iterrows():
            part=wide[(wide['repeat']==row['repeat'])&(wide.fold==row.fold)]
            fold_table.loc[index,'Brier']=brier_score_loss(part.outcome,part[row.model])
            fold_table.loc[index,'logloss']=log_loss(part.outcome,part[row.model],labels=[0,1])
        folds.append(fold_table.assign(cohort=cohort)); repeats.append(repeat_table.assign(cohort=cohort))
    csv('11_MODEL_PERFORMANCE.csv',performance); csv('12_PAIRED_COMPARISONS.csv',contrasts)
    csv('13a_EACH_OUTER_FOLD.csv',pd.concat(folds,ignore_index=True)); csv('13b_EACH_REPEAT.csv',pd.concat(repeats,ignore_index=True))
    csv('14a_REPEAT_PAIRED_DELTAS.csv',repeat_delta); csv('14b_FOLD_PAIRED_DELTAS.csv',fold_delta)
    print(pd.DataFrame(performance).round(6).to_string(index=False),flush=True)
    print(pd.DataFrame(contrasts).round(6).to_string(index=False),flush=True)


def run():
    d,jobs,protocol=prepare()
    save('progress.json',dict(stage='fitting',completed=0,total=len(jobs)))
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(fit_job,job) for job in jobs]
        for completed,future in enumerate(as_completed(futures),1):
            future.result()
            save('progress.json',dict(stage='fitting',completed=completed,total=len(jobs)))
            if completed%10==0:
                print(f'Completed {completed}/{len(jobs)} fixed outer jobs',flush=True)
    pred=collect(d,jobs)
    save('progress.json',dict(stage='fixed_OOF_paired_bootstrap'))
    metrics(pred)
    verify_sources(protocol)
    save('progress.json',dict(stage='metrics_complete_independent_audit_pending'))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='freeze':
        freeze()
    else:
        run()
