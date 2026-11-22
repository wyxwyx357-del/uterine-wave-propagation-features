"""Single prespecified PI+VFI sensitivity; frozen Clinical9/MSE and global folds."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime,timezone
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from analysis.pregnancy_prediction.code.clinical11_blood_audit import ROOT,OUT,PRIOR,OLD,BLOOD,sha,save,csv
from analysis.pregnancy_prediction.code.clinical9_data_audit import CORE,EXTRA,MSE
from analysis.pregnancy_prediction.code import clinical11_models as models
from analysis.pregnancy_prediction.code.frozen_foldwise_auc import evaluate_frozen_scores

SEED,B=20261120,10000
_DATA=None


def read_data():
    # Called only after the outcome-blind audit and PI/VFI definition are frozen.
    fields=['exam_id','patient_id','outcome','model_identity_eligible',MSE]+CORE+EXTRA
    base=pd.read_csv(PRIOR/'04_MASTER_CLINICAL9_LOCAL_311.csv',usecols=fields,dtype={'exam_id':str,'patient_id':str})
    blood=pd.read_csv(OUT/'07_PI_VFI_VALUES_LOCAL_311.csv',dtype={'exam_id':str})
    d=base.merge(blood,on='exam_id',how='left',validate='one_to_one',sort=False)
    assert d.exam_id.tolist()==base.exam_id.tolist() and len(d)==311
    return d


def fit_job(job):
    global _DATA
    if _DATA is None:
        _DATA=read_data()
    train,test=_DATA.iloc[job['train']],_DATA.iloc[job['test']]
    assert not set(train.patient_id)&set(test.patient_id)
    assert train[BLOOD].notna().all().all() and test[BLOOD].notna().all().all()
    names=models.FULL_MODELS if job['cohort']=='Blood287' else models.MSE_MODELS
    assignment=np.asarray(job['inner_assignment'])
    results=[]
    with threadpool_limits(limits=1):
        for model in names:
            columns=models.columns(model)
            scores={C:[] for C in models.GRID}
            inner_states=[]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                for fold in range(1,6):
                    a,b=np.flatnonzero(assignment!=fold),np.flatnonzero(assignment==fold)
                    tr,va=train.iloc[a],train.iloc[b]
                    assert not set(tr.patient_id)&set(va.patient_id)
                    assert tr.outcome.nunique()==va.outcome.nunique()==2
                    first=None
                    for C in models.GRID:
                        fitted=models.pipeline(model,C).fit(tr[columns],tr.outcome.to_numpy(int))
                        state,features=models.state(fitted,tr,model)
                        if first is None:
                            first=state
                        else:
                            assert state==first
                        p=fitted.predict_proba(va[columns])[:,1]
                        scores[C].append(float(log_loss(va.outcome,p,labels=[0,1])))
                    inner_states.append(dict(inner_fold=fold,train_indices=a.tolist(),validation_indices=b.tolist(),
                        preprocessing=first,patient_overlap=0))
                best=min(models.GRID,key=lambda C:np.mean(scores[C]))
                fitted=models.pipeline(model,best).fit(train[columns],train.outcome.to_numpy(int))
                state,features=models.state(fitted,train,model)
                p=fitted.predict_proba(test[columns])[:,1]
                if any(issubclass(w.category,ConvergenceWarning) for w in caught):
                    raise RuntimeError('Frozen blood Ridge convergence failure; no substitution or grid change')
            beta=fitted.named_steps['model'].coef_[0]
            intercept=float(fitted.named_steps['model'].intercept_[0])
            np.testing.assert_allclose(p,expit(models.reconstruct(test,state,features)@beta+intercept),atol=1e-12)
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            results.append(dict(model=model,selected_C=best,probabilities=p.tolist(),preprocessing=state,
                feature_names=features,coefficients=beta.tolist(),intercept=intercept,classes=[0,1],converged=True,
                inner_preprocessing=inner_states,
                inner_scores=[dict(C=C,mean_logloss=float(np.mean(scores[C])),fold_logloss=scores[C]) for C in models.GRID]))
    save(OUT/'jobs'/(job['key']+'.json'),dict(**job,results=results))
    return job['key']


def prepare():
    assert not (OUT/'PROTOCOL_BEFORE_FIT.json').exists()
    audit=json.loads((OUT/'BLOOD_AUDIT_BEFORE_OUTCOME.json').read_text(encoding='utf-8'))
    assert audit['outcome_columns_loaded'] is False
    assert all(sha(Path(p))==h for p,h in audit['source_hashes'].items())
    checks=models.check()
    d=read_data()
    membership=pd.read_csv(OUT/'06_MEMBERSHIP_BEFORE_OUTCOMES_LOCAL.csv',dtype={'exam_id':str,'patient_id':str})
    assert membership.exam_id.tolist()==d.exam_id.tolist()
    eligible=d.model_identity_eligible.eq(True)&d.outcome.isin([0,1])
    assert np.array_equal(eligible,membership.existing_eligible)
    masks={'Blood287':membership.Blood_full_observed.eq(True),'MSE_Blood180':membership.Blood_MSE_observed.eq(True)}
    assert masks['Blood287'].sum()==287 and masks['MSE_Blood180'].sum()==180
    outer=pd.read_csv(OLD/'02a_GLOBAL_PATIENT_FOLDS.csv',dtype={'exam_id':str,'patient_id':str})
    inner=pd.read_csv(OLD/'02b_GLOBAL_INNER_VALIDATION_FOLDS.csv',dtype={'exam_id':str,'patient_id':str})
    jobs,balances,cohorts=[],[],[]
    for cohort,mask in masks.items():
        sub=d[mask]
        assert sub[BLOOD].notna().all().all()
        if cohort=='MSE_Blood180':
            assert sub[MSE].notna().all()
        cohorts.append(dict(cohort=cohort,n_exams=len(sub),n_unique_patients=sub.patient_id.nunique(),
            repeated_patient_groups=int(sub.patient_id.value_counts().gt(1).sum()),pregnancy_n=int(sub.outcome.eq(1).sum()),
            non_pregnancy_n=int(sub.outcome.eq(0).sum()),Clinical9_complete_n=int(sub[CORE+EXTRA].notna().all(axis=1).sum()),
            no_blood_imputation=True))
        for repeat in range(10):
            mapping=outer[outer['repeat'].eq(repeat)].set_index('exam_id')
            assert sub.exam_id.map(mapping.patient_id).eq(sub.patient_id).all()
            folds=sub.exam_id.map(mapping.fold)
            assert folds.notna().all()
            for fold in range(1,6):
                a,b=sub.index[folds.ne(fold)].tolist(),sub.index[folds.eq(fold)].tolist()
                tr,te=d.iloc[a],d.iloc[b]
                assert not set(tr.patient_id)&set(te.patient_id)
                imap=inner[inner['repeat'].eq(repeat)&inner.outer_fold.eq(fold)].set_index('exam_id')
                ia=tr.exam_id.map(imap.inner_fold)
                assert ia.notna().all() and tr.exam_id.map(imap.patient_id).eq(tr.patient_id).all()
                for f in range(1,6):
                    x,y=tr.loc[ia.ne(f)],tr.loc[ia.eq(f)]
                    assert not set(x.patient_id)&set(y.patient_id)
                    assert x.outcome.nunique()==y.outcome.nunique()==2
                jobs.append(dict(key=f'{cohort}_r{repeat:02d}_f{fold}',cohort=cohort,repeat=repeat,fold=fold,
                    train=a,test=b,inner_assignment=ia.astype(int).tolist()))
                balances.append(dict(cohort=cohort,repeat=repeat,fold=fold,train_n=len(tr),test_n=len(te),
                    train_patients=tr.patient_id.nunique(),test_patients=te.patient_id.nunique(),
                    test_pregnancy_n=int(te.outcome.eq(1).sum()),test_nonpregnancy_n=int(te.outcome.eq(0).sum()),patient_overlap=0))
    csv('09_COHORT_COUNTS_BEFORE_FIT.csv',cohorts)
    csv('10_OUTER_FOLD_BALANCE_BEFORE_FIT.csv',balances)
    paths=[Path(p) for p in audit['source_hashes']]+[OUT/'BLOOD_AUDIT_BEFORE_OUTCOME.json',
        OUT/'07_PI_VFI_VALUES_LOCAL_311.csv',OUT/'06_MEMBERSHIP_BEFORE_OUTCOMES_LOCAL.csv',
        OLD/'02b_GLOBAL_INNER_VALIDATION_FOLDS.csv',PRIOR/'16_MODEL_PERFORMANCE.csv',
        PRIOR/'17_PAIRED_COMPARISONS.csv',PRIOR/'STOP_RULE.json',
        ROOT/'outputs/static_elasticnet_final_exploration_20261122/STOP_RULE.json']
    scripts=[Path(__file__),Path(models.__file__),Path(__file__).with_name('clinical11_blood_audit.py'),
        Path(__file__).with_name('clinical9_models.py'),Path(__file__).with_name('clinical9_data_audit.py'),
        Path(__file__).with_name('frozen_foldwise_auc.py')]
    protocol=dict(version='one_prespecified_Clinical9_PI_VFI_sensitivity_20261122',frozen_UTC=datetime.now(timezone.utc).isoformat(),
        analysis='Clinical9基础上的一次预先限定血流敏感性分析',clinical9=CORE+EXTRA,
        blood_only=BLOOD,blood_definition=audit['cleaning'],quality_limitations=audit['PI_VFI_quality_decision'],
        formula_anomalies='retained source values and flags, no VI/FI-derived replacement, no exclusion',
        MSE_definition_unchanged=MSE,Clinical9_source_unchanged=True,
        models={'Blood287':models.FULL_MODELS,'MSE_Blood180':models.MSE_MODELS},cohort_counts=cohorts,
        primary_comparison='MSE_Blood180:Clinical11_MSE minus Clinical11',
        baseline_blood_comparison='Blood287:Clinical11 minus Clinical9',
        necessary_same_sample_control='MSE_Blood180:Clinical9_MSE minus Clinical9; refit on same180, not restricted old185 predictions',
        probability_and_absolute_discrimination_comparison='same180:Clinical11_MSE minus Clinical9_MSE',
        paired_change_in_MSE_increment='(Clinical11_MSE-Clinical11)-(Clinical9_MSE-Clinical9), same180 same bootstrap draws',
        historical_185='copy Clinical6,Clinical9,Clinical9+position MSE comparisons unchanged; historical vs180 not directly matched',
        folds='original global patient-grouped outer5/inner5,10repeats; restrict maps, no seed regeneration',
        Ridge='same L2/lbfgs/max_iter10000/tol1e-4',C_grid=models.GRID,selection='minimum mean inner5 logloss; ascending C exact tie',
        preprocessing='Clinical9 unchanged train median/zscore numeric, train mode/unscaled binary; blood observed only train zscore; MSE observed only train zscore',
        no_blood_or_MSE_missing_imputation=True,Clinical9_missing_train_only_imputed=True,
        evaluation='within-fold pair-weighted AUC per repeat, mean10; Brier/logloss mean exam/repeat',
        bootstrap=dict(n=B,seed=SEED,unit='patient cluster',scope='paired fixed OOF conditional95%; no full training-refit/prior search uncertainty'),
        interpretation='positive MSE signal may include information not fully replaced by routine PI/VFI; never claim independence from blood-flow mechanism',
        no_other_blood_combinations=True,no_other_imaging_or_models=True,
        stop_rule='Complete once, then no SD+VFI/RI+VI/all6 blood search, no Clinical9/MSE changes, no static/dynamic model search.',
        pipeline_verification=checks,source_hashes={str(p):sha(p) for p in paths},script_hashes={str(p):sha(p) for p in scripts})
    (OUT/'jobs').mkdir()
    save(OUT/'FIT_JOBS_LOCAL.json',jobs)
    save(OUT/'PROTOCOL_BEFORE_FIT.json',protocol)
    save(OUT/'STOP_RULE.json',dict(status='one_prespecified_blood_analysis_in_progress',rule=protocol['stop_rule']))
    return d,jobs,protocol


def collect(d,jobs):
    rows,choices,coefficients,tuning=[],[],[],[]
    for job in jobs:
        result=json.loads((OUT/'jobs'/(job['key']+'.json')).read_text(encoding='utf-8'))
        for fit in result['results']:
            for index,p in zip(job['test'],fit['probabilities']):
                row=d.iloc[index]
                rows.append(dict(cohort=job['cohort'],model=fit['model'],repeat=job['repeat'],fold=job['fold'],exam_id=row.exam_id,
                    patient_id=row.patient_id,outcome=int(row.outcome),predicted_probability=p))
            choices.append(dict(cohort=job['cohort'],model=fit['model'],repeat=job['repeat'],fold=job['fold'],
                selected_C=fit['selected_C'],predictor_df=len(fit['feature_names'])))
            coefficients.extend(dict(cohort=job['cohort'],model=fit['model'],repeat=job['repeat'],fold=job['fold'],
                feature=f,coefficient=c,intercept=fit['intercept']) for f,c in zip(fit['feature_names'],fit['coefficients']))
            tuning.extend(dict(cohort=job['cohort'],model=fit['model'],repeat=job['repeat'],fold=job['fold'],
                C=s['C'],mean_inner_logloss=s['mean_logloss']) for s in fit['inner_scores'])
    pred=pd.DataFrame(rows)
    assert len(choices)==300 and not pred.duplicated(['cohort','model','repeat','exam_id']).any()
    assert pred.groupby(['cohort','model','exam_id']).size().eq(10).all()
    csv('11_OOF_PREDICTIONS_LOCAL.csv',pred)
    csv('12_C_SELECTION.csv',choices)
    csv('13_COEFFICIENTS.csv',coefficients)
    csv('14_INNER_TUNING.csv',tuning)
    return pred


def metrics(pred,protocol):
    performance,contrasts,folds,repeats=[],[],[],[]
    for cohort in ['Blood287','MSE_Blood180']:
        q=pred[pred.cohort.eq(cohort)]
        names=models.FULL_MODELS if cohort=='Blood287' else models.MSE_MODELS
        wide=q.pivot(index=['repeat','exam_id','patient_id','outcome','fold'],columns='model',values='predicted_probability').reset_index()
        assert wide[names].notna().all().all() and wide.groupby(['repeat','patient_id']).fold.nunique().eq(1).all()
        result=evaluate_frozen_scores(wide,names,iterations=B,seed=SEED)
        ci=np.quantile(result['draws'],[.025,.975],axis=0)
        for j,model in enumerate(names):
            p,y=wide[model].to_numpy(),wide.outcome.to_numpy()
            p=np.clip(p,np.finfo(float).eps,1-np.finfo(float).eps)
            performance.append(dict(cohort=cohort,model=model,n_exams=result['n_exams'],n_unique_patients=result['n_unique_patients'],
                repeated_patient_groups=result['repeated_patient_groups'],pregnancy_n=result['pregnancy_n'],non_pregnancy_n=result['non_pregnancy_n'],
                AUC=float(result['points'][j]),CI_low=float(ci[0,j]),CI_high=float(ci[1,j]),Brier=float(np.mean((p-y)**2)),
                logloss=float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))))
        pairs=[('Clinical11','Clinical9','blood_baseline')] if cohort=='Blood287' else [
            ('Clinical9_MSE','Clinical9','matched_preblood_MSE'),('Clinical11_MSE','Clinical11','primary_postblood_MSE'),
            ('Clinical11_MSE','Clinical9_MSE','blood_added_to_MSE_model')]
        perf=pd.DataFrame(performance).set_index(['cohort','model'])
        for augmented,baseline,role in pairs:
            i,j=names.index(augmented),names.index(baseline)
            draw=result['draws'][:,i]-result['draws'][:,j]
            low,high=np.quantile(draw,[.025,.975])
            contrasts.append(dict(cohort=cohort,role=role,augmented=augmented,baseline=baseline,
                n_exams=result['n_exams'],n_unique_patients=result['n_unique_patients'],
                Delta_AUC=float(result['points'][i]-result['points'][j]),CI_low=float(low),CI_high=float(high),crosses_zero=bool(low<=0<=high),
                Delta_Brier=float(perf.loc[(cohort,augmented),'Brier']-perf.loc[(cohort,baseline),'Brier']),
                Delta_logloss=float(perf.loc[(cohort,augmented),'logloss']-perf.loc[(cohort,baseline),'logloss'])))
        if cohort=='MSE_Blood180':
            def change(values):
                return values[...,names.index('Clinical11_MSE')]-values[...,names.index('Clinical11')]-values[...,names.index('Clinical9_MSE')]+values[...,names.index('Clinical9')]
            low,high=np.quantile(change(result['draws']),[.025,.975])
            csv('18_MATCHED_CHANGE_IN_MSE_INCREMENT.csv',[dict(postblood_minus_preblood_MSE_increment=float(change(result['points'])),
                CI_low=float(low),CI_high=float(high),crosses_zero=bool(low<=0<=high),n_exams=180,n_unique_patients=174)])
            for table,field,index,file in [(result['folds'],'AUC',['repeat','fold'],'19a_EACH_FOLD_MSE_INCREMENTS.csv'),
                (result['repeats'],'weighted_fold_AUC',['repeat'],'19b_EACH_REPEAT_MSE_INCREMENTS.csv')]:
                values=table.pivot(index=index,columns='model',values=field).reset_index()
                values['Clinical9_MSE_increment']=values.Clinical9_MSE-values.Clinical9
                values['Clinical11_MSE_increment']=values.Clinical11_MSE-values.Clinical11
                values['postblood_minus_preblood_increment']=values.Clinical11_MSE_increment-values.Clinical9_MSE_increment
                csv(file,values)
        csv('bootstrap_'+cohort+'.csv',pd.DataFrame(result['draws'],columns=names))
        folds.append(result['folds'].assign(cohort=cohort))
        repeats.append(result['repeats'].assign(cohort=cohort))
    csv('15_MODEL_PERFORMANCE.csv',performance)
    csv('16_PAIRED_COMPARISONS.csv',contrasts)
    csv('17a_EACH_OUTER_FOLD.csv',pd.concat(folds,ignore_index=True))
    csv('17b_EACH_REPEAT.csv',pd.concat(repeats,ignore_index=True))
    past=pd.read_csv(PRIOR/'17_PAIRED_COMPARISONS.csv')
    past=past[past.role.isin(['historical_MSE','primary_MSE','position_adjusted_MSE'])].copy()
    past['comparison_scope']='historical185_unchanged_different_case_set'
    new=pd.DataFrame(contrasts)
    new=new[new.role.isin(['matched_preblood_MSE','primary_postblood_MSE'])].copy()
    new['comparison_scope']='new180_same_cases_training_folds'
    csv('20_HISTORICAL_AND_NEW_MSE_INCREMENTS.csv',pd.concat([past,new],ignore_index=True))
    assert all(sha(Path(p))==h for p,h in protocol['source_hashes'].items())
    save(OUT/'CHECKS_FINAL.json',dict(all300_outer_and9000_inner_fits_converged=True,all_pairing_and_patient_groups_pass=True,
        all_preprocessing_inner_training_only=True,only_PI_VFI_added=True,Clinical9_MSE_unchanged=True,
        old_files_and_stop_rules_unchanged=True,no_alternative_blood_combination_or_model_search=True,
        fixed_OOF_conditional_bootstrap_only=True,independent_audit_pending=True))
    print(pd.DataFrame(performance).round(6).to_string(index=False),flush=True)
    print(pd.DataFrame(contrasts).round(6).to_string(index=False),flush=True)


def run():
    d,jobs,protocol=prepare()
    save(OUT/'progress.json',dict(stage='fitting',completed=0,total=100))
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(fit_job,job) for job in jobs]
        for completed,future in enumerate(as_completed(futures),1):
            future.result()
            save(OUT/'progress.json',dict(stage='fitting',completed=completed,total=100))
            if completed%10==0:
                print(f'Completed{completed}/100 frozen blood outer tasks',flush=True)
    pred=collect(d,jobs)
    save(OUT/'progress.json',dict(stage='paired_fixed_OOF_bootstrap'))
    metrics(pred,protocol)
    save(OUT/'progress.json',dict(stage='metrics_complete_audit_report_pending'))


if __name__=='__main__':
    run()
