"""One authorized Enhanced Clinical EN v1 development and compulsory-MSE test."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from analysis.pregnancy_prediction.code.enhanced_clinical_audit import ROOT,OUT,PRIOR,BLOOD_PRIOR,BLOOD,sha,save,csv
from analysis.pregnancy_prediction.code.clinical9_data_audit import CORE,EXTRA,MSE,OLD
from analysis.pregnancy_prediction.code import enhanced_clinical_models as en
from analysis.pregnancy_prediction.code.frozen_foldwise_auc import evaluate_frozen_scores

SEED,B=20261120,10000
_DATA=None
MODELS={'Full306':['Clinical6','Clinical9','Enhanced'],
        'MSE185':['Clinical6_historical','Clinical6_MSE_historical','Clinical9','Clinical9_MSE','Enhanced','Enhanced_MSE']}


def read_data():
    fields=['exam_id','patient_id','outcome','model_identity_eligible',MSE]+CORE+EXTRA+['uterine_position_clean']
    d=pd.read_csv(PRIOR/'04_MASTER_CLINICAL9_LOCAL_311.csv',usecols=fields,dtype={'exam_id':str,'patient_id':str})
    blood=pd.read_csv(OUT/'02_CLEAN_BLOOD_VALUES_LOCAL_311.csv',dtype={'exam_id':str})
    joined=d.merge(blood,on='exam_id',how='left',validate='one_to_one',sort=False)
    assert joined.exam_id.tolist()==d.exam_id.tolist() and len(joined)==311
    return joined


def fit_job(job):
    global _DATA
    if _DATA is None:
        _DATA=read_data()
    train,test=_DATA.iloc[job['train']],_DATA.iloc[job['test']]
    assert not set(train.patient_id)&set(test.patient_id)
    assignment=np.asarray(job['inner_assignment'])
    models=['Enhanced'] if job['cohort']=='Full306' else ['Enhanced','Enhanced_MSE']
    results=[]
    with threadpool_limits(limits=1):
        for model in models:
            mse=model.endswith('_MSE')
            scores={(a,l):[] for a in en.ALPHAS for l in en.LAMBDAS}
            diagnostics,states=[],[]
            for f in range(1,6):
                a,b=np.flatnonzero(assignment!=f),np.flatnonzero(assignment==f)
                tr,va=train.iloc[a],train.iloc[b]
                assert not set(tr.patient_id)&set(va.patient_id)
                assert tr.outcome.nunique()==va.outcome.nunique()==2
                X,V,state=en.preprocess(tr,va,mse)
                states.append(dict(inner_fold=f,train_indices=a.tolist(),validation_indices=b.tolist(),preprocessing=state,patient_overlap=0))
                for alpha in en.ALPHAS:
                    warm=None
                    for lam in en.LAMBDAS:
                        fitted=en.fit(X,tr.outcome.to_numpy(int),mse,alpha,lam,warm)
                        eta=fitted['intercept']+V@np.asarray(fitted['coef'])
                        loss=float(np.mean(np.logaddexp(0,eta)-va.outcome.to_numpy(int)*eta))
                        scores[(alpha,lam)].append(loss)
                        diagnostics.append(dict(inner_fold=f,alpha=alpha,lambda_sum_loss=lam,validation_logloss=loss,
                            KKT_residual=fitted['KKT_residual'],iterations=fitted['iterations'],
                            proximal_polish_iterations=fitted['proximal_polish_iterations'],
                            n_clinical_constructs=fitted['selected_clinical_constructs']))
                        warm=fitted
            tuning=[dict(alpha=a,lambda_sum_loss=l,C=1/l,mean_inner_logloss=float(np.mean(v)),fold_logloss=v)
                    for (a,l),v in scores.items()]
            best=min(tuning,key=lambda r:(r['mean_inner_logloss'],-r['lambda_sum_loss'],-r['alpha']))
            X,T,state=en.preprocess(train,test,mse)
            fitted=en.fit(X,train.outcome.to_numpy(int),mse,best['alpha'],best['lambda_sum_loss'])
            probability=en.solver.predict(fitted,T)
            assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
            results.append(dict(model=model,**fitted,probabilities=probability.tolist(),preprocessing=state,
                feature_names=([MSE] if mse else [])+en.FEATURES,MSE_always_entered=mse,
                inner_preprocessing=states,inner_tuning=tuning,inner_numerical_diagnostics=diagnostics,classes=[0,1]))
    save(OUT/'jobs'/(job['key']+'.json'),dict(**job,results=results))
    return job['key']


def prepare():
    assert not (OUT/'PROTOCOL_BEFORE_FIT.json').exists()
    quality=json.loads((OUT/'BLOOD_CLEANING_BEFORE_OUTCOME.json').read_text(encoding='utf-8'))
    assert quality['outcome_columns_loaded'] is False
    assert all(sha(Path(p))==h for p,h in quality['source_hashes'].items())
    checks=en.verify()
    d=read_data()
    jobs=json.loads((PRIOR/'FIT_JOBS_LOCAL.json').read_text(encoding='utf-8'))
    assert len(jobs)==100 and set(j['cohort'] for j in jobs)=={'Full306','MSE185'}
    old=pd.read_csv(PRIOR/'04_MASTER_CLINICAL9_LOCAL_311.csv',usecols=['exam_id','patient_id','outcome',MSE]+CORE+EXTRA,
        dtype={'exam_id':str,'patient_id':str})
    assert d[['exam_id','patient_id']].equals(old[['exam_id','patient_id']])
    np.testing.assert_allclose(d[CORE+EXTRA+[MSE,'outcome']],old[CORE+EXTRA+[MSE,'outcome']],equal_nan=True)
    memberships=[]
    for cohort in ['Full306','MSE185']:
        indices=sorted(set(i for j in jobs if j['cohort']==cohort for i in j['test']))
        part=d.iloc[indices]
        assert len(part)==(306 if cohort=='Full306' else 185)
        memberships.append(dict(cohort=cohort,n_exams=len(part),n_unique_patients=part.patient_id.nunique(),
            repeated_patient_groups=int(part.patient_id.value_counts().gt(1).sum()),pregnancy_n=int(part.outcome.eq(1).sum()),
            non_pregnancy_n=int(part.outcome.eq(0).sum()),PI_missing=int(part.PI_clean.isna().sum()),VFI_missing=int(part.VFI_clean.isna().sum())))
    for job in jobs:
        assert not set(d.iloc[job['train']].patient_id)&set(d.iloc[job['test']].patient_id)
    csv('04_COHORT_COUNTS_BEFORE_FIT.csv',memberships)
    paths=[Path(p) for p in quality['source_hashes']]+[OUT/'BLOOD_CLEANING_BEFORE_OUTCOME.json',OUT/'02_CLEAN_BLOOD_VALUES_LOCAL_311.csv',
        PRIOR/'FIT_JOBS_LOCAL.json',PRIOR/'12_OOF_PREDICTIONS_LOCAL.csv',PRIOR/'16_MODEL_PERFORMANCE.csv',
        PRIOR/'17_PAIRED_COMPARISONS.csv',PRIOR/'STOP_RULE.json',OLD/'02b_GLOBAL_INNER_VALIDATION_FOLDS.csv',
        ROOT/'outputs/static_elasticnet_final_exploration_20261122/STOP_RULE.json']
    scripts=[Path(__file__),Path(en.__file__),Path(en.solver.__file__),Path(__file__).with_name('enhanced_clinical_audit.py'),
        Path(__file__).with_name('frozen_foldwise_auc.py')]
    protocol=dict(name='Enhanced Clinical baseline v1',version='v1_20261122',frozen_UTC=datetime.now(timezone.utc).isoformat(),
        candidate_constructs=en.CONSTRUCTS,n_constructs=12,clinical_design_columns=en.FEATURES,n_clinical_columns=13,
        primary_baseline_comparison='Full306:Enhanced minus frozen Clinical9',
        primary_MSE_comparison='MSE185:Enhanced_MSE minus Enhanced',
        necessary_absolute_AUC_comparison='MSE185:Enhanced_MSE minus frozen Clinical9_MSE',
        secondary_baseline_MSE_subset='MSE185:Enhanced minus frozen Clinical9',
        clinical9_unchanged=True,blood_cleaning=quality['VFI_rule'],no_clinical_missing_row_deletion=True,
        preprocessing='continuous training median; stage/pureC training mode; position training mode+fixed anterior reference posterior/horizontal dummies; ALL13 clinical columns training zscore',
        MSE_handling='observed MSE mandatory first column, train zscore, L2-only same lambda scale, never L1-selected; effect may shrink toward zero',
        objective='sum(logloss)+lambda/2*beta_MSE^2(if present)+lambda*(1-alpha)/2*||clinical||2^2+lambda*alpha*||clinical||1; intercept free',
        mean_loss_solver='divide complete objective by training_n; same optimum',
        alpha_grid=en.ALPHAS,C_grid=[1/l for l in en.LAMBDAS],lambda_sum_loss_grid=en.LAMBDAS,
        tuning='18 fixed candidates, minimum mean inner5 logloss; exact ties stronger lambda then larger alpha',
        outer5_inner5_repeats10=True,all_old_global_splits_reused=True,seed_search_forbidden=True,
        algorithm='only one clinical EN procedure, same generic block convex solver L-BFGS-B with same-objective proximal polishing',
        KKT_tolerance=en.solver.KKT_TOLERANCE,solver_verification=checks,
        Clinical9_control='reuse frozen same-cohort OOF; original six/nine and MSE pairs remain unchanged',
        model_columns=MODELS,selection_frequency='per cohort/model50 outer fits; all12 constructs and13 columns; zeros retained; no frequency-based refit',
        evaluation='within-fold AUC weighted npositive*nnegative per repeat then mean10; Brier/logloss average exam/repeat',
        bootstrap=dict(n=B,seed=SEED,unit='patient-cluster paired fixed OOF',scope='conditional95%; not full workflow refit or prior exploration uncertainty'),
        stop_rule='Freeze Enhanced Clinical baseline v1 once regardless of result. No LASSO/RF/SVM/XGBoost, pool/grid/seed/coding expansion, no selected-frequency fixed-model refit.',
        interpretation='one prespecified procedure internal performance; cannot estimate theoretical maximum achievable AUC',
        source_hashes={str(p):sha(p) for p in paths},script_hashes={str(p):sha(p) for p in scripts})
    (OUT/'jobs').mkdir()
    save(OUT/'FIT_JOBS_LOCAL.json',jobs)
    save(OUT/'PROTOCOL_BEFORE_FIT.json',protocol)
    save(OUT/'STOP_RULE.json',dict(status='Enhanced_v1_frozen_analysis_running',rule=protocol['stop_rule']))
    return d,jobs,protocol


def collect(d,jobs):
    prior=pd.read_csv(PRIOR/'12_OOF_PREDICTIONS_LOCAL.csv',dtype={'exam_id':str,'patient_id':str})
    prior=prior[((prior.cohort.eq('Full306'))&prior.model.isin(['Clinical6','Clinical9']))|
        ((prior.cohort.eq('MSE185'))&prior.model.isin(MODELS['MSE185'][:4]))].copy()
    rows,choices,coefs,tuning,diagnostics=[],[],[],[],[]
    for job in jobs:
        saved=json.loads((OUT/'jobs'/(job['key']+'.json')).read_text(encoding='utf-8'))
        for fit in saved['results']:
            model=fit['model']
            for index,probability in zip(job['test'],fit['probabilities']):
                r=d.iloc[index]
                rows.append(dict(cohort=job['cohort'],model=model,repeat=job['repeat'],fold=job['fold'],exam_id=r.exam_id,
                    patient_id=r.patient_id,outcome=int(r.outcome),predicted_probability=probability))
            choices.append(dict(cohort=job['cohort'],model=model,repeat=job['repeat'],fold=job['fold'],alpha=fit['alpha'],
                C=1/fit['lambda_sum_loss'],lambda_sum_loss=fit['lambda_sum_loss'],
                selected_clinical_columns=fit['selected_clinical_columns'],selected_clinical_constructs=fit['selected_clinical_constructs'],
                MSE_forced=fit['MSE_always_entered'],KKT_residual=fit['KKT_residual']))
            for name,coef in zip(fit['feature_names'],fit['coef']):
                coefs.append(dict(cohort=job['cohort'],model=model,repeat=job['repeat'],fold=job['fold'],
                    feature=name,construct='uterine_position' if name in en.POSITION else name,standardized_coefficient=coef,
                    role='forced_MSE_L2' if name==MSE else 'clinical_EN',selected=bool(coef!=0) if name!=MSE else True,
                    intercept=fit['intercept']))
            tuning.extend(dict(cohort=job['cohort'],model=model,repeat=job['repeat'],fold=job['fold'],
                alpha=r['alpha'],C=r['C'],mean_inner_logloss=r['mean_inner_logloss']) for r in fit['inner_tuning'])
            diagnostics.extend(dict(cohort=job['cohort'],model=model,repeat=job['repeat'],fold=job['fold'],**r) for r in fit['inner_numerical_diagnostics'])
    pred=pd.concat([prior,pd.DataFrame(rows)],ignore_index=True)
    assert not pred.duplicated(['cohort','model','repeat','exam_id']).any() and pred.groupby(['cohort','model','exam_id']).size().eq(10).all()
    assert len(choices)==150 and len(diagnostics)==13500
    csv('05_OOF_PREDICTIONS_LOCAL.csv',pred)
    csv('06_OUTER_PARAMETERS_AND_COUNTS.csv',choices)
    csv('07_STANDARDIZED_COEFFICIENTS.csv',coefs)
    csv('08_INNER_TUNING.csv',tuning)
    csv('09_INNER_NUMERICAL_DIAGNOSTICS.csv',diagnostics)
    coef=pd.DataFrame(coefs)
    clinical=coef[coef.role.eq('clinical_EN')]
    freq=clinical.groupby(['cohort','model','feature','construct']).agg(evaluated_outer_fits=('selected','size'),
        selected_outer_fits=('selected','sum'),mean_standardized_coefficient=('standardized_coefficient','mean'),
        median_standardized_coefficient=('standardized_coefficient','median'),
        positive_fits=('standardized_coefficient',lambda x:int((x>0).sum())),negative_fits=('standardized_coefficient',lambda x:int((x<0).sum()))).reset_index()
    freq['selection_frequency']=freq.selected_outer_fits/freq.evaluated_outer_fits
    assert len(freq)==39 and freq.evaluated_outer_fits.eq(50).all()
    csv('10_COLUMN_SELECTION_FREQUENCIES.csv',freq)
    group=clinical.groupby(['cohort','model','repeat','fold','construct']).selected.any().reset_index()
    groups=group.groupby(['cohort','model','construct']).selected.agg(['size','sum']).reset_index().rename(columns={'size':'evaluated_outer_fits','sum':'selected_outer_fits'})
    groups['selection_frequency']=groups.selected_outer_fits/groups.evaluated_outer_fits
    assert len(groups)==36 and groups.evaluated_outer_fits.eq(50).all()
    csv('11_CONSTRUCT_SELECTION_FREQUENCIES.csv',groups)
    return pred


def metrics(pred,protocol):
    performances,contrasts,folds,repeats=[],[],[],[]
    for cohort,names in MODELS.items():
        q=pred[pred.cohort.eq(cohort)]
        wide=q.pivot(index=['repeat','exam_id','patient_id','outcome','fold'],columns='model',values='predicted_probability').reset_index()
        assert wide[names].notna().all().all() and wide.groupby(['repeat','patient_id']).fold.nunique().eq(1).all()
        result=evaluate_frozen_scores(wide,names,B,SEED)
        ci=np.quantile(result['draws'],[.025,.975],axis=0)
        for j,model in enumerate(names):
            p,y=wide[model].to_numpy(),wide.outcome.to_numpy()
            p=np.clip(p,np.finfo(float).eps,1-np.finfo(float).eps)
            performances.append(dict(cohort=cohort,model=model,n_exams=result['n_exams'],n_unique_patients=result['n_unique_patients'],
                repeated_patient_groups=result['repeated_patient_groups'],pregnancy_n=result['pregnancy_n'],non_pregnancy_n=result['non_pregnancy_n'],
                AUC=float(result['points'][j]),CI_low=float(ci[0,j]),CI_high=float(ci[1,j]),Brier=float(np.mean((p-y)**2)),
                logloss=float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))))
        pairs=[('Enhanced','Clinical9','primary_Enhanced_vs_Clinical9')] if cohort=='Full306' else [
            ('Enhanced_MSE','Enhanced','primary_MSE_on_Enhanced'),('Enhanced','Clinical9','Enhanced_vs_Clinical9_MSE_subset'),
            ('Enhanced_MSE','Clinical9_MSE','absolute_model_change_vs_Clinical9_MSE'),
            ('Clinical9_MSE','Clinical9','historical_MSE_on_Clinical9'),
            ('Clinical6_MSE_historical','Clinical6_historical','historical_MSE_on_Clinical6')]
        perf=pd.DataFrame(performances).set_index(['cohort','model'])
        for augmented,baseline,role in pairs:
            i,j=names.index(augmented),names.index(baseline)
            delta=result['draws'][:,i]-result['draws'][:,j]
            low,high=np.quantile(delta,[.025,.975])
            contrasts.append(dict(cohort=cohort,role=role,augmented=augmented,baseline=baseline,
                n_exams=result['n_exams'],n_unique_patients=result['n_unique_patients'],
                Delta_AUC=float(result['points'][i]-result['points'][j]),CI_low=float(low),CI_high=float(high),crosses_zero=bool(low<=0<=high),
                Delta_Brier=float(perf.loc[(cohort,augmented),'Brier']-perf.loc[(cohort,baseline),'Brier']),
                Delta_logloss=float(perf.loc[(cohort,augmented),'logloss']-perf.loc[(cohort,baseline),'logloss'])))
        if cohort=='MSE185':
            for table,field,keys,file in [(result['folds'],'AUC',['repeat','fold'],'16a_EACH_FOLD_MSE_INCREMENTS.csv'),
                (result['repeats'],'weighted_fold_AUC',['repeat'],'16b_EACH_REPEAT_MSE_INCREMENTS.csv')]:
                values=table.pivot(index=keys,columns='model',values=field).reset_index()
                values['Enhanced_MSE_increment']=values.Enhanced_MSE-values.Enhanced
                values['Clinical9_MSE_increment']=values.Clinical9_MSE-values.Clinical9
                csv(file,values)
        csv('bootstrap_'+cohort+'.csv',pd.DataFrame(result['draws'],columns=names))
        folds.append(result['folds'].assign(cohort=cohort))
        repeats.append(result['repeats'].assign(cohort=cohort))
    csv('12_MODEL_PERFORMANCE.csv',performances)
    csv('13_PAIRED_COMPARISONS.csv',contrasts)
    csv('14_EACH_OUTER_FOLD.csv',pd.concat(folds,ignore_index=True))
    csv('15_EACH_REPEAT.csv',pd.concat(repeats,ignore_index=True))
    assert all(sha(Path(p))==h for p,h in protocol['source_hashes'].items())
    save(OUT/'CHECKS_FINAL.json',dict(all150_outer_13500_inner_EN_fits_converged=True,all_patient_groups_disjoint=True,
        Clinical9_MSE_labels_and_old_files_unchanged=True,twelve_clinical_constructs_only=True,MSE_L2_only_forced=True,
        all_preprocessing_inner_training_only=True,fixed_grid_inner_only=True,no_final_frequency_subset_refit=True,
        fixed_OOF_conditional_intervals_only=True,independent_audit_pending=True))
    print(pd.DataFrame(performances).round(6).to_string(index=False),flush=True)
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
                print(f'Completed{completed}/100 frozen Enhanced-v1 outer tasks',flush=True)
    pred=collect(d,jobs)
    save(OUT/'progress.json',dict(stage='paired_fixed_OOF_bootstrap'))
    metrics(pred,protocol)
    save(OUT/'progress.json',dict(stage='metrics_complete_audit_report_pending'))


if __name__=='__main__':
    run()
