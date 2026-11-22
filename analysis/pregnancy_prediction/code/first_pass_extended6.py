"""First pre-specified Extended6 internal validation; no feature/model search.

Only Ridge C is tuned in patient-grouped nested5x5 CV. Imaging is never imputed.
Reads an explicit whitelist; historical frequency fields are not loaded.
"""
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT = Path(os.environ.get('DATA_ROOT', 'data'))
OUT=ROOT/'outputs/auc_first_pass_extended6_globalfolds_20261120'
SOURCE=ROOT/'outputs/extended6_increment_protocol_20261120/MASTER_CLINICAL_SOURCE_ALIGNED_311.csv'
REQUEST = Path(os.environ.get('REQUEST_TEXT_PATH', 'data/request.txt'))
CLINICAL=['age','transfer_embryo_count_component','blastocyst_transfer','bmi','infertility_years','endometrial_thickness_mm']
NUMERIC=['age','transfer_embryo_count_component','bmi','infertility_years','endometrial_thickness_mm']
STAGE='blastocyst_transfer'
RATE='significant_propagation_activity_frequency_cpm'
IMAGING=['record_MSE_time_equal','record_CC_time_equal','anterior_radial_stretch_activity_amplitude',
         'posterior_radial_stretch_activity_amplitude',RATE]
META=['case_id','exam_id','patient_id','cohort','outcome','model_identity_eligible','video_duration_s','effective_observation_seconds']
USECOLS=META+CLINICAL+IMAGING
SEEDS=list(range(20260531,20260541))
GRID=[.001,.01,.1,1.,10.,100.]
BLOCKS=[
 ('MSE','primary',['record_MSE_time_equal']),
 ('ACT','key_secondary_exploratory',[RATE]),
 ('AMP','secondary',['anterior_radial_stretch_activity_amplitude','posterior_radial_stretch_activity_amplitude']),
 ('CC','secondary_explanatory',['record_CC_time_equal']),
 ('IMG','exploratory_multimodal_model',IMAGING[:4]),
 ('IMG_ACT','exploratory_multimodal_model',IMAGING)]
_DATA=None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,obj):
    p=Path(path);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def csv(name,frame):pd.DataFrame(frame).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def read_data():return pd.read_csv(SOURCE,usecols=USECOLS,dtype={'patient_id':str,'exam_id':str})
def pipeline(images):
    # No imputer is applied to required observed imaging.
    transform=ColumnTransformer([
        ('clinical_numeric',Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),NUMERIC),
        ('stage',SimpleImputer(strategy='most_frequent',keep_empty_features=True),[STAGE]),
        ('imaging',StandardScaler(),images)] if images else [
        ('clinical_numeric',Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),NUMERIC),
        ('stage',SimpleImputer(strategy='most_frequent',keep_empty_features=True),[STAGE])])
    return Pipeline([('preprocess',transform),('model',LogisticRegression(penalty='l2',solver='lbfgs',max_iter=10000))])

def fit_job(job):
    global _DATA
    if _DATA is None:_DATA=read_data()
    tr=_DATA.iloc[job['train']];te=_DATA.iloc[job['test']]
    images=job['images'];names=NUMERIC+[STAGE]+images
    assert not set(tr.patient_id)&set(te.patient_id)
    assert not tr[images].isna().any().any() and not te[images].isna().any().any()
    groups=tr.patient_id.to_numpy(str);y=tr.outcome.to_numpy(int)
    inner_assignment=np.asarray(job['inner_assignment'],int)
    inner=[(np.flatnonzero(inner_assignment!=fold),np.flatnonzero(inner_assignment==fold)) for fold in range(1,6)]
    inner_audit=[]
    for i,(a,b) in enumerate(inner,1):
        assert not set(groups[a])&set(groups[b])
        assert len(np.unique(y[a]))==len(np.unique(y[b]))==2
        inner_audit.append(dict(fold=i,train_n=len(a),validation_n=len(b),patient_overlap=0))
    with threadpool_limits(limits=1):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            search=GridSearchCV(pipeline(images),{'model__C':GRID},scoring='neg_log_loss',cv=inner,
                                n_jobs=1,refit=True,error_score='raise',return_train_score=False)
            search.fit(tr[names],y)
        convergence=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
        if convergence:raise RuntimeError('Ridge did not converge; retain job failure rather than silently tune settings')
        fitted=search.best_estimator_
        prob=fitted.predict_proba(te[names])[:,1]
    assert np.isfinite(prob).all() and ((prob>0)&(prob<1)).all()
    prep=fitted.named_steps['preprocess']
    clinical=prep.named_transformers_['clinical_numeric']
    imputer=clinical.named_steps['imputer'];scale=clinical.named_steps['scale']
    expected=np.nanmedian(tr[NUMERIC].to_numpy(float),axis=0)
    np.testing.assert_allclose(imputer.statistics_,expected)
    filled=np.where(np.isnan(tr[NUMERIC].to_numpy(float)),imputer.statistics_,tr[NUMERIC].to_numpy(float))
    np.testing.assert_allclose(scale.mean_,filled.mean(axis=0))
    stage_imputer=prep.named_transformers_['stage']
    if images:
        np.testing.assert_allclose(prep.named_transformers_['imaging'].mean_,tr[images].mean().to_numpy())
    result=dict(key=job['key'],train=job['train'],test=job['test'],images=images,probability=prob.tolist(),
        selected_C=float(search.best_params_['model__C']),feature_order=names,
        standardized_coefficients=fitted.named_steps['model'].coef_[0].tolist(),
        intercept=float(fitted.named_steps['model'].intercept_[0]),
        inner_scores=[dict(C=float(p['model__C']),mean_negative_log_loss=float(s),SD=float(sd))
            for p,s,sd in zip(search.cv_results_['params'],search.cv_results_['mean_test_score'],search.cv_results_['std_test_score'])],
        inner_audit=inner_audit,outer_patient_overlap=0,
        preprocessing=dict(clinical_medians=imputer.statistics_.tolist(),clinical_means=scale.mean_.tolist(),clinical_scales=scale.scale_.tolist(),
                           stage_mode=stage_imputer.statistics_.tolist(),stage_scaled=False,
                           imaging_means=prep.named_transformers_['imaging'].mean_.tolist() if images else [],
                           imaging_scales=prep.named_transformers_['imaging'].scale_.tolist() if images else []),
        converged=True,preprocessing_train_only_checked=True)
    save(OUT/'jobs'/(job['key']+'.json'),result)
    return job['key']

def activity_preflight(d):
    original=pd.read_csv(ROOT/'outputs/wave_counts_223_20261119/wave_counts.csv',
        usecols=['case_id','processing_status',RATE,'effective_observation_seconds'])
    short=pd.read_csv(ROOT/'outputs/short_activity_rate_20261120/wave_counts.csv',
        usecols=['case_id','processing_status',RATE,'effective_observation_seconds'])
    saved=pd.concat([original,short],ignore_index=True)
    assert saved.case_id.is_unique
    joined=d[['case_id',RATE,'effective_observation_seconds']].merge(saved,on='case_id',how='left',validate='one_to_one',suffixes=('_master','_counter'))
    a=joined[RATE+'_master'];b=joined[RATE+'_counter'];time=joined.effective_observation_seconds_counter
    np.testing.assert_allclose(a,b,equal_nan=True)
    observed=a.notna()
    failures=dict(negative_values=int(a.lt(0).sum()),infinite_values=int(np.isinf(a.to_numpy(float)).sum()),
                  observed_with_nonpositive_or_missing_observation_time=int((observed&(~time.gt(0))).sum()),
                  no_support_or_failed_coded_zero=int((a.eq(0)&joined.processing_status.ne('completed')).sum()),
                  observed_counter_status_not_completed=int((observed&joined.processing_status.ne('completed')).sum()))
    def describe(v):
        q=v.dropna().quantile([0,.25,.5,.75,1])
        return dict(n=int(v.notna().sum()),minimum=float(q.iloc[0]),Q1=float(q.iloc[1]),median=float(q.iloc[2]),Q3=float(q.iloc[3]),maximum=float(q.iloc[4]))
    report=dict(scope='all311 registered exams, no outcome relationship examined',rate=describe(a),
                zero_n=int(a.eq(0).sum()),zero_fraction_of_observed=float(a.eq(0).sum()/observed.sum()),
                zero_fraction_of_registry=float(a.eq(0).mean()),missing_fraction_of_registry=float(a.isna().mean()),
                observation_seconds_among_observed=describe(time[observed]),checks=failures,
                status_counts=joined.processing_status.fillna('no_counter_input').value_counts().to_dict(),
                counter_sources_sha256={'discovery':sha(ROOT/'outputs/wave_counts_223_20261119/wave_counts.csv'),
                                       'short':sha(ROOT/'outputs/short_activity_rate_20261120/wave_counts.csv')})
    save(OUT/'ACTIVITY_PREFLIGHT_NO_OUTCOME.json',report)
    if any(failures.values()):raise ValueError('Activity preflight failed; no automatic repairs or modeling allowed')

def run():
    OUT.mkdir(exist_ok=False);(OUT/'jobs').mkdir()
    d=read_data()
    assert len(d)==311 and d.case_id.is_unique and d.exam_id.is_unique
    assert set(d[STAGE].dropna().unique())<=set([0,1])
    if np.isinf(d[CLINICAL+IMAGING].to_numpy(float)).any():raise ValueError('Infinite clinical/imaging input; do not silently recode')
    activity_preflight(d)
    identity=d.model_identity_eligible.eq(True)&d.outcome.isin([0,1])
    clinical_complete=d[CLINICAL].notna().all(axis=1)
    # One outer map for all modules; each outer-training set also has one inner map.
    global_indices=np.flatnonzero(identity)
    global_y=d.iloc[global_indices].outcome.to_numpy(int)
    global_groups=d.iloc[global_indices].patient_id.to_numpy(str)
    outer_maps={};inner_maps={};global_outer_rows=[];global_inner_rows=[]
    for repeat,seed in enumerate(SEEDS):
        outer_map={}
        global_cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
        for fold,(tr,te) in enumerate(global_cv.split(np.zeros(len(global_indices)),global_y,global_groups),1):
            assert not set(global_groups[tr])&set(global_groups[te])
            for k in te:
                idx=int(global_indices[k]);outer_map[idx]=fold
                global_outer_rows.append(dict(repeat=repeat,seed=seed,fold=fold,exam_id=d.iloc[idx].exam_id,patient_id=d.iloc[idx].patient_id))
            inner_map={}
            inner_cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20260531+1000+repeat*100+fold)
            for inner_fold,(itr,iva) in enumerate(inner_cv.split(np.zeros(len(tr)),global_y[tr],global_groups[tr]),1):
                assert not set(global_groups[tr][itr])&set(global_groups[tr][iva])
                for k in iva:
                    idx=int(global_indices[tr[k]]);inner_map[idx]=inner_fold
                    global_inner_rows.append(dict(repeat=repeat,outer_fold=fold,inner_fold=inner_fold,exam_id=d.iloc[idx].exam_id,patient_id=d.iloc[idx].patient_id))
            inner_maps[(repeat,fold)]=inner_map
        outer_maps[repeat]=outer_map
    csv('02a_GLOBAL_PATIENT_FOLDS.csv',global_outer_rows)
    csv('02b_GLOBAL_INNER_VALIDATION_FOLDS.csv',global_inner_rows)
    cohorts=[];registry=[];jobmap={};links=[];assignments=[];splitrows=[]
    for name,tier,images in BLOCKS:
        image_complete=d[images].notna().all(axis=1)
        scopes=['clinical_complete_case'] if name in ['IMG','IMG_ACT'] else ['clinical_fold_imputation','clinical_complete_case']
        for scope in scopes:
            mask=identity&image_complete
            if scope=='clinical_complete_case':mask &= clinical_complete
            indices=np.flatnonzero(mask).tolist()
            sub=d.iloc[indices];y=sub.outcome.to_numpy(int);groups=sub.patient_id.to_numpy(str)
            key=name+'__'+scope
            registry.append(dict(analysis_set=key,module=name,tier=tier,clinical_missing_policy=scope,
                baseline_model='M0-'+name,extended_model='M1-'+name if name not in ['IMG','IMG_ACT'] else 'M-'+name,
                n_exams=len(sub),n_unique_patients=sub.patient_id.nunique(),pregnancy_n=int(y.sum()),non_pregnancy_n=int((y==0).sum()),
                repeated_patient_groups=int(sub.patient_id.value_counts().gt(1).sum()),
                n_with_missing_clinical=int(sub[CLINICAL].isna().any(axis=1).sum()),
                long_n=int(sub.cohort.eq('desktop_276').sum()),short_n=int(sub.cohort.eq('short_35').sum()),
                image_df=len(images),total_df=6+len(images),imaging_fields=';'.join(images)))
            for i,row in d.iterrows():
                reasons=[]
                if not identity.iloc[i]:reasons.append('unresolved_identity_or_missing_clinical_outcome')
                if not image_complete.iloc[i]:reasons.append('missing_required_imaging:'+','.join(f for f in images if pd.isna(row[f])))
                if scope=='clinical_complete_case' and not clinical_complete.iloc[i]:reasons.append('missing_clinical:'+','.join(f for f in CLINICAL if pd.isna(row[f])))
                cohorts.append(dict(analysis_set=key,case_id=row.case_id,exam_id=row.exam_id,patient_id=row.patient_id,
                    included=bool(mask.iloc[i]),inclusion_or_missing_reason='eligible_observed_imaging_and_linked_identity_outcome' if mask.iloc[i] else ';'.join(reasons),
                    n_missing_clinical=int(row[CLINICAL].isna().sum()),video_duration_s=row.video_duration_s,long_short_source=row.cohort))
            for repeat,seed in enumerate(SEEDS):
                assigned=np.array([outer_maps[repeat][idx] for idx in indices],int)
                for fold in range(1,6):
                    tr=np.flatnonzero(assigned!=fold);te=np.flatnonzero(assigned==fold)
                    assert not set(groups[tr])&set(groups[te])
                    assert len(np.unique(y[tr]))==len(np.unique(y[te]))==2
                    assigned[te]=fold
                    train=[indices[i] for i in tr];test=[indices[i] for i in te]
                    seed_inner=20260531+1000+repeat*100+fold
                    splitrows.append(dict(analysis_set=key,repeat=repeat,fold=fold,train_n=len(train),test_n=len(test),patient_overlap=0))
                    for role,model,extra in [('baseline','M0-'+name,[]),('augmented','M1-'+name if name not in ['IMG','IMG_ACT'] else 'M-'+name,images)]:
                        identity_key=dict(train=train,test=test,images=extra,inner_seed=seed_inner,grid=GRID,
                                          inner_assignment=[inner_maps[(repeat,fold)][idx] for idx in train])
                        jobkey=hashlib.sha256(json.dumps(identity_key,sort_keys=True).encode()).hexdigest()[:24]
                        jobmap[jobkey]=dict(identity_key,key=jobkey)
                        links.append(dict(analysis_set=key,module=name,scope=scope,repeat=repeat,fold=fold,role=role,model=model,job_key=jobkey))
                assert (assigned>0).all()
                for i,idx in enumerate(indices):
                    assignments.append(dict(analysis_set=key,case_id=d.iloc[idx].case_id,exam_id=d.iloc[idx].exam_id,patient_id=groups[i],
                        outcome=int(y[i]),repeat=repeat,seed=seed,fold=int(assigned[i])))
    csv('01_ANALYSIS_COHORTS.csv',cohorts);csv('ANALYSIS_SET_SUMMARY.csv',registry)
    assert pd.DataFrame(assignments).groupby(['repeat','patient_id']).fold.nunique().eq(1).all()
    csv('02_FOLD_ASSIGNMENTS.csv',assignments);csv('OUTER_SPLIT_AUDIT.csv',splitrows);csv('JOB_MODEL_LINKS_LOCAL.csv',links)
    protocol=dict(version='first_formal_internal_pass_v2_global_patient_folds_20261120',supersedes='prep inner3/first-repeat-summary; aborted per-module-fold technical run before performance assessment',
        source_master=str(SOURCE),source_master_sha256=sha(SOURCE),request_sha256=sha(REQUEST),fit_script_sha256=sha(Path(__file__)),
        clinical_fields=CLINICAL,allowed_imaging_fields=IMAGING,read_columns=USECOLS,
        source_alignment='44 duration plus1BMI policy previously source-aligned; no new changes in this pass',
        analysis_sets=registry,clinical_sensitivities='Core3 and historical7 reserved only, not fitted this pass',
        outer_folds=5,inner_folds=5,repeats=10,seeds=SEEDS,grid_C=GRID,
        fold_policy='global grouped stratified outer map from all306 linked/outcome exams; global inner map from each outer training set only; restrict these maps to each module and complete-case scope',
        metrics_script_sha256=sha(Path(__file__).with_name('first_pass_metrics.py')),
        algorithm='Ridge Logistic L2 lbfgs, unpenalized intercept, max_iter10000',
        selection='negative log loss inner only; ascending C grid, exact ties first smaller C; no grid expansion',
        preprocessing='numeric clinical: train median then train z-score; binary stage: train mode, no z-score; imaging observed only then train z-score; entire pipeline re-fitted in each inner split',
        primary_summary='mean each exam OOF probability across10 repeats, one row per exam per model',
        repetition='do not concatenate repeats as independent patients; repeat-level stability separate',
        bootstrap='2000 paired patient-cluster draws of fixed mean-OOF probabilities; no full-pipeline refit',
        scope='exploratory internally validated incremental prediction; prior outcome access disclosed',
        primary_increment='MSE',activity_role='key secondary exploratory',
        complete_case_and_fold_imputation='both for4 single-construct modules; joint exploratory strictly clinical+imaging complete only',
        subgroup='same full-cohort OOF predictions split by predefined long/short source, no subgroup refit',
        frozen_counter_and_features_unchanged=True,no_old_frequency_columns_loaded=True,no_feature_selection=True,
        fold_assignments_sha256=sha(OUT/'02_FOLD_ASSIGNMENTS.csv'),unique_fit_jobs=len(jobmap))
    save(OUT/'PROTOCOL_BEFORE_FIT.json',protocol)
    save(OUT/'FIT_JOBS_LOCAL.json',list(jobmap.values()))
    save(OUT/'progress.json',dict(stage='nested_fitting',completed=0,total=len(jobmap)))
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(fit_job,j) for j in jobmap.values()]
        for n,future in enumerate(as_completed(futures),1):
            future.result()
            save(OUT/'progress.json',dict(stage='nested_fitting',completed=n,total=len(jobmap)))
    prediction_rows=[];coef_rows=[];tuning_rows=[];preprocessing_rows=[]
    for link in links:
        result=json.loads((OUT/'jobs'/(link['job_key']+'.json')).read_text(encoding='utf-8'))
        for idx,p in zip(result['test'],result['probability']):
            row=d.iloc[idx]
            prediction_rows.append(dict(analysis_set=link['analysis_set'],module=link['module'],role=link['role'],model=link['model'],
                outcome=int(row.outcome),patient_id=row.patient_id,exam_id=row.exam_id,case_id=row.case_id,
                repeat=link['repeat'],fold=link['fold'],predicted_probability=p,video_duration_s=row.video_duration_s,long_short_source=row.cohort))
        for name,coef in zip(result['feature_order'],result['standardized_coefficients']):
            coef_rows.append(dict(analysis_set=link['analysis_set'],repeat=link['repeat'],fold=link['fold'],model=link['model'],
                feature=name,coefficient=coef,coefficient_scale='binary0/1_unscaled' if name==STAGE else 'training_SD_standardized',
                ridge_C=result['selected_C'],inverse_C_reference=1/result['selected_C'],intercept=result['intercept'],
                C_grid_boundary=result['selected_C'] in [GRID[0],GRID[-1]]))
        tuning_rows.extend(dict(analysis_set=link['analysis_set'],repeat=link['repeat'],fold=link['fold'],model=link['model'],**r) for r in result['inner_scores'])
        preprocessing_rows.append(dict(analysis_set=link['analysis_set'],repeat=link['repeat'],fold=link['fold'],model=link['model'],
                                       preprocessing=json.dumps(result['preprocessing']),train_only_assertion=True))
    pred=pd.DataFrame(prediction_rows)
    assert not pred.duplicated(['analysis_set','model','repeat','exam_id']).any()
    assert pred.groupby(['analysis_set','model','exam_id']).size().eq(10).all()
    csv('03_OOF_PREDICTIONS.csv',pred);csv('07_MODEL_COEFFICIENTS.csv',coef_rows)
    csv('08_INNER_TUNING_LOG.csv',tuning_rows);csv('09_PREPROCESSING_AUDIT.csv',preprocessing_rows)
    assert sha(SOURCE)==protocol['source_master_sha256']
    save(OUT/'FIT_CHECKS.json',dict(all_source_hashes_unchanged=True,all_outer_patient_overlaps_zero=True,
        all_inner_patient_overlaps_zero=True,preprocessing_training_only_verified=True,
        overlapping_patients_share_folds_across_all_modules=True,
        every_exam_model_has10_distinct_OOF_predictions=True,prediction_rows=len(pred),unique_fit_jobs=len(jobmap),
        all_fits_converged=True,no_old_frequency_columns_read=True,clinical_sensitivities_not_fitted=True))
    save(OUT/'progress.json',dict(stage='metrics_bootstrap',completed_fit_jobs=len(jobmap)))
    from analysis.mse_pregnancy_association.code.first_pass_metrics import summarize
    summarize(OUT)

if __name__=='__main__':run()
