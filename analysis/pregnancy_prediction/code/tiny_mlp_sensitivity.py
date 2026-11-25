"""One fixed Tiny MLP model-form sensitivity on unchanged MSE185 grouped folds."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import json
import platform
import sys
import time
import numpy as np
import pandas as pd
from analysis.pregnancy_prediction.code.clinical9_data_audit import ROOT,OLD,sha,save,MSE
from analysis.pregnancy_prediction.code.frozen_foldwise_auc import evaluate_frozen_scores
from analysis.pregnancy_prediction.code import tiny_mlp_models as nn

PRIOR=ROOT/'outputs/clinical9_frozen_mse_20261122'
OUT=ROOT/'outputs/tiny_mlp_model_form_sensitivity_20261122'
MODELS=['Ridge_Clinical9','Ridge_Clinical9_MSE','MLP_Clinical9','MLP_Clinical9_MSE']
SEED,B=20261120,10000
_DATA=None


def csv(name,rows):
    pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')


def read_data():
    fields=['exam_id','patient_id','outcome','model_identity_eligible']+nn.FEATURES+[MSE]
    return pd.read_csv(PRIOR/'04_MASTER_CLINICAL9_LOCAL_311.csv',usecols=fields,dtype={'exam_id':str,'patient_id':str})


def fit_job(job):
    global _DATA
    started=time.perf_counter()
    if _DATA is None:
        _DATA=read_data()
    train,test=_DATA.iloc[job['train']],_DATA.iloc[job['test']]
    assert not set(train.patient_id)&set(test.patient_id)
    assert train[MSE].notna().all() and test[MSE].notna().all()
    assignment=np.asarray(job['inner_assignment'])
    result=[]
    for model in ['MLP_Clinical9','MLP_Clinical9_MSE']:
        with_MSE=model.endswith('_MSE')
        inner_cache,inner_states=[],[]
        for fold in range(1,6):
            a,b=np.flatnonzero(assignment!=fold),np.flatnonzero(assignment==fold)
            tr,va=train.iloc[a],train.iloc[b]
            assert not set(tr.patient_id)&set(va.patient_id)
            assert tr.outcome.nunique()==va.outcome.nunique()==2
            X,V,state=nn.preprocess(tr,va,with_MSE)
            inner_cache.append((X,tr.outcome.to_numpy(int),V,va.outcome.to_numpy(int)))
            inner_states.append(dict(inner_fold=fold,train_indices=a.tolist(),validation_indices=b.tolist(),preprocessing=state,patient_overlap=0))
        X,T,state=nn.preprocess(train,test,with_MSE)
        seeds=[]
        for seed in nn.SEEDS:
            inner_training=[]
            for fold,(ix,iy,vx,vy) in enumerate(inner_cache,1):
                _,training=nn.train(ix,iy,seed,nn.MAX_EPOCHS,validation=(vx,vy))
                inner_training.append(dict(inner_fold=fold,**training))
            best_epochs=[r['best_epoch'] for r in inner_training]
            epochs=int(np.median(best_epochs))
            fitted,training=nn.train(X,train.outcome.to_numpy(int),seed,epochs,validation=None)
            probability=nn.predict(fitted,T)
            assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
            seeds.append(dict(seed=seed,inner_best_epochs=best_epochs,selected_outer_epochs=epochs,
                inner_training=inner_training,outer_training=training,weights=nn.weights(fitted),
                outer_probabilities=probability.astype(float).tolist()))
        probability=np.mean(np.asarray([r['outer_probabilities'] for r in seeds],float),axis=0)
        result.append(dict(model=model,feature_order=state['feature_order'],preprocessing=state,
            inner_preprocessing=inner_states,seeds=seeds,classes=[0,1],outer_probability_mean3=probability.tolist()))
    save(OUT/'jobs'/(job['key']+'.json'),dict(**job,results=result,elapsed_seconds=time.perf_counter()-started))
    return job['key']


def prepare():
    OUT.mkdir(exist_ok=False)
    (OUT/'jobs').mkdir()
    checks=nn.verification()
    d=read_data()
    all_jobs=json.loads((PRIOR/'FIT_JOBS_LOCAL.json').read_text(encoding='utf-8'))
    jobs=[j for j in all_jobs if j['cohort']=='MSE185']
    assert len(jobs)==50
    indices=sorted(set(i for j in jobs for i in j['test']))
    sub=d.iloc[indices]
    assert len(sub)==185 and sub.patient_id.nunique()==179 and sub[MSE].notna().all()
    assert sub.outcome.eq(1).sum()==121 and sub.outcome.eq(0).sum()==64
    for job in jobs:
        train,test=d.iloc[job['train']],d.iloc[job['test']]
        assert not set(train.patient_id)&set(test.patient_id)
        for f in range(1,6):
            a=np.asarray(job['inner_assignment'])!=f
            assert not set(train.iloc[np.flatnonzero(a)].patient_id)&set(train.iloc[np.flatnonzero(~a)].patient_id)
    csv('01_COHORT_LOCAL.csv',sub[['exam_id','patient_id','outcome']])
    csv('02_FEATURE_AVAILABILITY_BEFORE_FIT.csv',[dict(feature=k,n_exams=185,n_available=int(sub[k].notna().sum()),
        n_missing=int(sub[k].isna().sum()),missing_rate=float(sub[k].isna().mean())) for k in nn.FEATURES+[MSE]])
    paths=[PRIOR/'04_MASTER_CLINICAL9_LOCAL_311.csv',PRIOR/'FIT_JOBS_LOCAL.json',PRIOR/'12_OOF_PREDICTIONS_LOCAL.csv',
        PRIOR/'16_MODEL_PERFORMANCE.csv',PRIOR/'17_PAIRED_COMPARISONS.csv',PRIOR/'STOP_RULE.json',
        OLD/'02a_GLOBAL_PATIENT_FOLDS.csv',OLD/'02b_GLOBAL_INNER_VALIDATION_FOLDS.csv',
        ROOT/'outputs/enhanced_clinical_v1_20261122/STOP_RULE.json',
        ROOT/'outputs/static_elasticnet_final_exploration_20261122/STOP_RULE.json',
        ROOT/'outputs/clinical11_PI_VFI_mse_sensitivity_20261122/STOP_RULE.json']
    scripts=[Path(__file__),Path(nn.__file__),Path(__file__).with_name('frozen_foldwise_auc.py')]
    protocol=dict(name='Tiny MLP nonlinear model-form sensitivity',version='one_fixed_tiny_MLP_20261122',
        frozen_UTC=datetime.now(timezone.utc).isoformat(),primary_model_remains='Ridge Logistic',
        scope='model-form sensitivity analysis only, not independent validation or model replacement',
        cohort=dict(exams=185,patients=179,repeated_patient_groups=6,pregnancy_n=121,non_pregnancy_n=64),
        full306_optional_not_part_of_this_core_analysis=True,clinical9_input_order=nn.FEATURES,MSE_appended_last=MSE,
        architectures={'MLP_Clinical9':[9,8,1],'MLP_Clinical9_MSE':[10,8,1]},
        hidden='Linear,ReLU,Dropout0.20',output='single logit; sigmoid only for prediction',
        dtype='CPU float32; deterministic algorithms; torch threads/inter-op1',
        optimizer=dict(name='AdamW',learning_rate=1e-3,weight_decay=1e-4,betas=[.9,.999],eps=1e-8,
            amsgrad=False,foreach=False,fused=False,all_parameters_including_bias_decay=True),
        loss='BCEWithLogitsLoss reduction mean, no class weights or other augmentation',
        batch_size=32,shuffle='torch.randperm CPU generator seeded same initialization seed, reshuffle every epoch, no dropped tail',
        maximum_epochs=200,patience=20,minimum_delta=0,
        epoch_rule='for EACH seed/model/outer fold train5 inner models; strict validation-BCE improvement, earliest exact tie; median5 best epochs; fresh full outer-training refit for exactly that count',
        no_untrained_epoch0_candidate=True,no_inner_weights_reused_for_outer_refit=True,no_outer_test_early_stopping=True,
        validation_mode='eval dropout off; training dropout on',initialization='torch.nn.Linear default initialization, fixed torch version',
        seeds=nn.SEEDS,seed_rule='same three seeds for corresponding Clinical9 and +MSE in every inner/outer split, no derived or added seeds',
        prediction='arithmetic mean of three outer-test probabilities, not logits; no seed performance selection',
        preprocessing='inner-training-only continuous median/zscore, fixed binary stage/pureC mode/unscaled; MSE observed-only training zscore; new outer-training preprocessing for final refit',
        patient_splits='reuse exact original global patient-grouped outer5/inner5/10repeats on same185',
        no_additional_inputs=True,no_architecture_optimizer_or_hyperparameter_search=True,
        Ridge_controls='copy unchanged Clinical9 and Clinical9_MSE same185 OOF',
        contrasts=[['MLP_Clinical9_MSE','MLP_Clinical9'],['MLP_Clinical9','Ridge_Clinical9'],
            ['MLP_Clinical9_MSE','Ridge_Clinical9_MSE'],['Ridge_Clinical9_MSE','Ridge_Clinical9']],
        paired_increment_difference='MLP MSE increment minus Ridge MSE increment, same joint bootstrap draws',
        evaluation='within-fold AUC, positive-negative-pair weighted within repeat, then mean10; Brier/logloss exam/repeat mean',
        bootstrap=dict(n=B,seed=SEED,unit='patient clusters',scope='fixed OOF paired conditional95%, no full workflow retraining/prior search uncertainty'),
        stop_rule='After this single Tiny MLP sensitivity stop model-form search: no bigger MLP/layers/nodes/dropout/lr/optimizer/activation/seed changes, no SVM/RF/XGBoost/CatBoost/attention/gating/transformer. Ridge stays primary.',
        verification=checks,environment=dict(Python=sys.version,platform=platform.platform(),torch=nn.torch.__version__,numpy=np.__version__,pandas=pd.__version__),
        source_hashes={str(p):sha(p) for p in paths},script_hashes={str(p):sha(p) for p in scripts},
        references=['https://docs.pytorch.org/docs/2.6/notes/randomness.html',
            'https://github.com/pytorch/pytorch/blob/v2.6.0/torch/optim/adamw.py',
            'https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/loss.py'])
    save(OUT/'FIT_JOBS_LOCAL.json',jobs)
    save(OUT/'PROTOCOL_BEFORE_FIT.json',protocol)
    save(OUT/'STOP_RULE.json',dict(status='fixed_Tiny_MLP_sensitivity_running',rule=protocol['stop_rule']))
    return d,jobs,protocol


def collect(d,jobs):
    prior=pd.read_csv(PRIOR/'12_OOF_PREDICTIONS_LOCAL.csv',dtype={'exam_id':str,'patient_id':str})
    prior=prior[prior.cohort.eq('MSE185')&prior.model.isin(['Clinical9','Clinical9_MSE'])].copy()
    prior['model']='Ridge_'+prior.model
    prior=prior[['model','repeat','fold','exam_id','patient_id','outcome','predicted_probability']]
    rows,epochs,early,seed_predictions=[],[],[],[]
    for job in jobs:
        saved=json.loads((OUT/'jobs'/(job['key']+'.json')).read_text(encoding='utf-8'))
        for fit in saved['results']:
            for index,p in zip(job['test'],fit['outer_probability_mean3']):
                r=d.iloc[index]
                rows.append(dict(model=fit['model'],repeat=job['repeat'],fold=job['fold'],exam_id=r.exam_id,
                    patient_id=r.patient_id,outcome=int(r.outcome),predicted_probability=p))
            for fitted_seed in fit['seeds']:
                seed=fitted_seed['seed']
                epochs.append(dict(model=fit['model'],repeat=job['repeat'],fold=job['fold'],seed=seed,
                    inner_best_epochs=json.dumps(fitted_seed['inner_best_epochs']),selected_outer_epochs=fitted_seed['selected_outer_epochs'],
                    final_epochs_run=fitted_seed['outer_training']['epochs_run'],parameter_count=fitted_seed['outer_training']['parameter_count']))
                for inner in fitted_seed['inner_training']:
                    early.append(dict(model=fit['model'],repeat=job['repeat'],outer_fold=job['fold'],seed=seed,
                        inner_fold=inner['inner_fold'],best_epoch=inner['best_epoch'],epochs_run=inner['epochs_run'],
                        best_validation_BCE=inner['best_validation_BCE']))
                for index,p in zip(job['test'],fitted_seed['outer_probabilities']):
                    r=d.iloc[index]
                    seed_predictions.append(dict(model=fit['model'],repeat=job['repeat'],fold=job['fold'],seed=seed,
                        exam_id=r.exam_id,patient_id=r.patient_id,outcome=int(r.outcome),predicted_probability=p))
    pred=pd.concat([prior,pd.DataFrame(rows)],ignore_index=True)
    assert len(pred)==4*185*10 and not pred.duplicated(['model','repeat','exam_id']).any()
    assert pred.groupby(['model','exam_id']).size().eq(10).all()
    assert len(epochs)==300 and len(early)==1500
    csv('03_OOF_PREDICTIONS_LOCAL.csv',pred)
    csv('04_OUTER_EPOCHS_AND_SEEDS.csv',epochs)
    csv('05_INNER_EARLY_STOPPING.csv',early)
    csv('06_SEED_PREDICTIONS_LOCAL.csv',seed_predictions)
    return pred


def metrics(pred,protocol):
    wide=pred.pivot(index=['repeat','exam_id','patient_id','outcome','fold'],columns='model',values='predicted_probability').reset_index()
    assert len(wide)==1850 and wide[MODELS].notna().all().all()
    assert wide.groupby(['repeat','patient_id']).fold.nunique().eq(1).all()
    result=evaluate_frozen_scores(wide,MODELS,B,SEED)
    ci=np.quantile(result['draws'],[.025,.975],axis=0)
    performance=[]
    def probability_scores(y,p):
        p=np.clip(np.asarray(p,float),np.finfo(float).eps,1-np.finfo(float).eps)
        y=np.asarray(y,int)
        return dict(Brier=float(np.mean((p-y)**2)),logloss=float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))))
    for i,model in enumerate(MODELS):
        performance.append(dict(model=model,classifier='Tiny MLP' if model.startswith('MLP') else 'Ridge Logistic',
            n_exams=185,n_unique_patients=179,repeated_patient_groups=6,pregnancy_n=121,non_pregnancy_n=64,
            AUC=float(result['points'][i]),CI_low=float(ci[0,i]),CI_high=float(ci[1,i]),
            **probability_scores(wide.outcome,wide[model])))
    perf=pd.DataFrame(performance).set_index('model')
    roles=['primary_MLP_MSE_increment','classifier_change_Clinical9','classifier_change_Clinical9_MSE','historical_Ridge_MSE_increment']
    contrasts=[]
    for (augmented,baseline),role in zip(protocol['contrasts'],roles):
        i,j=MODELS.index(augmented),MODELS.index(baseline)
        delta=result['draws'][:,i]-result['draws'][:,j]
        low,high=np.quantile(delta,[.025,.975])
        contrasts.append(dict(role=role,augmented=augmented,baseline=baseline,n_exams=185,n_unique_patients=179,
            Delta_AUC=float(result['points'][i]-result['points'][j]),CI_low=float(low),CI_high=float(high),crosses_zero=bool(low<=0<=high),
            Delta_Brier=float(perf.loc[augmented,'Brier']-perf.loc[baseline,'Brier']),
            Delta_logloss=float(perf.loc[augmented,'logloss']-perf.loc[baseline,'logloss'])))
    diff=lambda value:value[...,3]-value[...,2]-value[...,1]+value[...,0]
    low,high=np.quantile(diff(result['draws']),[.025,.975])
    csv('09_PAIRED_INCREMENT_DIFFERENCE.csv',[dict(MLP_minus_Ridge_MSE_increment=float(diff(result['points'])),
        CI_low=float(low),CI_high=float(high),crosses_zero=bool(low<=0<=high))])
    csv('07_MODEL_PERFORMANCE.csv',performance)
    csv('08_PAIRED_COMPARISONS.csv',contrasts)
    fold_scores=[]
    for keys,q in pred.groupby(['repeat','fold','model']):
        fold_scores.append(dict(repeat=keys[0],fold=keys[1],model=keys[2],**probability_scores(q.outcome,q.predicted_probability)))
    fold=result['folds'].merge(pd.DataFrame(fold_scores),on=['repeat','fold','model'],validate='one_to_one')
    csv('10_EACH_OUTER_FOLD.csv',fold)
    repeat_scores=[]
    for keys,q in pred.groupby(['repeat','model']):
        repeat_scores.append(dict(repeat=keys[0],model=keys[1],**probability_scores(q.outcome,q.predicted_probability)))
    repeat=result['repeats'].merge(pd.DataFrame(repeat_scores),on=['repeat','model'],validate='one_to_one')
    csv('11_EACH_REPEAT.csv',repeat)
    for table,field,index,name in [(fold,'AUC',['repeat','fold'],'12a_EACH_FOLD_MSE_INCREMENTS.csv'),
        (repeat,'weighted_fold_AUC',['repeat'],'12b_EACH_REPEAT_MSE_INCREMENTS.csv')]:
        v=table.pivot(index=index,columns='model',values=field).reset_index()
        v['MLP_MSE_increment']=v.MLP_Clinical9_MSE-v.MLP_Clinical9
        v['Ridge_MSE_increment']=v.Ridge_Clinical9_MSE-v.Ridge_Clinical9
        csv(name,v)
    csv('13_BOOTSTRAP_DRAWS.csv',pd.DataFrame(result['draws'],columns=MODELS))
    assert all(sha(Path(p))==h for p,h in protocol['source_hashes'].items())
    save(OUT/'CHECKS_FINAL.json',dict(all300_outer_1500_inner_MLP_fits_completed=True,all_seed_median_epochs_inner_only=True,
        no_outer_test_training_or_selection=True,same_patients_folds_seeds_all_MLP_pairs=True,
        all_preprocessing_training_only=True,three_probabilities_averaged_no_seed_selection=True,
        old_source_predictions_stops_unchanged=True,Ridge_remains_primary=True,independent_audit_pending=True))
    print(pd.DataFrame(performance).round(6).to_string(index=False),flush=True)
    print(pd.DataFrame(contrasts).round(6).to_string(index=False),flush=True)


def run():
    d,jobs,protocol=prepare()
    save(OUT/'progress.json',dict(stage='fitting',completed=0,total=50))
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(fit_job,job) for job in jobs]
        for completed,future in enumerate(as_completed(futures),1):
            future.result()
            save(OUT/'progress.json',dict(stage='fitting',completed=completed,total=50))
            if completed%5==0:
                print(f'Completed{completed}/50 fixed Tiny MLP outer tasks',flush=True)
    pred=collect(d,jobs)
    save(OUT/'progress.json',dict(stage='paired_fixed_OOF_bootstrap'))
    metrics(pred,protocol)
    save(OUT/'progress.json',dict(stage='metrics_complete_audit_report_pending'))


if __name__=='__main__':
    run()
