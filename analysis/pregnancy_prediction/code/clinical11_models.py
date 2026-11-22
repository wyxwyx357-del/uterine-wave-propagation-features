"""Frozen Clinical9 +/- MSE, with only a two-scalar PI/VFI extension."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from analysis.pregnancy_prediction.code import clinical9_models as old
from analysis.pregnancy_prediction.code.clinical11_blood_audit import BLOOD

GRID = old.GRID
FULL_MODELS = ['Clinical9','Clinical11']
MSE_MODELS = ['Clinical9','Clinical9_MSE','Clinical11','Clinical11_MSE']


def base_name(model):
    assert model in MSE_MODELS
    return model.replace('Clinical11','Clinical9')


def pipeline(model,C):
    result=old.pipeline(base_name(model),C)
    if model.startswith('Clinical11'):
        parts=list(result.named_steps['preprocess'].transformers)
        result.named_steps['preprocess'].set_params(transformers=parts+[('blood',StandardScaler(),BLOOD)])
    return result


def columns(model):
    return old.specification(base_name(model))[1]+(BLOOD if model.startswith('Clinical11') else [])


def state(fitted,train,model):
    parameters,names=old.preprocessing_state(fitted,train,base_name(model))
    if model.startswith('Clinical11'):
        scaler=fitted.named_steps['preprocess'].named_transformers_['blood']
        assert train[BLOOD].notna().all().all()
        np.testing.assert_allclose(scaler.mean_,train[BLOOD].mean(axis=0))
        parameters['blood_mean'],parameters['blood_scale']=scaler.mean_.tolist(),scaler.scale_.tolist()
        parameters['blood_fields']=BLOOD
        names=names+BLOOD
    return parameters,names


def reconstruct(frame,parameters,names):
    value=old.reconstruct(frame,parameters,names)
    if 'blood_fields' in parameters:
        value=np.column_stack([value,(frame[BLOOD].to_numpy(float)-parameters['blood_mean'])/parameters['blood_scale']])
    return value


def check():
    import pandas as pd
    from scipy.special import expit
    from threadpoolctl import threadpool_limits
    from analysis.pregnancy_prediction.code.clinical9_data_audit import CORE,EXTRA,MSE
    rng=np.random.default_rng(20261122)
    train=pd.DataFrame(rng.normal(size=(30,9)),columns=CORE+EXTRA)
    train['blastocyst_transfer']=np.tile([0.,1.],15)
    train['endometrial_pattern_pure_C']=np.tile([1.,0.],15)
    train[BLOOD]=abs(rng.normal(size=(30,2)))
    train[MSE]=abs(rng.normal(size=30))
    train.loc[0,'bmi']=np.nan
    train.loc[1,'endometrial_pattern_pure_C']=np.nan
    train.loc[2,BLOOD[1]]=0.
    y=np.tile([0,1],15)
    with threadpool_limits(limits=1):
        for model in MSE_MODELS:
            fitted=pipeline(model,.1).fit(train[columns(model)],y)
            parameters,names=state(fitted,train,model)
            future=train.iloc[:3].copy()
            future[BLOOD]=1e6
            beta=fitted.named_steps['model'].coef_.copy()
            p=fitted.predict_proba(future[columns(model)])[:,1]
            np.testing.assert_allclose(p,expit(reconstruct(future,parameters,names)@beta[0]+fitted.named_steps['model'].intercept_[0]))
            assert parameters==state(fitted,train,model)[0]
            np.testing.assert_array_equal(beta,fitted.named_steps['model'].coef_)
            assert len(names)==(9 if model=='Clinical9' else 10 if model=='Clinical9_MSE' else 11 if model=='Clinical11' else 12)
            if model.startswith('Clinical9'):
                reference=old.pipeline(model,.1).fit(train[columns(model)],y)
                np.testing.assert_array_equal(fitted.predict_proba(train[columns(model)]),reference.predict_proba(train[columns(model)]))
    return dict(old_Clinical9_pipeline_exactly_preserved=True, PI_VFI_only_extension=True,
        training_only_blood_scaling_checked=True, independent_prediction_reconstruction=True,
        source_zero_VFI_retained=True, heldout_does_not_change_state=True)


if __name__=='__main__':
    print(check())
