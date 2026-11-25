"""Clinical-only block EN; optional compulsory MSE receives L2, never L1."""
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from analysis.pregnancy_prediction.code.clinical9_data_audit import NUM9,MSE
from analysis.pregnancy_prediction.code.enhanced_clinical_audit import BLOOD
from analysis.pregnancy_prediction.code import block_elasticnet as solver

CONTINUOUS=NUM9+BLOOD
BINARY=['blastocyst_transfer','endometrial_pattern_pure_C']
POSITION=['uterine_position_posterior','uterine_position_horizontal']
FEATURES=CONTINUOUS+BINARY+POSITION
CONSTRUCTS=CONTINUOUS+BINARY+['uterine_position']
ALPHAS=[.9,.5,.1]
LAMBDAS=[1000.,100.,10.,1.,.1,.01]


def raw_design(frame,state):
    x=frame[CONTINUOUS].to_numpy(float)
    x=np.where(np.isnan(x),state['medians'],x)
    binary=frame[BINARY].to_numpy(float)
    binary=np.where(np.isnan(binary),state['binary_modes'],binary)
    position=frame.uterine_position_clean.fillna(state['position_mode'])
    return np.column_stack([x,binary,position.eq('posterior').astype(float),position.eq('horizontal').astype(float)])


def transform(frame,state):
    z=(raw_design(frame,state)-state['clinical_means'])/state['clinical_scales']
    if state['with_MSE']:
        assert frame[MSE].notna().all()
        z=np.column_stack([(frame[MSE].to_numpy(float)-state['MSE_mean'])/state['MSE_scale'],z])
    return z


def preprocess(train,test,with_MSE):
    assert set(train.uterine_position_clean.dropna())<={'anterior','posterior','horizontal'}
    state=dict(n_fit=len(train),continuous_names=CONTINUOUS,medians=np.nanmedian(train[CONTINUOUS],axis=0).tolist(),
        binary_names=BINARY,binary_modes=[float(train[k].mode().iloc[0]) for k in BINARY],
        position_mode=train.uterine_position_clean.mode().iloc[0],position_reference='anterior',
        clinical_column_names=FEATURES,with_MSE=bool(with_MSE))
    raw=raw_design(train,state)
    assert np.isfinite(raw).all()
    scale=StandardScaler().fit(raw)
    state['clinical_means'],state['clinical_scales']=scale.mean_.tolist(),scale.scale_.tolist()
    if with_MSE:
        assert train[MSE].notna().all()
        ms=StandardScaler().fit(train[[MSE]])
        state['MSE_mean'],state['MSE_scale']=float(ms.mean_[0]),float(ms.scale_[0])
    return transform(train,state),transform(test,state),state


def fit(X,y,with_MSE,alpha,lam,warm=None):
    # Generic solver's historical "static" block is used for these13 clinical
    # columns only. No radiomics data or method search is loaded/reopened.
    result=solver.fit(X,y,int(with_MSE),alpha,lam,warm)
    result['selected_clinical_columns']=result.pop('selected_static')
    beta=np.asarray(result['coef'][int(with_MSE):])
    result['selected_clinical_constructs']=int(np.count_nonzero(beta[:11]))+int(np.any(beta[11:]!=0))
    return result


def verify():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    rng=np.random.default_rng(20261122)
    frame=pd.DataFrame(rng.normal(size=(120,9)),columns=CONTINUOUS)
    frame['blastocyst_transfer']=np.tile([0.,1.],60)
    frame['endometrial_pattern_pure_C']=np.tile([1.,0.],60)
    frame['uterine_position_clean']=np.tile(['anterior','posterior','horizontal'],40)
    frame[MSE]=rng.normal(size=120)
    frame.loc[0,'PI_clean']=np.nan
    frame.loc[1,'endometrial_pattern_pure_C']=np.nan
    y=(rng.random(120)<expit(.1+frame[MSE].to_numpy())).astype(int)
    with threadpool_limits(limits=1):
        X,T,s=preprocess(frame,frame.iloc[:4],False)
        out=fit(X,y,False,.5,2.)
        reference=LogisticRegression(C=.5,penalty='elasticnet',l1_ratio=.5,solver='saga',tol=1e-11,max_iter=100000,random_state=20261122).fit(X,y)
        np.testing.assert_allclose(solver.predict(out,X),reference.predict_proba(X)[:,1],atol=5e-6,rtol=5e-6)
        X,T,s=preprocess(frame,frame.iloc[:4],True)
        fixed=fit(X,y,True,.9,1000.)
        assert fixed['selected_clinical_columns']==0 and fixed['coef'][0]!=0
        ridge=LogisticRegression(C=.001,penalty='l2',solver='lbfgs',tol=1e-11,max_iter=10000).fit(X[:,:1],y)
        np.testing.assert_allclose(solver.predict(fixed,X),ridge.predict_proba(X[:,:1])[:,1],atol=3e-6,rtol=3e-6)
        before=np.asarray(fixed['coef']).copy()
        future=frame.iloc[:4].copy()
        future['age']=1e6
        transform(future,s)
        np.testing.assert_array_equal(before,fixed['coef'])
    return dict(clinical_EN_matches_sklearn_on_synthetic=True, compulsory_MSE_not_L1_selected=True,
        zero_clinical_allowed=True,MSE_only_limit_matches_Ridge=True,train_only_preprocessing=True,
        thirteen_columns_twelve_constructs=True,solver_KKT_tolerance=solver.KKT_TOLERANCE)


if __name__=='__main__':
    print(verify())
