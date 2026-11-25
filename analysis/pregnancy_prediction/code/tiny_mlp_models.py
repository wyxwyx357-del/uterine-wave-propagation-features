"""One fixed CPU Tiny MLP; patient-group CV orchestration lives in separate module."""
import random
import numpy as np
import torch
from torch import nn

MSE='record_MSE_time_equal'
FEATURES=['age','bmi','infertility_years','transfer_embryo_count_component','blastocyst_transfer',
          'endometrial_thickness_mm','endometrial_volume_ml','endometrial_pattern_pure_C','days_before_transfer']
BINARY=['blastocyst_transfer','endometrial_pattern_pure_C']
CONTINUOUS=[k for k in FEATURES if k not in BINARY]
SEEDS=[20261122,20261123,20261124]
MAX_EPOCHS,PATIENCE,BATCH_SIZE=200,20,32
CONFIGURED=False


def configure():
    global CONFIGURED
    if not CONFIGURED:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        CONFIGURED=True


def preprocess(train,test,with_MSE):
    names=FEATURES+([MSE] if with_MSE else [])
    numeric=CONTINUOUS+([MSE] if with_MSE else [])
    if with_MSE:
        assert train[MSE].notna().all() and test[MSE].notna().all()
    raw=train[numeric].to_numpy(float)
    medians=np.nanmedian(raw,axis=0)
    filled=np.where(np.isnan(raw),medians,raw)
    scales=filled.std(axis=0)
    state=dict(n_fit=len(train),feature_order=names,numeric_names=numeric,medians=medians.tolist(),
        means=filled.mean(axis=0).tolist(),scales=np.where(scales==0,1,scales).tolist(),
        binary_names=BINARY,binary_modes=[float(train[k].mode().iloc[0]) for k in BINARY],
        binary_unscaled=True,MSE_observed_only=bool(with_MSE))
    return transform(train,state),transform(test,state),state


def transform(frame,state):
    values={}
    for k,median,mean,scale in zip(state['numeric_names'],state['medians'],state['means'],state['scales']):
        x=frame[k].to_numpy(float)
        values[k]=(np.where(np.isnan(x),median,x)-mean)/scale
    for k,mode in zip(state['binary_names'],state['binary_modes']):
        x=frame[k].to_numpy(float)
        values[k]=np.where(np.isnan(x),mode,x)
    result=np.column_stack([values[k] for k in state['feature_order']]).astype(np.float32)
    assert np.isfinite(result).all()
    return result


class TinyMLP(nn.Module):
    def __init__(self,n_inputs):
        super().__init__()
        assert n_inputs in [9,10]
        self.hidden=nn.Linear(n_inputs,8)
        self.activation=nn.ReLU()
        self.dropout=nn.Dropout(.20)
        self.output=nn.Linear(8,1)

    def forward(self,x):
        return self.output(self.dropout(self.activation(self.hidden(x)))).squeeze(-1)


def weights(model):
    return {k:v.detach().cpu().numpy().tolist() for k,v in model.state_dict().items()}


def numpy_probability(X,state):
    x=np.asarray(X,np.float32)
    w={k:np.asarray(v,np.float32) for k,v in state.items()}
    h=np.maximum(x@w['hidden.weight'].T+w['hidden.bias'],np.float32(0))
    logits=(h@w['output.weight'].T+w['output.bias']).ravel()
    from scipy.special import expit
    return expit(logits)


def predict(model,X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(np.asarray(X,np.float32)))).cpu().numpy()


def train(X,y,seed,epochs,validation=None):
    configure()
    assert seed in SEEDS and 1<=epochs<=MAX_EPOCHS
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model=TinyMLP(X.shape[1]).float().cpu()
    criterion=nn.BCEWithLogitsLoss(reduction='mean')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4,
        betas=(.9,.999),eps=1e-8,amsgrad=False,foreach=False,fused=False)
    x=torch.from_numpy(np.asarray(X,np.float32))
    target=torch.from_numpy(np.asarray(y,np.float32))
    permutation=torch.Generator(device='cpu').manual_seed(seed)
    if validation is not None:
        vx=torch.from_numpy(np.asarray(validation[0],np.float32))
        vy=torch.from_numpy(np.asarray(validation[1],np.float32))
    training_curve,validation_curve=[],[]
    best_loss,best_epoch,stale=float('inf'),None,0
    for epoch in range(1,epochs+1):
        model.train()
        order=torch.randperm(len(x),generator=permutation)
        cumulative=0.
        for index in order.split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(x[index]),target[index])
            assert bool(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            cumulative+=float(loss.detach())*len(index)
        training_curve.append(cumulative/len(x))
        model.eval()
        if validation is not None:
            with torch.no_grad():
                value=float(criterion(model(vx),vy))
            assert np.isfinite(value)
            validation_curve.append(value)
            if value<best_loss:
                best_loss,best_epoch,stale=value,epoch,0
            else:
                stale+=1
            if stale>=PATIENCE:
                break
    model.eval()
    assert all(bool(torch.isfinite(v).all()) for v in model.parameters())
    record=dict(seed=seed,epochs_run=epoch,training_minibatch_BCE_curve=training_curve,
        validation_BCE_curve=validation_curve,best_epoch=best_epoch,
        best_validation_BCE=None if best_epoch is None else best_loss,
        used_inner_validation=validation is not None,parameter_count=sum(v.numel() for v in model.parameters()))
    return model,record


def verification():
    import pandas as pd
    rng=np.random.default_rng(20261122)
    frame=pd.DataFrame(rng.normal(size=(45,9)),columns=FEATURES)
    frame['blastocyst_transfer']=np.resize([0.,1.],45)
    frame['endometrial_pattern_pure_C']=np.resize([1.,0.],45)
    frame.loc[0,'bmi']=np.nan
    frame.loc[1,'endometrial_pattern_pure_C']=np.nan
    frame[MSE]=rng.normal(size=45)
    y=np.resize([0,1],45)
    X,V,state=preprocess(frame.iloc[:32],frame.iloc[32:],False)
    model,record=train(X,y[:32],SEEDS[0],8,(V,y[32:]))
    second,second_record=train(X,y[:32],SEEDS[0],8,(V,y[32:]))
    assert record==second_record and weights(model)==weights(second)
    np.testing.assert_allclose(predict(model,V),numpy_probability(V,weights(model)),atol=2e-7,rtol=2e-7)
    assert record['best_epoch']==int(np.argmin(record['validation_BCE_curve'])+1)
    assert record['parameter_count']==89
    np.testing.assert_array_equal(predict(model,V),predict(model,V))
    zero=np.zeros((32,9),np.float32)
    _,early=train(zero,np.zeros(32),SEEDS[2],MAX_EPOCHS,(zero,np.ones(32)))
    assert early['epochs_run']<MAX_EPOCHS and early['epochs_run']==early['best_epoch']+PATIENCE
    fixed,fixed_record=train(X,y[:32],SEEDS[0],int(record['best_epoch']))
    assert fixed_record['epochs_run']==record['best_epoch'] and not fixed_record['used_inner_validation']
    before=weights(fixed)
    future=frame.iloc[32:].copy()
    future['age']=1e6
    transform(future,state)
    assert weights(fixed)==before
    X,V,state=preprocess(frame.iloc[:32],frame.iloc[32:],True)
    augmented,aug_record=train(X,y[:32],SEEDS[1],3)
    assert aug_record['parameter_count']==97 and X.shape[1]==10
    return dict(same_seed_training_reproducible=True,eval_logits_independently_reconstructed=True,
        best_epoch_uses_validation_BCE_only=True,final_refit_has_no_validation_or_test_selection=True,
        patience20_and_eval_dropout_off_verified=True,
        train_only_missing_and_scale=True,net_parameters_Clinical9=89,net_parameters_Clinical9_MSE=97,
        fixed_CPU_float32=True,torch_version=torch.__version__)


if __name__=='__main__':
    print(verification())
