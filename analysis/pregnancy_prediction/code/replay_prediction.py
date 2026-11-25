"""Replay the primary archived fits on authorized private inputs and original jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,brier_score_loss
from .frozen_foldwise_auc import evaluate_frozen_scores


def validate_jobs(data,jobs):
    jobs=[j for j in jobs if j['cohort']=='MSE185']
    assert len(jobs)==50 and {(j['repeat'],j['fold']) for j in jobs}=={(r,f) for r in range(10) for f in range(1,6)}
    indices=sorted({i for j in jobs for i in j['test']})
    cohort=data.iloc[indices]
    assert len(cohort)==185 and cohort.patient_id.nunique()==179
    assert cohort.record_MSE_time_equal.notna().all()
    assert (cohort.outcome.eq(1).sum(),cohort.outcome.eq(0).sum())==(121,64)
    for repeat in range(10):
        assert sorted(i for j in jobs if j['repeat']==repeat for i in j['test'])==indices
    for job in jobs:
        assert set(job['train'])==set(indices)-set(job['test'])
        train,test=data.iloc[job['train']],data.iloc[job['test']]
        assert not set(train.patient_id)&set(test.patient_id)
        assignment=np.asarray(job['inner_assignment'])
        assert assignment.shape==(len(train),) and set(assignment)=={1,2,3,4,5}
        for fold in range(1,6):
            a,b=train.iloc[assignment!=fold],train.iloc[assignment==fold]
            assert not set(a.patient_id)&set(b.patient_id)
            assert a.outcome.nunique()==b.outcome.nunique()==2
    return jobs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master',default=os.environ.get('INPUT_PATH'),required='INPUT_PATH' not in os.environ)
    parser.add_argument('--jobs',required=True,help='Original Clinical9 FIT_JOBS_LOCAL.json; original master row order is mandatory')
    parser.add_argument('--output',default=os.environ.get('OUTPUT_PATH'),required='OUTPUT_PATH' not in os.environ)
    parser.add_argument('--classifier',choices=['ridge','tiny_mlp'],default='ridge')
    args=parser.parse_args()
    expected=json.loads((Path(__file__).parents[1]/'PRIVATE_INPUT_SHA256.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(Path(args.master).read_bytes()).hexdigest()==expected['canonical_master_SHA256'], 'Private master bytes/order differ from the frozen canonical input.'
    assert hashlib.sha256(Path(args.jobs).read_bytes()).hexdigest()==expected['original_jobs_SHA256'], 'Use the exact original frozen patient-grouped jobs.'
    data=pd.read_csv(args.master,dtype={'exam_id':str,'patient_id':str})
    jobs=validate_jobs(data,json.loads(Path(args.jobs).read_text(encoding='utf-8-sig')))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);(out/'jobs').mkdir()
    if args.classifier=='ridge':
        from . import clinical9_frozen_mse as engine
        # Run only the archived primary pair, not the archived position sensitivity.
        engine.MSE_MODELS=['Clinical9','Clinical9_MSE']
        probability_key='predictions'
    else:
        from . import tiny_mlp_sensitivity as engine
        engine.nn.configure()
        probability_key='outer_probability_mean3'
    engine.OUT=out;engine._DATA=data
    rows=[]
    for job in jobs:
        engine.fit_job(job)
        result=json.loads((out/'jobs'/(job['key']+'.json')).read_text(encoding='utf-8'))
        for fitted in result['results']:
            for index,p in zip(job['test'],fitted[probability_key]):
                record=data.iloc[index]
                rows.append(dict(repeat=job['repeat'],fold=job['fold'],model=fitted['model'],
                    exam_id=record.exam_id,patient_id=record.patient_id,outcome=int(record.outcome),predicted_probability=p))
    scores=pd.DataFrame(rows)
    scores.to_csv(out/'PRIVATE_OOF.csv',index=False)
    wide=scores.pivot(index=['repeat','fold','exam_id','patient_id','outcome'],columns='model',values='predicted_probability').reset_index()
    names=['Clinical9','Clinical9_MSE'] if args.classifier=='ridge' else ['MLP_Clinical9','MLP_Clinical9_MSE']
    result=evaluate_frozen_scores(wide,names,iterations=10000,seed=20261120)
    summary=[]
    for j,name in enumerate(names):
        lo,hi=np.quantile(result['draws'][:,j],[.025,.975])
        summary.append(dict(model=name,AUC=float(result['points'][j]),CI_low=float(lo),CI_high=float(hi),
            Brier=float(brier_score_loss(wide.outcome,wide[name])),logloss=float(log_loss(wide.outcome,wide[name]))))
    pd.DataFrame(summary).to_csv(out/'aggregate_performance.csv',index=False)
    lo,hi=np.quantile(result['draws'][:,1]-result['draws'][:,0],[.025,.975])
    pd.DataFrame([dict(Delta_AUC=float(result['points'][1]-result['points'][0]),CI_low=float(lo),CI_high=float(hi),
        scope='paired patient-cluster fixed-OOF conditional; not independent validation')]).to_csv(out/'paired_increment.csv',index=False)
    result['folds'].to_csv(out/'aggregate_fold_results.csv',index=False)


if __name__=='__main__':main()
