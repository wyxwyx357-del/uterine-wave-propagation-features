"""One extraction per input, shared by every salience branch."""
import hashlib
import json
import numpy as np

from analysis.mse_pregnancy_association.code.activity_strength import filtered_runs
from analysis.mse_pregnancy_association.code.partial_phase_events import detect_partial
from analysis.mse_pregnancy_association.code.candidate_split_verifier import prepare_parent, decide_split
from analysis.mse_pregnancy_association.code.global_event_ownership import reconcile
from analysis.mse_pregnancy_association.code.patient_activity_aggregation import observation_union


def stable_hash(value):
    def encode(x):
        if isinstance(x, np.ndarray):
            return dict(shape=x.shape, dtype=str(x.dtype), sha256=hashlib.sha256(x.tobytes()).hexdigest())
        if isinstance(x, dict):
            return {str(k):encode(v) for k,v in sorted(x.items(),key=lambda item:str(item[0]))}
        if isinstance(x, (list,tuple)):
            return [encode(v) for v in x]
        if isinstance(x, np.generic):
            return x.item()
        return x
    return hashlib.sha256(json.dumps(encode(value),sort_keys=True,allow_nan=True).encode()).hexdigest()


def pool_hash(shared):
    return stable_hash({k:shared[k] for k in ('A','valid','filtered','t','s','span','dt','noise_floor','walls')})


def extract_shared_upstream(maps, noise_floor):
    a = np.asarray(maps['A'],float).copy()
    valid = np.asarray(maps['valid_mask'],bool) & np.isfinite(a)
    t = np.asarray(maps['interval_center_s'],float).copy()
    s = np.asarray(maps['s'],float).copy()
    if a.shape != (len(t),2,len(s)) or valid.shape != a.shape:
        raise ValueError('Expected native time x wall x section arrays')
    dt = float(np.median(np.diff(t)))
    span = float(np.median(maps['interval_dt_s']))
    filtered = filtered_runs(a,valid,t)
    finite = np.isfinite(filtered)
    observable = bool((finite[:,:,:-2] & finite[:,:,1:-1] & finite[:,:,2:]).any()) if len(s)>=3 else False
    exposure = observation_union(dict(maps,valid_mask=valid),min_run_s=12.)
    if observable != bool(exposure['seconds']>0):
        raise RuntimeError('Detector support and observation support diverged')
    walls=[]
    if observable:
        for wall in (0,1):
            report=detect_partial(a[:,wall],valid[:,wall],t,s,noise_floor=noise_floor)
            report.pop('filtered_A')
            splits=[decide_split(prepare_parent(a[:,wall],valid[:,wall],t,s,parent,
                         noise_floor=noise_floor,measurement_span_s=span),minimum_adjacent_support=3)
                    for parent in report['events']]
            ownership=reconcile(report,splits,t,valid[:,wall],measurement_span_s=span)
            walls.append(dict(wall=wall,parent_report=report,split_proposals=splits,reconciliation=ownership))
    shared=dict(A=a,valid=valid,filtered=filtered,t=t,s=s,span=span,dt=dt,
                noise_floor=float(noise_floor),walls=walls,observable=observable,exposure=exposure)
    for x in (a,valid,filtered,t,s):
        x.setflags(write=False)
    shared['pool_hash']=pool_hash(shared)
    return shared
