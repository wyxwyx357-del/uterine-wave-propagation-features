"""Exactly one amplitude rule and one Butterworth filter; scoped experiment hooks."""
from contextlib import contextmanager,ExitStack
from unittest.mock import patch
import numpy as np
from scipy.signal import butter,sosfiltfilt
from analysis.mse_pregnancy_association.code import activity_strength as strength
from analysis.mse_pregnancy_association.code import partial_phase_events as partial
from ..upstream import stable_hash

MODELS=('M0','M1','M2','M3')
CORE=partial.baseline
ORIGINAL_PAIR=CORE._pair_phases
ORIGINAL_DETECT=CORE.detect_radial_events
ORIGINAL_OBSERVED=partial.observed_lobes
ORIGINAL_FFT=CORE.band_limited_signal
OLD_STRENGTH_FILTER=strength.band_limited_signal

@contextmanager
def filter_only(model):
    with ExitStack() as stack:
        if model in ('M2','M3'):
            stack.enter_context(patch.object(CORE,'band_limited_signal',butter_run))
            stack.enter_context(patch.object(strength,'band_limited_signal',butter_run))
        yield

def butter_run(values,t,band_cpm=(.5,5.)):
    assert tuple(band_cpm)==(.5,5.),'Only the one prelocked band is allowed'
    fs=1/float(np.median(np.diff(t)))
    sos=butter(2,[.5/60,5/60],btype='bandpass',fs=fs,output='sos')
    return sosfiltfilt(sos,np.asarray(values,float),axis=0)

def component_factors(ridges,edges,filtered):
    """Normalize only within the pre-weight candidate-edge connected component."""
    adjacency={i:set() for edge in edges for i in edge}
    for a,b in edges:adjacency[a].add(b);adjacency[b].add(a)
    components=[];owner={}
    for first in sorted(adjacency):
        if first in owner:continue
        members=set();todo=[first]
        while todo:
            n=todo.pop()
            if n in members:continue
            members.add(n);todo.extend(adjacency[n]-members)
        index=len(components);components.append(sorted(members))
        for n in members:owner[n]=index
    amplitudes={};references={}
    for index,members in enumerate(components):
        references[index]={}
        for rid in members:
            for node in ridges[rid]['nodes']:
                c=node['column'];samples=filtered[node['first']:node['last']+1,c]
                # This is a production lobe inside an already-valid run. An
                # inconsistent/missing amplitude is an implementation failure,
                # never a new validity exclusion or silently dropped section.
                assert len(samples)>0 and np.isfinite(samples).all()
                value=float(np.max(abs(samples)))
                assert value==node['amplitude']
                amplitudes[rid,c]=value
                references[index][c]=max(references[index].get(c,0.),value)
    result={}
    for key,edge in edges.items():
        a,b=key;index=owner[a];assert owner[b]==index
        ids=edge['columns'];ai=np.array([amplitudes[a,c] for c in ids]);aj=np.array([amplitudes[b,c] for c in ids])
        reference=np.array([references[index][c] for c in ids])
        joint=np.zeros(len(ids));positive=reference>0
        joint[positive]=np.minimum(ai[positive]/reference[positive],aj[positive]/reference[positive])
        factor=float(np.median(joint));assert 0<=factor<=1
        result[key]=dict(component_id=index,component_ridge_ids=components[index],sections=list(ids),
            phase_i_amplitude=ai.tolist(),phase_j_amplitude=aj.tolist(),R=reference.tolist(),section_joint_factor=joint.tolist(),G=factor,
            phase_i_amplitude_summary=dict(min=float(ai.min()),median=float(np.median(ai)),max=float(ai.max())),
            phase_j_amplitude_summary=dict(min=float(aj.min()),median=float(np.median(aj)),max=float(aj.max())))
    return result

def weighted_pair(ridges,edges,filtered,apply_factor,force_one=False):
    factors=component_factors(ridges,edges,filtered);weighted={};audit=[]
    for key,edge in edges.items():
        factor=1. if force_one else factors[key]['G']
        old=edge['association_weight'];new=old*factor
        used=new if apply_factor else old
        weighted[key]=dict(edge,association_weight=used)
        audit.append(dict(ridge_key=list(key),phase_ridge_ids=[edge['first'],edge['second']],W_old=old,
                          **factors[key],W_new=new,W_used=used))
    selected,ambiguous=ORIGINAL_PAIR(ridges,weighted)
    for row in audit:row['selected']=tuple(row['ridge_key']) in selected
    return selected,ambiguous,audit

@contextmanager
def intervention(model,logs=None,force_one=False):
    """Only global FFT detector/scoring filter and global pairing inputs change.

    split.prepare_parent also calls _pair_phases locally. Those calls bypass
    the experimental weight entirely. Its original _measure_pair and
    measurement_span_mean remain untouched, including child ranking.
    OOB MAD imports its separate original FFT function and stays unchanged.
    """
    assert model in MODELS
    logs=[] if logs is None else logs
    state={}
    def detector(*args,**kwargs):
        assert not state.get('global_detector_active',False)
        state['global_detector_active']=True
        try:return ORIGINAL_DETECT(*args,**kwargs)
        finally:state['global_detector_active']=False
    def observed(y,valid,t,floor):
        filtered,columns,unresolved=ORIGINAL_OBSERVED(y,valid,t,floor)
        state.update(filtered=filtered,t=t,columns=columns)
        return filtered,columns,unresolved
    def pairing(ridges,edges):
        if not state.get('global_detector_active',False):return ORIGINAL_PAIR(ridges,edges)
        assert 'filtered' in state
        selected,ambiguous,audit=weighted_pair(ridges,edges,state['filtered'],model in ('M1','M3'),force_one)
        logs.append(dict(wall=len(logs),ridges=ridges,edges=audit,selected=[list(x) for x in selected],
                         raw_edge_signature=stable_hash(edges),filtered_signature=stable_hash(state['filtered'])))
        return selected,ambiguous
    with ExitStack() as stack:
        if model in ('M2','M3'):
            stack.enter_context(patch.object(CORE,'band_limited_signal',butter_run))
            stack.enter_context(patch.object(strength,'band_limited_signal',butter_run))
        stack.enter_context(patch.object(partial,'observed_lobes',observed))
        stack.enter_context(patch.object(CORE,'detect_radial_events',detector))
        stack.enter_context(patch.object(CORE,'_pair_phases',pairing))
        yield logs
    assert CORE._pair_phases is ORIGINAL_PAIR
    assert CORE.band_limited_signal is ORIGINAL_FFT
    assert strength.band_limited_signal is OLD_STRENGTH_FILTER

def run_inventory(maps,execute=False):
    t=maps['interval_center_s'];A=maps['A'];valid=maps['valid_mask']&np.isfinite(A)
    minimum=max(3,int(np.ceil(12/np.median(np.diff(t)))))
    rows=[]
    for wall in (0,1):
        for c in range(A.shape[2]):
            for ordinal,(a,b) in enumerate(np.flatnonzero(np.diff(np.r_[False,valid[:,wall,c],False])).reshape(-1,2)):
                if b-a<minimum:continue
                row=dict(wall=wall,section=c,run_id=ordinal,first=int(a),last_exclusive=int(b),n_samples=int(b-a),
                    center_span_s=float(t[b-1]-t[a]),support_seconds=float(maps['interval_end_s'][b-1]-maps['interval_start_s'][a]),
                    fs=1/float(np.median(np.diff(t[a:b]))),M2_filter_unsupported=False)
                if execute:
                    try:
                        values=butter_run(A[a:b,wall,c],t[a:b])
                    except ValueError as error:
                        row.update(M2_filter_unsupported=True,reason=str(error),exception_type=type(error).__name__)
                    else:
                        assert np.isfinite(values).all(),'Butterworth produced nonfinite output'
                rows.append(row)
    return rows
