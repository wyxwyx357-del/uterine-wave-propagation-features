"""Segment-supported propagation mode features, separate from event detection."""
import numpy as np
from .spectrum import mode_velocity
from .amplitude import summarize_interval_strain


def contiguous_blocks(mask, minimum=3):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)
    return [(int(a), int(b)) for a, b in edges if b-a >= minimum]


def analyze_modes(spacetime, *, window_s=20., step_s=1., regions=None, refine_velocity=False):
    """Each spectrum uses an intact rectangular block; no concatenated gaps.

    20 s follows the prior literature-inspired mode protocol; 1 s step and
    three-column numerical minimum are project settings, not event criteria.
    All supported spatial blocks are kept. Coordination compares identical
    sections and window times on the two walls, grouped by spatial block.
    """
    a=np.asarray(spacetime['A'],float);v=np.asarray(spacetime['valid_mask'],bool)
    epsilon=np.asarray(spacetime['epsilon'],float) if 'epsilon' in spacetime else None
    t=np.asarray(spacetime['interval_center_s'],float);s=np.asarray(spacetime['s'],float)
    if a.shape != v.shape or a.shape != (len(t),2,len(s)) or len(t)<3:
        raise ValueError('Expected matching time x 2 walls x spatial section maps')
    if epsilon is not None and epsilon.shape != a.shape:
        raise ValueError('epsilon must match the time x wall x section map')
    dt=np.diff(t)
    if np.any(dt<=0) or not np.allclose(dt,np.median(dt),rtol=1e-5,atol=1e-9):
        raise ValueError('Mode FFT requires native uniform timestamps, without filling')
    if window_s<=0 or step_s<=0:raise ValueError('Positive window and step required')
    n=max(3,int(round(window_s/np.median(dt))));step=max(1,int(round(step_s/np.median(dt))))
    support=v&np.isfinite(a);rows=[];paired=[];missing=[]
    requested=regions or [(0,0,len(s),'anterior'),(1,0,len(s),'posterior')]
    requested_mask=np.zeros((2,len(s)),bool)
    for wall,begin,end,name in requested:
        if not 0<=wall<2 or not 0<=begin<end<=len(s):raise ValueError('Invalid spatial region')
        requested_mask[wall,begin:end]=True
    support &= requested_mask[None]
    for first in range(0,len(t)-n+1,step):
        last=first+n;complete=support[first:last].all(0)
        window_results={}
        def base(lo,hi):
            return dict(window_start_s=float(t[first]),window_end_s=float(t[last-1]),
                window_center_s=float((t[first]+t[last-1])/2),first_section=lo+1,last_section=hi,
                s_start=float(s[lo]),s_end=float(s[hi-1]),center_span=float(s[hi-1]-s[lo]))
        for wall,begin,end,name in requested:
            if not 0<=wall<2 or not 0<=begin<end<=len(s):raise ValueError('Invalid spatial region')
            blocks=contiguous_blocks(complete[wall,begin:end])
            if not blocks:missing.append(dict(region=name,window_start_s=float(t[first]),reason='no_contiguous_three_column_support'))
            for lo,hi in blocks:
                lo+=begin;hi+=begin
                result=mode_velocity(a[first:last,wall,lo:hi],t[first:last],s[lo:hi],refine=refine_velocity)
                if epsilon is not None:
                    amplitude=summarize_interval_strain(epsilon[first:last,wall,lo:hi],t[first:last])
                    result.update(interval_strain_raw_SD_mean=float(amplitude['raw_SD_mean']),
                        interval_strain_band_SD_mean=float(amplitude['band_SD_mean']),
                        amplitude_band_cpm=amplitude['band_cpm'],
                        amplitude_positive_frequency_bins=amplitude['positive_frequency_bins'])
                window_results[wall,lo,hi]=result
                rows.append(dict(**base(lo,hi),wall=wall,region=name,**result))
        # Compare the same original spatial block, not two unrelated local ROIs.
        for lo,hi in contiguous_blocks(complete.all(0)):
            pair_results=[]
            for wall in (0,1):
                key=(wall,lo,hi)
                if key not in window_results:
                    window_results[key]=mode_velocity(a[first:last,wall,lo:hi],t[first:last],s[lo:hi],refine=refine_velocity)
                pair_results.append(window_results[key])
            left,right=pair_results
            if left['ER'] is not None and right['ER'] is not None:
                paired.append(dict(**base(lo,hi),anterior_ER=left['ER'],posterior_ER=right['ER']))
    coordination=[]
    groups=sorted(set((r['first_section'],r['last_section']) for r in paired))
    for lo,hi in groups:
        group=[r for r in paired if (r['first_section'],r['last_section'])==(lo,hi)]
        # Split temporal gaps too; do not align separated fragments by row index.
        times=np.array([r['window_center_s'] for r in group]);boundaries=np.r_[0,np.flatnonzero(np.diff(times)>step*np.median(dt)*1.5)+1,len(group)]
        for begin,end in zip(boundaries[:-1],boundaries[1:]):
            part=group[begin:end];x=np.array([r['anterior_ER'] for r in part]);y=np.array([r['posterior_ER'] for r in part])
            cc=float(np.corrcoef(x,y)[0,1]) if len(x)>=3 and np.std(x)>1e-12 and np.std(y)>1e-12 else None
            coordination.append(dict(first_section=lo,last_section=hi,window_count=len(part),
                start_center_s=part[0]['window_center_s'],end_center_s=part[-1]['window_center_s'],
                zero_lag_ER_Pearson_CC=cc,ER_MSE=float(np.mean((x-y)**2)),
                status='mode_similarity_candidate',overlapping_windows=True))
    return dict(windows=rows,paired_ER_windows=paired,coordination=coordination,missing=missing,
        nominal_window_s=float(window_s),nominal_step_s=float(step_s),
        actual_window_sample_span_s=float((n-1)*np.median(dt)),
        actual_step_s=float(step*np.median(dt)),
        numerical_minimum_columns=3,refined_velocity=bool(refine_velocity),event_count=None,event_frequency_cpm=None)
