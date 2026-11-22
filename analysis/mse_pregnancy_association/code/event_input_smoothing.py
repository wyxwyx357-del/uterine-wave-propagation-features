"""Event-input ablation only: local averaging at the measurement time scale.

This is not the physiological-band amplitude/ER filter. The detector applies
it separately inside each continuous valid run; missing samples are not filled.
"""
import numpy as np


def measurement_span_mean(values, times, *, span_s):
    """Demean and apply an odd centered boxcar near one measurement interval.

    Reflection pads only the short kernel edge; padded values are not observed
    support. There is no spectral band selection, polynomial detrending or new
    duration parameter independent of the existing radial measurement span.
    """
    y=np.asarray(values,float);t=np.asarray(times,float)
    if y.ndim<1 or y.shape[0]!=len(t) or len(t)<3:
        raise ValueError('Expected time-first values on at least three timestamps')
    differences=np.diff(t)
    if (not np.isfinite(y).all() or not np.isfinite(t).all() or
        np.any(differences<=0) or not np.allclose(differences,np.median(differences),rtol=1e-5,atol=1e-9) or
        not np.isfinite(span_s) or span_s<=0):
        raise ValueError('Expected finite complete values, uniform time and positive measurement span')
    samples=int(2*np.floor(span_s/np.median(differences)/2)+1)
    if samples>len(t):
        raise ValueError('Smoothing span exceeds the continuous input run')
    half=samples//2
    centered=y-y.mean(axis=0,keepdims=True)
    padded=np.pad(centered,[(half,half)]+[(0,0)]*(y.ndim-1),mode='reflect')
    integral=np.concatenate([np.zeros_like(padded[:1]),np.cumsum(padded,axis=0)],axis=0)
    return (integral[samples:]-integral[:-samples])/samples
