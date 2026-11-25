"""Four experimental families beyond S1; continuous scores, no chosen cutoffs."""
import numpy as np
from analysis.mse_pregnancy_association.code.activity_strength import RigidReference, filtered_runs
from analysis.mse_pregnancy_association.code.measurement.amplitude import band_limited_signal
from .core import event_windows
from .upstream import stable_hash


def mad_scale(values):
    values=np.asarray(values,float)
    values=values[np.isfinite(values)]
    return float(1.4826*np.median(abs(values-np.median(values)))) if len(values) else None


def out_of_band_residual(shared,band_cpm):
    low,high=band_cpm
    if not 5<low<high<30/shared['dt']:
        raise ValueError('Out-of-band proxy must be above 5 cpm and below native Nyquist frequency')
    result=np.full_like(shared['A'],np.nan)
    minimum=max(3,int(np.ceil(12/shared['dt'])))
    for wall in (0,1):
        for c in range(len(shared['s'])):
            valid=shared['valid'][:,wall,c]
            for a,b in np.flatnonzero(np.diff(np.r_[False,valid,False])).reshape(-1,2):
                if b-a>=minimum:
                    result[a:b,wall,c]=band_limited_signal(shared['A'][a:b,wall,c],shared['t'][a:b],band_cpm)
    return result


class DevRigidBank:
    def __init__(self,controls):
        if not controls:
            raise ValueError('Explicit Dev zero/rigid reference bank required')
        if len({c['control_id'] for c in controls})!=len(controls):
            raise ValueError('Reference control IDs must be unique')
        self.provenance=[];banks=[]
        for control in controls:
            if control['split']!='Dev' or control['condition'] not in ('zero','rigid'):
                raise ValueError('Only Dev zero/rigid controls may enter the reference bank; Val is forbidden')
            maps=control['maps']
            y=filtered_runs(maps['A'],maps['valid_mask'],maps['interval_center_s'])
            banks.append((control['control_id'],y,maps['interval_center_s']))
            self.provenance.append({k:control[k] for k in ('control_id','split','condition')})
        self.reference=RigidReference(banks)
        self.bank_hash=stable_hash(banks)
        self._cache={}

    def maximum_for_duration(self,start,end):
        duration=float(end-start)
        if duration not in self._cache:
            self._cache[duration]=self.reference.maximum(np.array([0]),np.array([0.]),np.array([duration]))
        return self._cache[duration]


class _RatioScorer:
    numerator='event-section RMS of filtered A'

    def score(self,shared,wall,event):
        rows=[]
        for w in event_windows(shared,wall,event):
            numerator=float(np.sqrt(np.mean(w['values']**2))) if w['complete'] else None
            denominator=self.scale(w,wall) if w['complete'] else None
            score=numerator/denominator if denominator is not None and denominator>0 else None
            rows.append(dict(section=w['section'],score=score,complete=w['complete'],sample_scores=None,
                             numerator=numerator,denominator=denominator,
                             unscored_reason=None if score is not None else 'missing_support_or_zero_or_missing_scale'))
        return rows


class PerSectionRigidReferenceScorer(_RatioScorer):
    name='S2_rigid_reference_rms_ratio'

    def __init__(self,bank):
        self.bank=bank

    def scale(self,window,wall):
        return self.bank.maximum_for_duration(window['start_s'],window['end_s'])['maximum_rms']


class OutOfBandMADScorer(_RatioScorer):
    name='S3a_mad_out_of_band_ratio'

    def __init__(self,shared,band_cpm):
        self.band_cpm=tuple(band_cpm)
        self.residual=out_of_band_residual(shared,self.band_cpm)
        self.scales=[[mad_scale(self.residual[:,wall,c]) for c in range(len(shared['s']))] for wall in (0,1)]

    def scale(self,window,wall):
        return self.scales[wall][window['section']]


class InBandMADScorer(_RatioScorer):
    name='S3b_mad_in_band_ratio'

    def __init__(self,shared):
        self.scales=[[mad_scale(shared['filtered'][:,wall,c]) for c in range(len(shared['s']))] for wall in (0,1)]

    def scale(self,window,wall):
        return self.scales[wall][window['section']]


class PercentileScorer:
    numerator='event-section peak |filtered A| minus video P05'

    def __init__(self,shared,upper_percentile):
        if upper_percentile not in (95,99):
            raise ValueError('Only requested P95/P99 variants are implemented')
        self.name=f'S4_percentile_P{upper_percentile}'
        finite=np.abs(shared['filtered'][np.isfinite(shared['filtered']) & shared['valid']])
        self.low,self.high=np.percentile(finite,[5,upper_percentile]) if len(finite) else (None,None)

    def score(self,shared,wall,event):
        rows=[]
        for w in event_windows(shared,wall,event):
            numerator=float(np.max(abs(w['values']))-self.low) if w['complete'] and self.low is not None else None
            denominator=float(self.high-self.low) if self.high is not None else None
            value=numerator/denominator if numerator is not None and denominator>0 else None
            rows.append(dict(section=w['section'],score=value,complete=w['complete'],sample_scores=None,
                             numerator=numerator,denominator=denominator,clipped=False,
                             unscored_reason=None if value is not None else 'missing_support_or_zero_or_missing_scale'))
        return rows
