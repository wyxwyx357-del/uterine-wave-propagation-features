"""Analytical component readout of existing inputs, never a detector experiment."""
import inspect
from collections import defaultdict
import numpy as np
from scipy.signal import find_peaks
from analysis.mse_pregnancy_association.code.activity_strength import filtered_runs
from analysis.mse_pregnancy_association.code.measurement.amplitude import band_limited_signal
from analysis.mse_pregnancy_association.code import activity_strength, partial_phase_events, candidate_split_verifier
from ..benchmark_data import ROOT, sha
from .attribution_rules import consecutive_count

def truth_component(record,truth):
    s=record['shared']['s'];arrival=truth['truth_center_s']+(1 if truth['truth_direction']=='C2F' else -1)*(s-.5)/truth['truth_speed']
    def length(t):
        delta=t[:,None]-arrival
        return 1+truth['truth_amplitude']*np.where(abs(delta)<=6.4,np.exp(-.5*(delta/1.6)**2),0)
    lo=length(record['interval_start']);hi=length(record['interval_end'])
    raw=(hi-lo)/lo/(record['interval_end']-record['interval_start'])[:,None]
    raw=np.broadcast_to(raw[:,None,:],record['shared']['A'].shape).copy()
    return raw,filtered_runs(raw,record['shared']['valid'],record['shared']['t']),arrival

def metrics(pred,record,truth,component):
    raw,filtered,arrival=component;shared=record['shared'];t=shared['t'];span=shared['span']
    observed=[]
    for chain in pred['source_chain']:
        for col,phases in chain['retained_phase_profile'].items():
            col=int(col);wall=chain['wall']
            for phase in phases:
                idx=int(np.argmin(abs(t-phase['peak_s'])))
                values=filtered[:,wall,col];finite=np.isfinite(values)
                safe=np.where(finite,values,0.)*phase['sign']
                peaks=find_peaks(safe)[0];peaks=peaks[finite[peaks] & (safe[peaks]>0)]
                near=int(peaks[np.argmin(abs(t[peaks]-phase['peak_s']))]) if len(peaks) else None
                observed.append(dict(wall=wall,section=col,sign=phase['sign'],peak_s=phase['peak_s'],
                    offset_from_truth_arrival_s=phase['peak_s']-float(arrival[col]),
                    actual_filtered_peak=float(shared['filtered'][idx,wall,col]),
                    truth_component_raw=float(raw[idx,wall,col]),truth_component_filtered=float(values[idx]),
                    filtered_truth_extremum_s=float(t[near]) if near is not None else None,
                    extremum_deviation_s=float(abs(t[near]-phase['peak_s'])) if near is not None else None,
                    corresponds_to_filtered_truth_lobe=bool(near is not None and abs(t[near]-phase['peak_s'])<=span),
                    outside_local_truth_measurement_support=bool(record['interval_end'][idx]<arrival[col]-6.4 or record['interval_start'][idx]>arrival[col]+6.4)))
    candidates=[]
    for wall in pred['walls']:
        for sign in (-1,1):
            rows=[x for x in observed if x['wall']==wall and x['sign']==sign and x['corresponds_to_filtered_truth_lobe'] and x['outside_local_truth_measurement_support']]
            offsets=[x['offset_from_truth_arrival_s'] for x in rows]
            candidates.append(dict(wall=wall,sign=sign,native_contiguous_sections=consecutive_count(x['section'] for x in rows),
                median_offset_s=float(np.median(offsets)) if offsets else None,
                offset_range_s=float(np.ptp(offsets)) if offsets else None,
                coherent=bool(consecutive_count(x['section'] for x in rows)>=3 and np.ptp(offsets)<=span) if offsets else False))
    mask=(t>=pred['evidence_start_s'])&(t<=pred['evidence_end_s'])
    selected=shared['filtered'][mask][:,pred['walls']][:,:,pred['sections']]
    physical=(t>=truth['truth_start_s'])&(t<=truth['truth_end_s'])
    ref=shared['filtered'][physical][:,pred['walls']][:,:,pred['sections']]
    numerator=float(np.nanmax(abs(selected))) if selected.size else None
    denominator=float(np.nanmax(abs(ref))) if ref.size else None
    return dict(truth_id=truth['truth_event_id'],phase_peak_correspondence=observed,coherent_lobe_candidates=candidates,
                amplitude_ratio_to_truth_window=numerator/denominator if denominator else None,
                ringing_like=False,interpretation='Fixed analytical truth component through unchanged filter; correspondence is not a causal intervention')

def finalize_recurrence(events):
    for p in events:
        if p['primary_class']!='B':continue
        for c in p['waveform']['coherent_lobe_candidates']:
            if not c['coherent']:continue
            compatible=[]
            for q in events:
                if q is p or q['primary_class']!='B' or q['texture_id']==p['texture_id']:continue
                if q['nearest_truth_relation']['temporal_side']!=p['nearest_truth_relation']['temporal_side']:continue
                if any(d['coherent'] and d['sign']==c['sign'] and abs(d['median_offset_s']-c['median_offset_s'])<=p['measurement_span_s'] for d in q['waveform']['coherent_lobe_candidates']):
                    compatible.append(q['event_key'])
            c['recurs_across_textures']=compatible
        p['waveform']['ringing_like']=any(c.get('recurs_across_textures') for c in p['waveform']['coherent_lobe_candidates'])

def filter_sources():
    result=[]
    for fn in (band_limited_signal,filtered_runs,partial_phase_events.observed_lobes,candidate_split_verifier.prepare_parent):
        path=inspect.getsourcefile(fn);lines,first=inspect.getsourcelines(fn)
        result.append(dict(file=str(__import__('pathlib').Path(path).resolve().relative_to(ROOT)),function=fn.__name__,
                           first_line=first,last_line=first+len(lines)-1,sha256=sha(path),source=''.join(lines)))
    assert inspect.getsource(partial_phase_events.baseline.band_limited_signal)==inspect.getsource(band_limited_signal)
    return dict(source_references=result,detector_filter_same_source=True,FFT=True,rectangular_keep_mask=True,
                explicit_taper_or_window=False,out_of_band_spectrum_zeroed=True,per_continuous_run=True,
                split_auxiliary_smoothing='Existing measurement_span_mean; distinct from FFT bandpass',
                conclusion='Source form establishes theoretical possibility only, not event-level causation')

def make_plot(pred,record,all_predictions,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    shared=record['shared'];t=shared['t'];wall=pred['walls'][0];cols=pred['sections']
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    lo=min([pred['evidence_start_s']]+[q['truth_start_s'] for q in record['truth']])-4
    hi=max([pred['evidence_end_s']]+[q['truth_end_s'] for q in record['truth']])+4
    chosen=sorted(set([cols[0],cols[len(cols)//2],cols[-1]]))
    for c in chosen:
        axes[0].plot(t,shared['A'][:,wall,c],label=f'section {c}',lw=.8)
        axes[1].plot(t,shared['filtered'][:,wall,c],label=f'section {c}',lw=1)
    for ax in axes[:2]:
        for i,truth in enumerate(record['truth']):ax.axvspan(truth['truth_start_s'],truth['truth_end_s'],color='green',alpha=.12,label='truth support' if i==0 else None)
        labeled=False
        for tp in all_predictions:
            if tp['matched_TP']:
                ax.axvspan(tp['evidence_start_s'],tp['evidence_end_s'],color='blue',alpha=.08,label='matched TP' if not labeled else None);labeled=True
        ax.axvspan(pred['evidence_start_s'],pred['evidence_end_s'],color='red',alpha=.16,label='selected detection')
        artifact=pred.get('artifact_relation')
        if artifact:ax.axvspan(artifact['start_s'],artifact['end_s'],color='orange',alpha=.15,label='artifact support')
        ax.legend(ncol=4,fontsize=7,loc='upper right')
    for chain in pred['source_chain']:
        for c,phases in chain['retained_phase_profile'].items():
            for phase in phases:axes[2].scatter(phase['peak_s'],int(c),c='red' if phase['sign']>0 else 'purple',s=15,marker='o' if chain['wall']==0 else 'x')
    for truth in record['truth']:
        _,_,arrival=truth_component(record,truth);axes[2].plot(arrival,np.arange(len(arrival)),color='green',lw=1)
    axes[0].set_ylabel('raw A (1/s)');axes[1].set_ylabel('filtered A (1/s)');axes[2].set_ylabel('native section');axes[2].set_xlabel('time (s)')
    axes[0].set_title(f"{pred.get('primary_class','TP')}/{pred.get('subtype','')} | {record['case_id']} | {pred['point']} | {pred['phase_origin']}/{pred['split_origin']}")
    axes[-1].set_xlim(max(0,lo),min(t[-1],hi));fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)
