"""Test separation/noise tradeoff of one prespecified event-input change.

Compare periodic+partial to local span-mean+partial, first using the frozen
absolute floor, then the SAME leave-one-texture-out 3*P95 SD calibration rule.
No parameter sweep, default replacement or clinical event-count claims.
"""
import argparse
from functools import partial
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from uterine_wave import radial_events
from uterine_wave.amplitude import band_limited_signal
from uterine_wave.io import jsonable
from experiments.audit_wave_evidence import control_arrays
from experiments.audit_event_filtering import known_match
from experiments.partial_phase_events import detect_partial
from experiments.test_double_event_signals import double_fixture,raw_truth,evaluate,GAPS,filter_phase_evidence
from experiments.test_weak_wave_signals import radial_fixture
from experiments.event_input_smoothing import measurement_span_mean


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    calibration=read(args.noise_reference)
    frozen=read(args.source_protocol)['source_files']
    cached={};sources={};sd={};periodic_sd={}
    def check(path):
        sha=digest(path)
        if frozen.get(str(path))!=sha:
            raise ValueError('Input cache changed or is absent from the frozen protocol')
        sources[str(path)]=sha
    for path in sorted(args.controls_root.glob('texture*_rigid.npz')):
        check(path)
        texture=path.stem.split('_')[0]
        a,valid,t,s,span=control_arrays(path)
        if not valid.all():
            raise ValueError('Expected original complete rigid-residual fixtures')
        cached[texture]=(a,valid,t,s,span)
        function=partial(measurement_span_mean,span_s=span)
        sd[texture]=np.std(function(a,t),axis=0).ravel()
        periodic_sd[texture]=np.std(band_limited_signal(a,t),axis=0).ravel()
    if len(cached)!=3:
        raise ValueError('Expected the same three frozen residual textures')
    alternate_floors={texture:3*float(np.percentile(np.concatenate([
        values for key,values in sd.items() if key!=texture]),95)) for texture in cached}
    for texture in cached:
        reproduced=3*float(np.percentile(np.concatenate([
            values for key,values in periodic_sd.items() if key!=texture]),95))
        if not np.isclose(reproduced,calibration['leave_one_texture_out_thresholds'][texture[7:]],rtol=1e-10,atol=1e-14):
            raise ValueError('Original noise-calibration rule failed to reproduce')
    rows=[]
    def compare(source,family,a,valid,t,s,span,texture,*,truth=None,clean=None,
                arrival=None,direction=None,negative=False,gap=None,second=None):
        floor=calibration['leave_one_texture_out_thresholds'][texture[7:]]
        smooth=partial(measurement_span_mean,span_s=span)
        methods=[('periodic_frozen',band_limited_signal,floor),
                 ('span_mean_frozen',smooth,floor),
                 ('span_mean_same_noise_rule',smooth,alternate_floors[texture])]
        row=dict(source=source,family=family,texture=texture,expected_direction=direction,
            known_nonpropagating=negative,gap_s=gap,second_amplitude=second,
            raw_observable_count=truth['uniform_observable_count'] if truth else None,methods={})
        for name,function,threshold in methods:
            walls=[]
            with patch.object(radial_events,'band_limited_signal',function):
                for wall in (0,1):
                    report=detect_partial(a[:,wall],valid[:,wall],t,s,noise_floor=threshold)
                    filtered=report.pop('filtered_A')
                    match=None
                    if clean is not None and truth is None:
                        report['time_s']=t
                        match=known_match(report,direction,arrival,clean,s)
                        report.pop('time_s')
                    evaluation=evaluate(report,truth) if truth else None
                    walls.append(dict(wall=wall,report=report,evaluation=evaluation,known_single_match=match))
            row['methods'][name]=dict(walls=walls,noise_floor=threshold,
                clean_phase_evidence=filter_phase_evidence(function(clean,t),truth,t,threshold)
                    if truth is not None else None)
        rows.append(row);save(args.output/(source+'.json'),row)
        print(source,flush=True)
    for texture,(noise,valid,t,s,span) in cached.items():
        for direction,amp,gap in product(('C2F','F2C'),(.04,.01),GAPS):
            clean,arrival=double_fixture(t,s,span,direction,gap,amp)
            truth=raw_truth(clean,t)
            compare(f'{texture}_double_{direction}_{amp:.2f}_gap{gap:g}','double',
                noise+clean[:,None],valid,t,s,span,texture,truth=truth,clean=clean,
                direction=direction,gap=gap,second=amp)
        definitions=[(d,amp) for d in ('C2F','F2C') for amp in (.04,.02,.01)]
        definitions += [(d,.04) for d in ('rigid_only','local_synchronous','global_synchronous')]
        for kind,amp in definitions:
            clean,arrival,_=radial_fixture(t,s,span,kind,amp)
            wave=kind in ('C2F','F2C')
            compare(f'{texture}_single_{kind}_{amp:.2f}','single_or_negative',
                noise+clean[:,None],valid,t,s,span,texture,clean=clean if wave else None,
                arrival=arrival,direction=kind if wave else None,negative=not wave)
    for path in sorted(args.controls_root.glob('texture*_*.npz')):
        texture,kind=path.stem.split('_',1)
        if kind not in ('C2F','F2C','rigid','local_once'):
            continue
        check(path)
        a,valid,t,s,span=control_arrays(path)
        compare(path.stem,'image_control',a,valid,t,s,span,texture,
                direction=kind if kind in ('C2F','F2C') else None,
                negative=kind in ('rigid','local_once'))
    if len(rows)!=87 or any(digest(Path(p))!=sha for p,sha in sources.items()):
        raise ValueError('Expected 48 double, 27 single/negative and 12 unchanged image controls')
    names=['periodic_frozen','span_mean_frozen','span_mean_same_noise_rule']
    def aggregate(selected,name):
        walls=[w for r in selected for w in r['methods'][name]['walls']]
        matches=[m for w in walls if w['evaluation'] is not None for m in w['evaluation']['matches'] or []]
        matched=[m for m in matches if m['matched']]
        singles=[w['known_single_match'] for w in walls if w['known_single_match'] is not None]
        return dict(wall_conditions=len(walls),
            direction_structures=sum(w['report']['traveling_candidate_count'] for w in walls),
            observable_double_truth_events=len(matches) if matches else None,
            assigned_double_events=len(matched) if matches else None,
            minimum_double_native_section_fraction=min((m['native_section_fraction'] for m in matches),default=None),
            median_double_arrival_RMSE_s=float(np.median([m['arrival_RMSE_s'] for m in matched])) if matched else None,
            maximum_double_arrival_RMSE_s=max((m['arrival_RMSE_s'] for m in matched),default=None),
            single_direction_hits=len(singles) if any(r['family']=='single_or_negative' and r['expected_direction'] for r in selected) else None,
            minimum_single_path_fraction=min((m['retained_path_fraction'] for m in singles),default=None),
            median_single_arrival_RMSE_s=float(np.median([m['arrival_RMSE_to_injection_s'] for m in singles])) if singles else None,
            maximum_single_arrival_RMSE_s=max((m['arrival_RMSE_to_injection_s'] for m in singles),default=None),
            nonpropagating_direction_structures=sum(w['report']['traveling_candidate_count'] for r in selected
                if r['known_nonpropagating'] for w in r['methods'][name]['walls']) if any(r['known_nonpropagating'] for r in selected) else None)
    def table(selected):
        return {name:aggregate(selected,name) for name in names}
    summary=dict(double_conditions=[dict(gap_s=gap,second_amplitude=amp,
        methods=table([r for r in rows if r['family']=='double' and r['gap_s']==gap and r['second_amplitude']==amp]))
        for gap,amp in product(GAPS,(.04,.01))],
        single_levels=[dict(amplitude=amp,methods=table([r for r in rows
            if r['family']=='single_or_negative' and r['expected_direction'] and r['source'].endswith(f'{amp:.2f}')]))
            for amp in (.04,.02,.01)],
        signal_negatives=table([r for r in rows if r['family']=='single_or_negative' and r['known_nonpropagating']]),
        image_controls=table([r for r in rows if r['family']=='image_control']),
        span_mean_noise_floors=alternate_floors,formal_event_count=None,patient_accuracy=None,
        adoption='none; assess separation, noise and weak-event preservation jointly')
    save(args.output/'summary.json',summary)
    save(args.output/'protocol.json',dict(source_files=sources,
        code_sha256={name:digest(ROOT/name) for name in (
            'experiments/event_input_smoothing.py','experiments/test_event_input_smoothing.py',
            'experiments/test_double_event_signals.py','experiments/partial_phase_events.py',
            'uterine_wave/radial_events.py','uterine_wave/amplitude.py')},
        reference_sha256={str(p):digest(p) for p in (args.noise_reference,args.source_protocol)},
        smoothing_span='Actual existing radial measurement interval; closest odd boxcar sample count',
        mean_removal='Per continuous run; no polynomial detrending, no spectral band mask',
        noise_rule='Same 3*P95 SD pooled across other textures; original rule reproduced exactly',
        formal_features_unchanged=True,new_physiological_threshold=False,adoption='none'))
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controls-root',type=Path,required=True)
    p.add_argument('--noise-reference',type=Path,required=True)
    p.add_argument('--source-protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
