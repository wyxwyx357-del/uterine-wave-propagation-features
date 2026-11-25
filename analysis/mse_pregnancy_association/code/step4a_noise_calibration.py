"""Zero-deformation development controls; frozen measurement/count code is reused.

Generate uses CUDA A0, never a CPU tracking substitute. Analyze uses saved maps.
No patient manifest, outcomes, new cutoff, or alternative normalization is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from analysis.mse_pregnancy_association.code.measurement.deformation import rate_A, interval_support, compute_spacetime

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (20261116, 20261117, 20261118)
CONDITIONS = ('static', 'rigid', 'brightness')
MODULES = ('tracking', '_lk', 'geometry', 'deformation', 'amplitude', 'controls')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def source_hashes():
    paths = [ROOT / 'analysis/mse_pregnancy_association/code/measurement' / (name+'.py') for name in MODULES]
    paths += [ROOT / 'analysis/mse_pregnancy_association/code' / (name+'.py') for name in
              ('activity_counting', 'activity_strength', 'normalized_activity', 'batch_wave_counts',
               'patient_activity_aggregation', 'candidate_split_verifier', 'partial_phase_events',
               'global_event_ownership', 'direction_identity_trial')]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def cached_maps(path):
    """Only named A0 arrays are read; no P3 candidate tracks or feature tables."""
    with np.load(path, allow_pickle=False) as z:
        t = z['time_s']
        p = z['A0_positions']
        tracks = dict(time_s=t, target_valid=z['A0_target_valid'],
                      step_valid=z['A0_step_valid'], segment_id=z['A0_segment_id'])
    if p.ndim != 3 or p.shape[1] % 7:
        raise ValueError('Expected cached [4N radial, 3N band] A0 points')
    n = p.shape[1]//7
    ids = np.arange(4*n).reshape(2, n, 2)
    lengths = np.linalg.norm(p[0, :4*n].reshape(2,n,2,2)[:,:,1] - p[0, :4*n].reshape(2,n,2,2)[:,:,0], axis=-1)
    if not np.allclose(lengths, 21., atol=5e-5, rtol=0):
        raise ValueError('Cache is not the fixed 21 px radial baseline')
    hop = max(1, int(np.floor(2/np.median(np.diff(t))+.5)))
    a = rate_A(p[:, :4*n], dict(A_pair_ids=ids), t, hop)
    valid = interval_support(tracks, hop)[:, :4*n][:, ids].all(3) & np.isfinite(a)
    inner = p[0, :4*n].reshape(2,n,2,2)[:,:,0]
    retained = np.linalg.norm(np.diff(inner,axis=1),axis=2).sum(1).min()
    trim = 21/(retained+42)
    return dict(A=a, valid_mask=valid, s=np.linspace(trim,1-trim,n),
                interval_start_s=t[:-hop], interval_end_s=t[hop:],
                interval_center_s=(t[:-hop]+t[hop:])/2, interval_dt_s=t[hop:]-t[:-hop])


def generate(output):
    import cv2
    from analysis.mse_pregnancy_association.code.measurement.controls import synthetic_geometry, synthetic_texture, known_motion, reference_tracks
    from analysis.mse_pregnancy_association.code.measurement.geometry import layout, CONFIG
    from analysis.mse_pregnancy_association.code.measurement.tracking import track_segments
    if cv2.cuda.getCudaEnabledDeviceCount() < 1:
        raise RuntimeError('Frozen A0 requires CUDA OpenCV')
    output.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    save(output/'protocol_before_generation.json', dict(
        source_hashes=hashes, step1_tag='activity-count-v2-step1-freeze',
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        geometry=CONFIG, seeds=list(SEEDS), conditions=list(CONDITIONS),
        time='501 native samples, dt=.12s, 0..60s', nominal_interval_s=2.,
        rigid='existing controls.known_motion: dx=3*sin(2*pi*t/30), dy=2*(cos(2*pi*t/30)-1)',
        brightness='static image * (1+.10*sin(2*pi*t/20)); round and clip uint8',
        pose_source='analytic exact poses; background pose estimator not tested',
        truth='zero geometric deformation in all conditions',
        scope='development software controls, not independent ultrasound/physiological validation',
        threshold_selection=False, outcomes_read=False, cv2_version=cv2.__version__))
    t = np.arange(501)*.12
    shapes = synthetic_geometry()
    grid = layout(shapes)
    initial = np.vstack((grid['A_points'], grid['B_points']))
    for seed in SEEDS:
        gray = synthetic_texture(seed)
        for condition in CONDITIONS:
            frames, truth, poses = known_motion(gray, shapes, grid, t, 'rigid' if condition=='rigid' else 'static')
            if condition == 'brightness':
                frames = [np.uint8(np.clip(np.rint(frame.astype(float)*(1+.10*np.sin(2*np.pi*time/20))),0,255))
                          for frame,time in zip(frames,t)]
            tracks = track_segments(frames, initial, t, poses)
            maps = compute_spacetime(tracks, grid, interval_s=2.)
            reference = compute_spacetime(reference_tracks(truth,t),grid,interval_s=2.)
            np.testing.assert_allclose(reference['A'],0.,atol=1e-12,rtol=0)
            name = f'synthetic_{seed}_{condition}'
            np.savez_compressed(output/(name+'_maps.npz'), **maps)
            np.savez_compressed(output/(name+'_tracks.npz'), **tracks)
            valid = tracks['target_valid'] & np.isfinite(tracks['corrected']).all(2)
            error = np.linalg.norm(tracks['corrected']-truth,axis=2)
            save(output/(name+'_truth.json'), dict(
                zero_deformation_truth_max_abs_A=float(np.max(np.abs(reference['A']))),
                valid_point_fraction=float(valid.mean()),
                point_rmse_px=float(np.sqrt(np.mean(error[valid]**2))) if valid.any() else None,
                reanchor_count=int(tracks['reanchor'].sum())))
            print('generated',name,flush=True)
    if source_hashes() != hashes:
        raise RuntimeError('Frozen source changed during generation')


def distribution(values):
    x = np.asarray(values,float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0, mean=None, rms=None, median=None, mad=None,
                    abs_p50=None, abs_p90=None, abs_p95=None, abs_p99=None, abs_max=None)
    median = float(np.median(x))
    q = np.percentile(np.abs(x),[50,90,95,99,100])
    return dict(n=int(len(x)),mean=float(x.mean()),rms=float(np.sqrt(np.mean(x*x))),
                median=median,mad=float(np.median(np.abs(x-median))),
                **{key:float(v) for key,v in zip(('abs_p50','abs_p90','abs_p95','abs_p99','abs_max'),q)})


def analyze(args):
    import pandas as pd
    from analysis.mse_pregnancy_association.code.activity_strength import filtered_runs
    from analysis.mse_pregnancy_association.code.normalized_activity import normalize_magnitude
    from analysis.mse_pregnancy_association.code.batch_wave_counts import count_map
    from analysis.mse_pregnancy_association.code.measurement.io import jsonable
    args.output.mkdir(parents=True,exist_ok=False)
    hashes = source_hashes()
    floor = float(json.loads(args.noise_reference.read_text(encoding='utf-8'))['all_texture_threshold'])
    generation_protocol = args.generated/'protocol_before_generation.json'
    if json.loads(generation_protocol.read_text(encoding='utf-8'))['source_hashes'] != hashes:
        raise RuntimeError('Generated controls used different frozen source code')
    sources = {str(generation_protocol):digest(generation_protocol),
               str(args.noise_reference):digest(args.noise_reference)}
    inputs = []
    for i in range(3):
        path = args.cached_controls/f'texture{i}_rigid.npz'
        sources[str(path)] = digest(path)
        inputs.append((f'cached_texture{i}_rigid', 'cached_real_texture', 'rigid', cached_maps(path)))
    for path in sorted(args.generated.glob('synthetic_*_maps.npz')):
        sources[str(path)] = digest(path)
        with np.load(path,allow_pickle=False) as z:
            inputs.append((path.stem.removesuffix('_maps'), 'synthetic_texture', path.stem.split('_')[-2], dict(z)))
    if len(inputs) != 12:
        raise ValueError('Expected three cached rigid controls and nine newly generated controls')
    save(args.output/'protocol_before_analysis.json',dict(
        source_hashes=hashes,input_sha256=sources,driver_sha256=digest(__file__),
        noise_reference_path=str(args.noise_reference),noise_reference_sha256=digest(args.noise_reference),
        fixed_noise_floor_rate=floor,cutoff=.6,minimum_adjacent_sections=3,
        count_algorithm_version='activity_count_v2_logic_fix_only',thresholds_changed=False,
        mad_and_percentiles='descriptive only; no rejection thresholds or normalization changes',
        overlapping_samples='not independent observations; report individual textures, no patient confidence interval',
        reference_reuse='cached rigid textures also contributed to historical noise reference; development evidence only',
        scope='Interpolation/quantization/tracking residuals with exact pose correction; not acoustic simulation, probe jitter or pose-estimation validation',
        pregnancy_outcomes_read=False,patient_cohort_run=False))
    rows=[]; sections=[]
    for name,group,condition,maps in inputs:
        valid = maps['valid_mask'] & np.isfinite(maps['A'])
        filtered = filtered_runs(maps['A'],valid,maps['interval_center_s'])
        normalized,_ = normalize_magnitude(filtered,valid)
        salient = np.isfinite(normalized) & (normalized > .6)
        spatial = (salient[:,:,:-2] & salient[:,:,1:-1] & salient[:,:,2:]).any((1,2))
        result = count_map(maps,floor)
        save(args.output/(name+'_count.json'),jsonable(result))
        row=dict(control=name,group=group,condition=condition,truth_propagation_count=0,
                 native_sections=len(maps['s']),measurement_dt_s=float(np.median(maps['interval_dt_s'])),
                 raw_valid_fraction=float(valid.mean()),filtered_valid_fraction=float(np.isfinite(filtered).mean()),
                 raw_observation_seconds=result['raw_observation_seconds'],filtered_observation_seconds=result['filtered_observation_seconds'],
                 false_significant_propagation_count_min=result['significant_propagation_activity_count_min'],
                 false_significant_propagation_count_max=result['significant_propagation_activity_count_max'],
                 false_significant_propagation_frequency_cpm_min=result['significant_propagation_activity_frequency_cpm_min'],
                 false_significant_propagation_frequency_cpm_max=result['significant_propagation_activity_frequency_cpm_max'],
                 other_significant_activity_count=result.get('other_significant_activity_count'),
                 raw_spatial_salience_sample_count=int(spatial.sum()),
                 salience_sample_fraction=float(salient.sum()/np.isfinite(filtered).sum()) if np.isfinite(filtered).any() else None,
                 unresolved_activity_rows=result.get('unresolved_activity_rows'))
        if group == 'synthetic_texture':
            truth_path = args.generated/(name+'_truth.json')
            row.update(json.loads(truth_path.read_text(encoding='utf-8')))
        row.update({'raw_A_'+k:v for k,v in distribution(maps['A'][valid]).items()})
        row.update({'filtered_A_'+k:v for k,v in distribution(filtered).items()})
        for wall in (0,1):
            for c in range(len(maps['s'])):
                section=dict(control=name,wall=wall,section=c,s=float(maps['s'][c]))
                section.update({'raw_A_'+k:v for k,v in distribution(maps['A'][:,wall,c][valid[:,wall,c]]).items()})
                section.update({'filtered_A_'+k:v for k,v in distribution(filtered[:,wall,c]).items()})
                sections.append(section)
        rows.append(row)
        print(name, 'RMS',row['raw_A_rms'],'filtered',row['filtered_A_rms'],
              'false count',[row['false_significant_propagation_count_min'],row['false_significant_propagation_count_max']],flush=True)
    pd.DataFrame(rows).to_csv(args.output/'controls.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(sections).to_csv(args.output/'sections.csv',index=False,encoding='utf-8-sig')
    if source_hashes()!=hashes or any(digest(Path(p))!=h for p,h in sources.items()):
        raise RuntimeError('Source inputs or frozen code changed during calibration')
    save(args.output/'summary.json',dict(controls=rows,
        interpretation='All activities in these zero-deformation controls are false detections conditional on current identity ambiguity, fixed noise floor and 0.6 cutoff.',
        limitations=['Three cached real-texture controls and three synthetic texture seeds are development controls, not independent patients.',
                     'Exact supplied poses exclude background estimator error; brightness changes are one fixed software perturbation.',
                     'MAD is unscaled descriptive MAD; no rigid-reference/MAD/percentile method has been selected.',
                     'Pointwise normalized salience and three-section sample counts are diagnostics, not detector event counts.',
                     'No empirical population false-positive rate, physiological validation or Step 2 conclusion.'],
        source_and_input_hashes_unchanged=True))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('generate','analyze'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--generated',type=Path)
    parser.add_argument('--cached-controls',type=Path)
    parser.add_argument('--noise-reference',type=Path)
    args=parser.parse_args()
    generate(args.output) if args.mode=='generate' else analyze(args)
