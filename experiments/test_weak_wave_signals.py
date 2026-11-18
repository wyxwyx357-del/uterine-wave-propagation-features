"""Fixed-noise signal-layer propagation and synchronous-activity controls.

This adds known finite-interval radial-rate signals to saved rigid residuals.
It is not an image-warp or weak-speckle tracking validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.audit_wave_evidence import control_arrays
from experiments.phase_prediction_evidence import analyze_prediction
from uterine_wave.radial_events import detect_radial_events
from uterine_wave.amplitude import band_limited_signal
from uterine_wave.io import jsonable


def radial_fixture(t, s, interval_s, condition, stretch=.04):
    """One Gaussian length change, observed at the frozen finite interval.

    Fixed center=30s, width=4s, wave speed=.04 full-wall paths/s.
    Local control activates the same five central native positions at once.
    """
    t, s = np.asarray(t, float), np.asarray(s, float)
    arrival = np.full(len(s), 30.)
    spatial = np.ones(len(s))
    if condition in ('C2F', 'F2C'):
        arrival += (1 if condition == 'C2F' else -1)*(s-.5)/.04
    elif condition == 'local_synchronous':
        spatial[:] = 0
        spatial[np.argsort(abs(s-.5))[:5]] = 1
    elif condition == 'rigid_only':
        spatial[:] = 0
    elif condition != 'global_synchronous':
        raise ValueError('Unknown signal fixture')
    offset = t[:, None]-arrival
    before = 1+stretch*spatial*np.exp(-.5*((offset-interval_s/2)/4)**2)
    after = 1+stretch*spatial*np.exp(-.5*((offset+interval_s/2)/4)**2)
    rate = (after-before)/before/interval_s
    return rate, arrival, spatial


def add_to_fixed_noise(noise, signal):
    result = np.asarray(noise, float)+np.asarray(signal, float)
    np.testing.assert_allclose(result-signal, noise, atol=1e-15, rtol=0)
    return result


def match_known_structure(options):
    """Match the injected whole-path structure before comparing timing error.

    Picking minimum timing error first can favor a tiny incidental fragment.
    This rule is only for the known full-path fixture, not for real short waves.
    """
    return min(options, key=lambda x:(-x['retained_path_fraction'],
                                      x['arrival_RMSE_to_injection_s'])) if options else None


def save(path, value):
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(args.noise_reference.read_text(encoding='utf-8'))
    previous = {r['source']:r for r in json.loads(args.prior_controls.read_text(encoding='utf-8'))}
    rows, sources, clean_reference = [], {}, []
    paths = sorted(args.controls_root.glob('texture*_rigid.npz'))
    if len(paths) != 3:
        raise ValueError('Expected the three frozen rigid textures')
    for path in paths:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != previous[path.stem]['source_sha256']:
            raise ValueError('Rigid residual cache changed')
        sources[str(path)] = sha
        texture = path.stem.split('_')[0]
        noise, valid, t, s, dt = control_arrays(path)
        threshold = calibration['leave_one_texture_out_thresholds'][texture[7:]]
        for direction in ('C2F', 'F2C'):
            signal, _, _ = radial_fixture(t, s, dt, direction, .04)
            for wall in (0, 1):
                reference = detect_radial_events(signal, valid[:, wall], t, s, noise_floor=threshold)
                clean_reference.append(dict(texture=texture, direction=direction, wall=wall,
                    directional_candidates=reference['traveling_candidate_count']))
        conditions = [(direction, amplitude) for direction in ('C2F', 'F2C')
                      for amplitude in (.04, .02, .01)]
        conditions += [(condition, .04) for condition in
                       ('local_synchronous', 'global_synchronous', 'rigid_only')]
        for condition, stretch in conditions:
            clean, truth, active = radial_fixture(t, s, dt, condition, stretch)
            clean_filtered = band_limited_signal(clean, t)
            noisy = add_to_fixed_noise(noise, clean[:, None])
            walls, filtered = [], np.full_like(noisy, np.nan)
            for wall in (0, 1):
                report = detect_radial_events(noisy[:, wall], valid[:, wall], t, s,
                                               noise_floor=threshold)
                filtered[:, wall] = report.pop('filtered_A')
                for event in report['events']:
                    event['event_id'] = f'wall{wall}_'+event['event_id']
                walls.append(dict(wall=wall, **report))
            evidence = analyze_prediction(dict(walls=walls), dict(filtered_A=filtered,
                candidate_valid_mask=valid&np.isfinite(filtered), interval_center_s=t, s=s))
            directional = [e for w in walls for e in w['events'] if e['classification'] in ('C2F', 'F2C')]
            matches = []
            if condition in ('C2F', 'F2C'):
                for wall in walls:
                    options = []
                    for event in wall['events']:
                        if event['classification'] != condition:
                            continue
                        ids = event['sections']
                        expected = truth[ids] if event['evidence_kind'] == 'paired_phase' else (
                            t[np.argmax(event['leading_polarity']*clean_filtered[:, ids], axis=0)])
                        rmse = float(np.sqrt(np.mean((np.asarray(event['arrival_time_s'])-expected)**2)))
                        options.append(dict(event_id=event['event_id'], arrival_RMSE_to_injection_s=rmse,
                            spatial_positions=len(ids), retained_path_fraction=float(np.ptp(s[ids])/np.ptp(s)),
                            evidence_kind=event['evidence_kind']))
                    matches.append(match_known_structure(options))
            evidence_by_id = {e['event_id']:e for e in evidence['events']}
            for match in matches:
                if match is not None:
                    e = evidence_by_id[match['event_id']]
                    match['prediction_cosine'] = e['predicted_shape_cosine_mean']
                    match['normalized_arrival_error'] = e['arrival_loo_RMSE_over_range']
            predict = [e['predicted_shape_cosine_mean'] for e in evidence['events']
                       if e['predicted_shape_cosine_mean'] is not None]
            order = [e['arrival_loo_RMSE_over_range'] for e in evidence['events']
                     if e['arrival_loo_RMSE_over_range'] is not None]
            name = f'{texture}_{condition}_{stretch:.2f}'
            row = dict(source=name, texture=texture, condition=condition,
                injected_relative_length_peak=stretch if condition != 'rigid_only' else 0.,
                actual_interval_s=dt, clean_rate_peak=float(np.max(abs(clean))),
                noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),
                fixed_noise_floor_rate=threshold,
                directional_candidates=len(directional),
                correct_direction_walls=sum(x is not None for x in matches) if matches else None,
                matches_to_injected_structure=matches,
                prediction_cosine_median=float(np.median(predict)) if predict else None,
                normalized_arrival_error_median=float(np.median(order)) if order else None)
            rows.append(row)
            save(args.output/(name+'.json'), dict(summary=row, walls=walls, evidence=evidence))
            print(json.dumps(row), flush=True)
    for path, sha in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise RuntimeError('Rigid source cache changed')
    levels = []
    for stretch in (.04, .02, .01):
        group = [r for r in rows if r['condition'] in ('C2F', 'F2C') and r['injected_relative_length_peak'] == stretch]
        matches = [m for r in group for m in r['matches_to_injected_structure'] if m is not None]
        levels.append(dict(relative_length_peak=stretch,
            correct_direction_walls=sum(r['correct_direction_walls'] for r in group), expected_walls=12,
            directional_structures_total=sum(r['directional_candidates'] for r in group),
            median_arrival_RMSE_to_injection_s=float(np.median([m['arrival_RMSE_to_injection_s'] for m in matches])) if matches else None,
            minimum_retained_path_fraction=min(m['retained_path_fraction'] for m in matches) if matches else None,
            prediction_cosine_median=float(np.median([r['prediction_cosine_median'] for r in group if r['prediction_cosine_median'] is not None]))))
    negative = {condition:dict(directional_candidates=sum(r['directional_candidates'] for r in rows if r['condition'] == condition))
                for condition in ('local_synchronous', 'global_synchronous', 'rigid_only')}
    save(args.output/'summary.json', dict(levels=levels, nonpropagating_controls=negative,
        clean_reference_directional_candidates=sum(x['directional_candidates'] for x in clean_reference),
        clean_reference_injected_events=12,
        combinations=len(rows), source_files_unchanged=True, thresholds_changed=False,
        patient_candidates_changed=False, clinical_accuracy=None))
    save(args.output/'all_conditions.json', rows)
    save(args.output/'clean_reference.json', clean_reference)
    save(args.output/'protocol.json', dict(
        scope='Downstream A-signal layer only; not image-level tracking validation',
        source_files=sources, fixed_noise_across_amplitude=True,
        stretch_levels=[.04, .02, .01], center_s=30., width_s=4., speed_paths_per_s=.04,
        local_support='five central retained native sections, synchronous arrival',
        known_fixture_matching='Highest retained spatial extent, then lowest arrival error; never use a tiny fragment to claim whole-path recovery',
        same_wall_signal='Same known signal added to both wall-specific frozen residuals',
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        decisions='No new cutoff, classifier or formal event count'))
    print('SUMMARY', json.dumps(dict(levels=levels, nonpropagating_controls=negative)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls-root', type=Path, required=True)
    parser.add_argument('--noise-reference', type=Path, required=True)
    parser.add_argument('--prior-controls', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
