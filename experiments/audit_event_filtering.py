"""Paired event-layer filter ablations, not replacement patient features.

The frozen amplitude/ER filters, tracking, time axis and noise thresholds stay
unchanged. Reflected FFT tests the periodic boundary assumption; a fixed
second-order Butterworth tests a gradual frequency response. Both are
engineering diagnostics, not literature-derived physiological classifiers.
SciPy is an optional dependency of this experiment only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uterine_wave import radial_events
from uterine_wave.amplitude import band_limited_signal
from uterine_wave.io import jsonable
from experiments.audit_wave_evidence import control_arrays
from experiments.partial_phase_events import detect_partial
from experiments.test_weak_wave_signals import radial_fixture, match_known_structure


def reflected_fft(values, times, band_cpm=(.5, 5.)):
    """Even reflection without repeating endpoints, then original FFT/crop.

    Extra samples are a boundary assumption, never observed motion. The longer
    transform changes frequency bins and does not add acquisition information.
    """
    extended = np.concatenate([values, values[-2:0:-1]], axis=0)
    extended_times = times[0]+np.arange(len(extended))*np.median(np.diff(times))
    return band_limited_signal(extended, extended_times, band_cpm)[:len(times)]


def gradual_bandpass(values, times, band_cpm=(.5, 5.)):
    """Fixed order=2 SOS, SciPy default odd padding, forward/backward pass.

    Nominal cutoffs equal the existing band, but effective responses differ.
    No order/cutoff/padding search; no claim of identical band-limited SD.
    """
    from scipy.signal import butter, sosfiltfilt
    sos = butter(2, np.asarray(band_cpm)/60, btype='bandpass',
                 fs=1/np.median(np.diff(times)), output='sos')
    return sosfiltfilt(sos, values-values.mean(axis=0), axis=0)


FILTERS = dict(periodic_fft=band_limited_signal,
               reflected_fft=reflected_fft, gradual_bandpass=gradual_bandpass)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False,
                               indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def known_match(report, direction, truth_arrival, clean_filtered, s):
    """Same prespecified whole-path matching as the fixed-noise experiment.

    A direction hit alone is not recovery: retain timing and spatial support.
    Ground-truth injection is used only for evaluation, never for detection.
    """
    options = []
    for event in report['events']:
        if event['classification'] != direction:
            continue
        ids = event['sections']
        expected = truth_arrival[ids] if event['evidence_kind'] == 'paired_phase' else (
            report['time_s'][np.argmax(event['leading_polarity']*clean_filtered[:, ids], axis=0)])
        error = np.asarray(event['arrival_time_s'])-expected
        options.append(dict(event_id=event['event_id'],
            retained_path_fraction=float(np.ptp(s[ids])/np.ptp(s)),
            arrival_RMSE_to_injection_s=float(np.sqrt(np.mean(error**2))),
            evidence_kind=event['evidence_kind'], sections=ids))
    return match_known_structure(options)


def compare(source, family, a, valid, t, s, floor, *, clean=None,
            truth_arrival=None, direction=None, known_nonpropagating=False):
    output = dict(source=source, family=family, methods={})
    # Both detector versions are evaluated independently under each filter.
    # This prevents censored-peak handling from being a hidden confound.
    for method, function in FILTERS.items():
        versions = {}
        with patch.object(radial_events, 'band_limited_signal', function):
            for version, detector in [('original', radial_events.detect_radial_events),
                                      ('observable_partial', detect_partial)]:
                walls = []
                for wall in (0, 1):
                    report = detector(a[:, wall], valid[:, wall], t, s, noise_floor=floor)
                    report.pop('filtered_A')
                    report['time_s'] = t
                    match = known_match(report, direction, truth_arrival, function(clean, t), s) if clean is not None else None
                    report.pop('time_s')
                    walls.append(dict(wall=wall, report=report,
                                      known_injection_match=match))
                versions[version] = dict(walls=walls,
                    directional_structures=sum(w['report']['traveling_candidate_count'] for w in walls),
                    expected_direction_walls=sum(any(e['classification'] == direction for e in w['report']['events'])
                                                 for w in walls) if direction else None,
                    known_nonpropagating=known_nonpropagating)
        # Patch must be local; formal amplitude code is never patched.
        if radial_events.band_limited_signal is not band_limited_signal:
            raise RuntimeError('Experimental filter escaped its context')
        output['methods'][method] = versions
    return output


def aggregate(rows):
    result = {}
    injected_truth = any(r['family'] in ('fixed_noise_signal', 'boundary_stress_signal')
                         and r.get('condition') in ('C2F', 'F2C') for r in rows)
    for method in FILTERS:
        result[method] = {}
        for version in ('original', 'observable_partial'):
            data = [row['methods'][method][version] for row in rows]
            matches = [w['known_injection_match'] for row in data for w in row['walls']
                       if w['known_injection_match'] is not None]
            result[method][version] = dict(
                directional_structures=sum(row['directional_structures'] for row in data),
                direction_hit_walls=sum(row['expected_direction_walls'] or 0 for row in data)
                    if any(row['expected_direction_walls'] is not None for row in data) else None,
                nonpropagating_directional_structures=sum(row['directional_structures'] for row in data
                    if row['known_nonpropagating']) if any(row['known_nonpropagating'] for row in data) else None,
                matched_injected_walls=len(matches) if injected_truth else None,
                minimum_retained_path_fraction=min((m['retained_path_fraction'] for m in matches), default=None),
                median_arrival_RMSE_to_injection_s=float(np.median([m['arrival_RMSE_to_injection_s'] for m in matches])) if matches else None)
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = read(args.review_root/'protocol.json')
    calibration = read(args.noise_reference)
    frozen_sources = read(args.partial_trial/'protocol.json')['source_files']
    formal_code = {name:digest(ROOT/'uterine_wave'/name) for name in (
        'tracking.py', 'deformation.py', 'amplitude.py', 'spectrum.py', 'radial_events.py', 'events.py', 'patient.py')}
    sources, real, controls, signals, boundary_stress = {}, [], [], [], []
    def check(path):
        sha = digest(path)
        # Compare the same cached data against the previous run's hashes.
        previous = frozen_sources.get(str(path))
        if previous is None or previous != sha:
            raise ValueError(f'Input missing from frozen protocol or changed: {path}')
        sources[str(path)] = sha
    def emit(row, group):
        group.append(row)
        save(args.output/(row['source']+'.json'), row)
        print(row['source'], flush=True)
    for selected in protocol['selected']:
        folder = args.review_root/selected['review_id']
        for name in ('tracks.npz', 'maps.npz', 'candidates.json'):
            check(folder/name)
        with np.load(folder/'maps.npz', allow_pickle=False) as z:
            row = compare(selected['review_id'], 'real_development', z['A'], z['valid_mask'],
                          z['interval_center_s'], z['s'], protocol['noise_floor_rate'])
        emit(row, real)
    for path in sorted(args.controls_root.glob('texture*_*.npz')):
        texture, condition = path.stem.split('_', 1)
        if condition not in ('C2F', 'F2C', 'rigid', 'local_once'):
            continue
        check(path)
        a, valid, t, s, dt = control_arrays(path)
        floor = calibration['leave_one_texture_out_thresholds'][texture[7:]]
        emit(compare(path.stem, 'image_warp_control', a, valid, t, s, floor,
                     direction=condition if condition in ('C2F', 'F2C') else None,
                     known_nonpropagating=condition in ('rigid', 'local_once')), controls)
        if condition != 'rigid':
            continue
        definitions = [(d, amp) for d in ('C2F', 'F2C') for amp in (.04, .02, .01)]
        definitions += [(d, .04) for d in ('local_synchronous', 'global_synchronous', 'rigid_only')]
        for kind, amplitude in definitions:
            clean, arrival, _ = radial_fixture(t, s, dt, kind, amplitude)
            is_wave = kind in ('C2F', 'F2C')
            row = compare(f'{texture}_{kind}_{amplitude:.2f}', 'fixed_noise_signal',
                a+clean[:, None], valid, t, s, floor, clean=clean if is_wave else None,
                truth_arrival=arrival if is_wave else None, direction=kind if is_wave else None,
                known_nonpropagating=not is_wave)
            row.update(condition=kind, injected_length_peak=amplitude if kind != 'rigid_only' else 0.)
            emit(row, signals)
        # Same waveform/noise/speed, shifted by +/-8 s without wrapping. These
        # fixed stress positions test reliance on the original centered pulse;
        # they are not filter parameters or a search for a favorable setting.
        for shift in (-8., 8.):
            for kind in ('C2F', 'F2C'):
                for amplitude in (.04, .02, .01):
                    clean, arrival, _ = radial_fixture(t-shift, s, dt, kind, amplitude)
                    row = compare(f'{texture}_{kind}_{amplitude:.2f}_shift{shift:+.0f}',
                        'boundary_stress_signal', a+clean[:, None], valid, t, s, floor,
                        clean=clean, truth_arrival=arrival+shift, direction=kind)
                    row.update(condition=kind, injected_length_peak=amplitude,
                               true_center_s=30.+shift)
                    emit(row, boundary_stress)
    if len(real) != 12 or len(controls) != 12 or len(signals) != 27:
        raise ValueError('Expected the frozen 12-case, 12-control, 27-fixture set')
    # Independently reproduce the previous periodic-FFT counts, including
    # partial structures that were evidence-only, not new event identities.
    prior = read(args.partial_trial/'summary.json')
    for group, key in [(real, 'real'), (controls, 'frozen_controls'), (signals, 'signal_fixtures')]:
        counts = aggregate(group)['periodic_fft']
        if (counts['original']['directional_structures'] != prior[key]['baseline'] or
            counts['observable_partial']['directional_structures'] != prior[key]['all_partial_direction_structures']):
            raise ValueError('Periodic baseline differs from the previous paired trial')
    if any(digest(Path(path)) != sha for path, sha in sources.items()):
        raise RuntimeError('Source data changed')
    if any(digest(ROOT/'uterine_wave'/name) != sha for name, sha in formal_code.items()):
        raise RuntimeError('Formal feature code changed during diagnostic')
    if len(boundary_stress) != 36:
        raise ValueError('Expected 36 fixed shifted-wave stress conditions')
    summary = dict(real=aggregate(real), frozen_controls=aggregate(controls),
        signal_fixtures=aggregate(signals), weak_levels={str(amp):aggregate([
            r for r in signals if r['condition'] in ('C2F','F2C') and r['injected_length_peak'] == amp])
            for amp in (.04,.02,.01)},
        boundary_stress=aggregate(boundary_stress),
        boundary_stress_by_center={str(center):aggregate([r for r in boundary_stress
            if r['true_center_s'] == center]) for center in (22.,38.)},
        patient_accuracy=None, formal_event_count=None,
        adoption='sandbox_only; assess direction, spatial support and false structures jointly',
        source_files_unchanged=True, formal_features_unchanged=True)
    save(args.output/'summary.json', summary)
    import scipy
    save(args.output/'protocol.json', dict(source_files=sources,
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        code_sha256=digest(Path(__file__)), filter_methods=list(FILTERS),
        formal_code_sha256=formal_code,
        reference_sha256={str(p):digest(p) for p in (args.noise_reference,
            args.partial_trial/'protocol.json', args.review_root/'protocol.json')},
        nominal_band_cpm=[.5,5.], gradual_order=2, gradual_padding='SciPy default odd padding',
        boundary_stress_centers_s=[22.,38.],
        noise_thresholds='Frozen original absolute thresholds, not tuned or recalibrated per filter',
        interpretation='Diagnostic response/boundary sensitivity; filter-dependent noise calibration remains a limitation',
        missing='Each detector filters each continuous valid column-run independently; no gap filling',
        direction_hit_is_not_whole_wave_recovery=True, adoption='none'))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-root', type=Path, required=True)
    parser.add_argument('--controls-root', type=Path, required=True)
    parser.add_argument('--noise-reference', type=Path, required=True)
    parser.add_argument('--partial-trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
