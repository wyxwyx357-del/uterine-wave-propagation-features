"""Apply held-out phase checks to all existing cases and frozen controls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.phase_prediction_evidence import analyze_prediction
from experiments.audit_wave_evidence import control_arrays
from uterine_wave.radial_events import detect_radial_events
from uterine_wave.io import jsonable


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def save(path, value):
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(events):
    def values(key):
        a = [e[key] for e in events if e[key] is not None]
        return dict(count=len(a), median=float(np.median(a)) if a else None,
                    minimum=float(min(a)) if a else None, maximum=float(max(a)) if a else None)
    return dict(candidates=len(events), arrival_RMSE_s=values('arrival_loo_RMSE_s'),
                arrival_RMSE_over_range=values('arrival_loo_RMSE_over_range'),
                prediction_cosine=values('predicted_shape_cosine_mean'),
                peak_aligned_cosine=values('peak_aligned_shape_cosine_mean'))


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = read(args.review_root/'protocol.json')
    sources, real, case_rows, controls = {}, [], [], []
    for case in protocol['selected']:
        rid = case['review_id']; folder = args.review_root/rid
        for name in ('candidates.json', 'candidate_maps.npz'):
            sources[str(folder/name)] = digest(folder/name)
        result = analyze_prediction(read(folder/'candidates.json'), load(folder/'candidate_maps.npz'))
        save(args.output/(rid+'.json'), result)
        row = dict(source=rid, **summarize(result['events']))
        case_rows.append(row)
        real.extend(dict(source=rid, **e) for e in result['events'])
        print(json.dumps(row), flush=True)
    calibration = read(args.noise_reference)
    previous = {r['source']:r for r in read(args.prior_controls)}
    for path in sorted(args.controls_root.glob('texture*_*.npz')):
        texture, condition = path.stem.split('_', 1)
        if condition not in ('C2F', 'F2C', 'rigid', 'local_once'):
            continue
        if digest(path) != previous[path.stem]['source_sha256']:
            raise ValueError('Frozen control cache differs')
        sources[str(path)] = digest(path)
        a, valid, t, s, actual_interval = control_arrays(path)
        walls, filtered = [], np.full_like(a, np.nan)
        for wall in (0, 1):
            r = detect_radial_events(a[:, wall], valid[:, wall], t, s,
                noise_floor=calibration['leave_one_texture_out_thresholds'][texture[7:]])
            filtered[:, wall] = r.pop('filtered_A')
            for event in r['events']:
                event['event_id'] = f'wall{wall}_'+event['event_id']
            walls.append(dict(wall=wall, **r))
        arrays = dict(filtered_A=filtered, candidate_valid_mask=valid&np.isfinite(filtered),
                      interval_center_s=t, s=s)
        result = analyze_prediction(dict(walls=walls), arrays)
        save(args.output/(path.stem+'.json'), result)
        controls.append(dict(source=path.stem, condition=condition,
                             actual_interval_s=actual_interval, events=result['events']))
    if len(controls) != 12:
        raise ValueError('Expected the same twelve controls')
    known = [e for r in controls if r['condition'] in ('C2F', 'F2C') for e in r['events']]
    negative = [e for r in controls if r['condition'] in ('rigid', 'local_once') for e in r['events']]
    if any(digest(Path(path)) != value for path, value in sources.items()):
        raise RuntimeError('Original sources changed')
    summary = dict(real_cases=len(case_rows), real=summarize(real), known=summarize(known),
                   rigid_and_local_once=summarize(negative), decision_threshold=None,
                   candidate_labels_changed=False, formal_accuracy=None, sources_unchanged=True)
    save(args.output/'summary.json', summary)
    save(args.output/'per_case.json', case_rows)
    save(args.output/'all_real_events.json', real)
    save(args.output/'controls.json', controls)
    save(args.output/'protocol.json', dict(source_files=sources,
        code_sha256={name:digest(ROOT/name) for name in ('experiments/phase_prediction_evidence.py',
                                                        'experiments/audit_phase_prediction.py')},
        prediction='Leave one spatial position out, fit peak time vs full-path s on remaining positions, predict normalized full-phase template without using held-out arrival',
        no_interpolation=True, no_new_threshold=True, A0_changed=False,
        physiological_features_added=False, case_specific_parameters=False,
        limitations=['Phase selection originally used all positions; this is not a fully independent test.',
                     'Acceleration, changing wave shape and tracking errors can also reduce prediction.',
                     'Smooth band-limited noise can remain predictable; high similarity alone does not prove a wave.',
                     'Clipping at the held-out continuous-run boundary is recorded per prediction.']))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-root', type=Path, required=True)
    parser.add_argument('--controls-root', type=Path, required=True)
    parser.add_argument('--noise-reference', type=Path, required=True)
    parser.add_argument('--prior-controls', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
