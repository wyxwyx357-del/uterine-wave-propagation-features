"""One frozen phase-priority ablation on all preselected cases and controls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.phase_link_priority import detect_priority
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


def stats(walls):
    events = [e for w in walls for e in w['events'] if e['classification'] in ('C2F', 'F2C')]
    return dict(direction_candidates=len(events),
                single_phase=sum(e['evidence_kind'] == 'single_phase' for e in events),
                order_warning=sum('nonmonotone_arrivals' in e['quality_flags'] for e in events),
                physiological_count=None)


def signatures(walls):
    return [[(e['classification'], e['sections'], e['arrival_time_s'], e['evidence_kind'])
             for e in wall['events']] for wall in walls]


def paired(name, a, valid, t, s, threshold, output, expected=None, cached=None):
    old_walls, new_walls = [], []
    for wall in (0, 1):
        before = detect_radial_events(a[:, wall], valid[:, wall], t, s, noise_floor=threshold)
        after = detect_priority(a[:, wall], valid[:, wall], t, s, noise_floor=threshold)
        np.testing.assert_array_equal(before.pop('filtered_A'), after.pop('filtered_A'))
        for report in (before, after):
            for event in report['events']:
                event['event_id'] = f'wall{wall}_'+event['event_id']
            report['wall'] = wall
        old_walls.append(before); new_walls.append(after)
    if cached is not None and signatures(old_walls) != signatures(cached['walls']):
        raise ValueError('Recomputed baseline differs from saved source candidates')
    result = dict(source=name, baseline=stats(old_walls), candidate=stats(new_walls),
                  candidate_changed=signatures(old_walls) != signatures(new_walls),
                  changed_spatial_boundaries=sum(w['link_priority_diagnostics']['changed_spatial_boundaries'] for w in new_walls),
                  eligible_links=sum(w['link_priority_diagnostics']['eligible_links'] for w in new_walls),
                  baseline_selected_links=sum(w['link_priority_diagnostics']['baseline_selected_link_count'] for w in new_walls),
                  baseline_competing_links=sum(w['link_priority_diagnostics']['baseline_competing_link_count'] for w in new_walls),
                  expected_direction=expected,
                  baseline_correct_direction_walls=sum(any(e['classification'] == expected for e in w['events']) for w in old_walls) if expected else None,
                  candidate_correct_direction_walls=sum(any(e['classification'] == expected for e in w['events']) for w in new_walls) if expected else None)
    save(output/(name+'.json'), dict(summary=result, baseline_walls=old_walls, candidate_walls=new_walls))
    print(json.dumps(result), flush=True)
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = read(args.review_root/'protocol.json')
    calibration = read(args.noise_reference)
    previous_controls = {row['source']:row for row in read(args.prior_controls)}
    sources = {}
    real, controls = [], []
    for case in protocol['selected']:
        name = case['review_id']; folder = args.review_root/name
        maps = load(folder/'maps.npz')
        cached = read(folder/'candidates.json')
        for file in ('tracks.npz', 'maps.npz', 'candidate_maps.npz', 'candidates.json'):
            sources[str(folder/file)] = digest(folder/file)
        real.append(paired(name, maps['A'], maps['valid_mask'], maps['interval_center_s'], maps['s'],
                           protocol['noise_floor_rate'], args.output, cached=cached))
    paths = sorted(args.controls_root.glob('texture*_*.npz'))
    for path in paths:
        texture, condition = path.stem.split('_', 1)
        if condition not in ('C2F', 'F2C', 'rigid', 'local_once'):
            continue
        if digest(path) != previous_controls[path.stem]['source_sha256']:
            raise ValueError('Frozen control cache differs from preceding experiments')
        sources[str(path)] = digest(path)
        a, valid, t, s, _ = control_arrays(path)
        threshold = calibration['leave_one_texture_out_thresholds'][texture[7:]]
        controls.append(paired(path.stem, a, valid, t, s, threshold, args.output,
                               expected=condition if condition in ('C2F', 'F2C') else None))
    if len(controls) != 12:
        raise ValueError('Expected the same three textures and four conditions')
    if any(digest(Path(path)) != value for path, value in sources.items()):
        raise RuntimeError('Original source changed during experiment')
    total = lambda rows, arm, key: sum(row[arm][key] for row in rows)
    negatives = [r for r in controls if r['expected_direction'] is None]
    known = [r for r in controls if r['expected_direction'] is not None]
    summary = dict(real_cases=len(real), real_changed_cases=sum(r['candidate_changed'] for r in real),
        real_baseline_direction_candidates=total(real, 'baseline', 'direction_candidates'),
        real_candidate_direction_candidates=total(real, 'candidate', 'direction_candidates'),
        real_baseline_order_warning=total(real, 'baseline', 'order_warning'),
        real_candidate_order_warning=total(real, 'candidate', 'order_warning'),
        real_changed_spatial_boundaries=sum(r['changed_spatial_boundaries'] for r in real),
        real_baseline_selected_links=sum(r['baseline_selected_links'] for r in real),
        real_baseline_competing_links=sum(r['baseline_competing_links'] for r in real),
        known_baseline_direction_walls=sum(r['baseline_correct_direction_walls'] for r in known),
        known_candidate_direction_walls=sum(r['candidate_correct_direction_walls'] for r in known),
        negative_baseline_direction_candidates=total(negatives, 'baseline', 'direction_candidates'),
        negative_candidate_direction_candidates=total(negatives, 'candidate', 'direction_candidates'),
        filtered_arrays_identical=True, all_baselines_reproduced=True, sources_unchanged=True,
        new_rejection_threshold=False, formal_accuracy=None, adoption='experimental_only')
    n = summary['real_baseline_selected_links']
    summary['real_baseline_noncompeting_link_fraction'] = (1-summary['real_baseline_competing_links']/n) if n else None
    save(args.output/'summary.json', summary)
    save(args.output/'protocol.json', dict(
        variant='Full-phase cosine x duration agreement x arrival-order agreement x existing delay proximity; geometric combination ranks eligible links only',
        eligibility='Unchanged same sign, temporal overlap and existing maximum delay',
        case_specific_parameters=False, parameter_sweep=False, physiological_features_added=False,
        A0_changed=False, noise_reference_sha256=digest(args.noise_reference),
        source_files=sources, source_code_sha256={name:digest(ROOT/name) for name in (
            'uterine_wave/radial_events.py', 'experiments/phase_link_priority.py',
            'experiments/test_phase_link_priority_cohort.py')}))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-root', type=Path, required=True)
    parser.add_argument('--controls-root', type=Path, required=True)
    parser.add_argument('--noise-reference', type=Path, required=True)
    parser.add_argument('--prior-controls', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
