"""Run retained-direction and identity-first trial on existing saved evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.audit_wave_evidence import control_arrays
from experiments.direction_identity_trial import reconcile_identity_then_direction
from experiments.patient_activity_aggregation import candidate_nodes, patient_summary
from experiments.test_activity_strength_trial import read, save


def _control_geometry(outputs, texture):
    _, _, time_s, s, span_s = control_arrays(
        outputs/f'p3_outer_ideal_prior_20261117_v2/texture{texture}_rigid.npz')
    return s, float(span_s), float(np.median(np.diff(time_s)))


def run(output):
    outputs = ROOT/'outputs'
    salience = outputs/'normalized_activity_trial_20261119_v1'
    real_ownership = outputs/'global_ownership_real_20261119_v1'
    trial_ownership = outputs/'global_ownership_trial_20261119_v1'
    maps_root = outputs/'cohort_wave_review_20261118_v1'
    output.mkdir(parents=True, exist_ok=True)

    real_rows = []
    for path in sorted(salience.glob('case_*.json')):
        selected = read(path); original = read(real_ownership/path.name)
        with np.load(maps_root/path.stem/'maps.npz', allow_pickle=False) as maps:
            s = maps['s']; dt = float(np.median(np.diff(maps['interval_center_s'])))
            span = float(np.median(maps['interval_dt_s']))
            old = patient_summary(selected, original, dict(maps))
        trial = reconcile_identity_then_direction(
            candidate_nodes(selected, original), s, span, dt)
        real_rows.append(dict(case=path.stem,
            previous_direction_counts=old['directional_candidate_counts'],
            trial_direction_counts=trial['direction_counts'],
            trial_direction_unresolved_rows=trial['direction_unresolved_rows'],
            trial_cross_wall_identity_pairs=trial['cross_wall_identity_pairs']))

    minimal = []
    for texture in range(3):
        s, span, dt = _control_geometry(outputs, texture)
        for direction in ('C2F', 'F2C'):
            for family, expected in (('single', 1), ('double', 2)):
                name = f'texture{texture}_{family}_{direction}_0.04' + (
                    '_gap20' if family == 'double' else '')
                selected = read(salience/(name+'.json'))
                original = read(trial_ownership/(name+'.json'))
                result = reconcile_identity_then_direction(
                    candidate_nodes(selected, original), s, span, dt)
                passed = (result['activity_rows'] == expected and
                          result['direction_counts'][direction] == expected and
                          result['direction_unresolved_rows'] == 0)
                minimal.append(dict(source=name, expected=expected,
                    direction=direction, passed=passed,
                    result_counts=result['direction_counts']))

    truth = Counter(); duplicated = []
    negative = Counter()
    for path in sorted(salience.glob('texture*.json')):
        original_path = trial_ownership/path.name
        if not original_path.exists():
            continue
        texture = int(re.match(r'texture(\d)', path.name).group(1))
        s, span, dt = _control_geometry(outputs, texture)
        selected = read(path); original = read(original_path)
        result = reconcile_identity_then_direction(
            candidate_nodes(selected, original), s, span, dt)
        if not all('after_matches' in wall['metrics'] for wall in selected['walls']):
            if any(key in path.stem for key in
                   ('local_once', 'rigid', 'global_synchronous', 'local_synchronous')):
                negative.update(files=1,
                    retained_direction_rows=sum(result['direction_counts'].values()),
                    unresolved_rows=result['direction_unresolved_rows'])
            continue
        labels = {}
        for wall in selected['walls']:
            for index, match in enumerate(wall['metrics']['after_matches']):
                if wall['metrics']['after_supported_truth'][index]:
                    labels[f"wall{wall['wall']}/"+match['group_id']] = index
        represented = Counter(); wrong_merge = 0; wrong_direction = 0
        expected_direction = 'C2F' if 'C2F' in path.stem else 'F2C'
        for activity in result['activities']:
            identities = {labels[node] for node in activity['source_nodes'] if node in labels}
            wrong_merge += len(identities) > 1
            for identity in identities:
                represented[identity] += 1
            if identities and activity['direction'] != expected_direction:
                wrong_direction += 1
        identities = set(labels.values())
        truth.update(files=1, identities=len(identities),
            represented_once=sum(represented[i] == 1 for i in identities),
            wrong_merges=wrong_merge, wrong_directions=wrong_direction)
        repeated = {str(i): represented[i] for i in identities if represented[i] != 1}
        if repeated:
            duplicated.append(dict(source=path.stem, representations=repeated))

    summary = dict(experimental=True, formal_results_changed=False,
        real_cases=real_rows,
        real_totals={direction: sum(row['trial_direction_counts'][direction]
                                    for row in real_rows)
                     for direction in ('C2F', 'F2C')},
        real_direction_unresolved_rows=sum(
            row['trial_direction_unresolved_rows'] for row in real_rows),
        minimal_known_controls=dict(conditions=len(minimal),
            passed=sum(row['passed'] for row in minimal), rows=minimal),
        all_existing_truth_support=dict(truth),
        duplicated_truth_support=duplicated,
        nonpropagating_controls=dict(negative),
        interpretation=('retained-direction refit removes unsupported direction labels; '
                        'one existing 10-second double-event condition still duplicates identities'),
        event_count=None, event_frequency_cpm=None)
    save(output/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
