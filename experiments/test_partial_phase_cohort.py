"""Test observable boundary evidence and source-preserving ownership proposals."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.partial_phase_events import detect_partial, anchor_to_existing_candidates
from experiments.audit_wave_evidence import control_arrays
from experiments.test_weak_wave_signals import radial_fixture
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


def signature(report):
    return [(e['classification'], e['sections'], e['arrival_time_s'], e['evidence_kind'])
            for e in report['events']]


def compare(name, a, valid, t, s, floor, output, cached=None, expected=None):
    records, walls = [], []
    for wall in (0, 1):
        before = detect_radial_events(a[:, wall], valid[:, wall], t, s, noise_floor=floor)
        after = detect_partial(a[:, wall], valid[:, wall], t, s, noise_floor=floor)
        np.testing.assert_array_equal(before.pop('filtered_A'), after.pop('filtered_A'))
        if cached is not None and signature(before) != signature(cached['walls'][wall]):
            raise ValueError('Saved baseline does not reproduce')
        ownership = anchor_to_existing_candidates(before, after)
        for fragment in after['fragments']:
            fragment['wall'] = wall
        for proposal in ownership['existing_candidate_groups']:
            proposal['source_candidate_references'] = [dict(wall=wall, event_id=f'wall{wall}_'+eid)
                                                        for eid in proposal['source_candidate_ids']]
        old_ids = {e['event_id'] for e in before['events']}
        represented = [eid for g in ownership['existing_candidate_groups'] for eid in g['source_candidate_ids']]
        if set(represented) != old_ids or len(represented) != len(old_ids):
            raise ValueError('Original evidence must remain represented exactly once')
        record = dict(wall=wall, baseline_direction_candidates=before['traveling_candidate_count'],
            partial_evidence_direction_structures=after['traveling_candidate_count'],
            existing_direction_group_proposals=ownership['existing_direction_group_proposals'],
            shared_identity_proposals=sum(len(g['source_candidate_ids']) > 1 for g in ownership['existing_candidate_groups']),
            observed_boundary_nodes=len(after['observed_boundary_nodes']),
            unresolved_boundary_segments=len(after['unresolved_boundary_segments']),
            additional_support_groups=len(ownership['additional_support_groups']),
            source_candidates_preserved=True, filtered_arrays_identical=True,
            expected_direction_in_baseline=any(e['classification'] == expected for e in before['events']) if expected else None,
            expected_direction_in_partial_evidence=any(e['classification'] == expected for e in after['events']) if expected else None)
        records.append(record)
        walls.append(dict(wall=wall, original=before, partial_evidence=after, ownership=ownership))
    summary = dict(source=name, baseline_direction_candidates=sum(r['baseline_direction_candidates'] for r in records),
        partial_evidence_direction_structures=sum(r['partial_evidence_direction_structures'] for r in records),
        existing_direction_group_proposals=sum(r['existing_direction_group_proposals'] for r in records),
        shared_identity_proposals=sum(r['shared_identity_proposals'] for r in records),
        expected_direction=expected,
        expected_baseline_walls=sum(r['expected_direction_in_baseline'] for r in records) if expected else None,
        expected_partial_walls=sum(r['expected_direction_in_partial_evidence'] for r in records) if expected else None,
        formal_event_count=None)
    save(output/(name+'.json'), dict(summary=summary, walls=walls))
    print(json.dumps(summary), flush=True)
    return summary


def export_real_tables(output, real):
    fragments, identities, patients = [], [], []
    for item in real:
        source = item['source']
        data = read(output/(source+'.json'))
        for wall_data in data['walls']:
            w = wall_data['wall']; observed = wall_data['partial_evidence']
            pieces = {p['fragment_id']:p for p in observed['fragments']}
            originals = {e['event_id']:e for e in wall_data['original']['events']}
            ownership = wall_data['ownership']
            def emit(piece, proposal, role, source_ids):
                fragments.append(dict(source=source, wall=w, fragment_id=piece['fragment_id'],
                    activity_proposal_id=proposal, support_role=role,
                    source_candidates=json.dumps([f'wall{w}_'+eid for eid in source_ids]),
                    observed_start_s=piece['observed_start_s'], observed_end_s=piece['observed_end_s'],
                    structure_start_s=piece['structure_start_s'], structure_end_s=piece['structure_end_s'],
                    start_unknown=piece['boundary_status']['start_unknown'],
                    end_unknown=piece['boundary_status']['end_unknown'],
                    sections=json.dumps(piece['sections']), arrival_time_s=json.dumps(piece['arrival_time_s']),
                    direction=piece['direction'], identity_reviewed=False))
            for group in ownership['existing_candidate_groups']:
                group_id = f'wall{w}_'+group['proposal_id']
                identities.append(dict(source=source, wall=w, activity_proposal_id=group_id,
                    source_candidates=json.dumps([f'wall{w}_'+eid for eid in group['source_candidate_ids']]),
                    fragment_ids=json.dumps(group['observed_fragment_ids']), ownership_status=group['ownership_status'],
                    identity_reviewed=False, formal_event_count_eligible=False))
                if group['observed_fragment_ids']:
                    for pid in group['observed_fragment_ids']:
                        emit(pieces[pid], group_id, 'existing_candidate_support', group['source_candidate_ids'])
                else:
                    for eid in group['source_candidate_ids']:
                        e = originals[eid]
                        piece = dict(fragment_id='original_'+eid, observed_start_s=e['start_s'],
                            observed_end_s=e['end_s'], structure_start_s=e['start_s'], structure_end_s=e['end_s'],
                            boundary_status=dict(start_unknown=False, end_unknown=False),
                            sections=e['sections'], arrival_time_s=e['arrival_time_s'], direction=e['classification'])
                        emit(piece, group_id, 'original_support_no_unique_new_mapping', [eid])
            for group in ownership['additional_support_groups']:
                for pid in group['fragment_ids']:
                    emit(pieces[pid], None, 'additional_unassigned_support', [])
        patients.append(dict(source=source, baseline_direction_candidates=item['baseline_direction_candidates'],
            direction_identity_proposals=item['existing_direction_group_proposals'],
            confirmed_event_count=None, event_frequency_cpm=None, status='unvalidated_identity_proposals'))
    for name, rows in [('fragment_measurements.csv', fragments), ('event_identity_proposals.csv', identities),
                       ('patient_event_status.csv', patients)]:
        with (output/name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    real_protocol = read(args.review_root/'protocol.json')
    calibration = read(args.noise_reference)
    previous = {r['source']:r for r in read(args.prior_controls)}
    sources, real, controls, signals = {}, [], [], []
    for row in real_protocol['selected']:
        folder = args.review_root/row['review_id']
        for name in ('tracks.npz', 'maps.npz', 'candidates.json'):
            sources[str(folder/name)] = digest(folder/name)
        maps = load(folder/'maps.npz')
        real.append(compare(row['review_id'], maps['A'], maps['valid_mask'], maps['interval_center_s'], maps['s'],
                            real_protocol['noise_floor_rate'], args.output, cached=read(folder/'candidates.json')))
    for path in sorted(args.controls_root.glob('texture*_*.npz')):
        texture, condition = path.stem.split('_', 1)
        if condition not in ('C2F', 'F2C', 'rigid', 'local_once'):
            continue
        if digest(path) != previous[path.stem]['source_sha256']:
            raise ValueError('Frozen control differs')
        sources[str(path)] = digest(path)
        a, valid, t, s, dt = control_arrays(path)
        floor = calibration['leave_one_texture_out_thresholds'][texture[7:]]
        controls.append(compare(path.stem, a, valid, t, s, floor, args.output,
                                 expected=condition if condition in ('C2F', 'F2C') else None))
        if condition == 'rigid':
            definitions = [(d, amp) for d in ('C2F', 'F2C') for amp in (.04, .02, .01)]
            definitions += [(d, .04) for d in ('local_synchronous', 'global_synchronous', 'rigid_only')]
            for kind, amplitude in definitions:
                injected, _, _ = radial_fixture(t, s, dt, kind, amplitude)
                signals.append(compare(f'{texture}_{kind}_{amplitude:.2f}', a+injected[:, None], valid, t, s,
                    floor, args.output, expected=kind if kind in ('C2F', 'F2C') else None))
    if len(controls) != 12 or len(signals) != 27:
        raise ValueError('Expected the same frozen control set and signal fixtures')
    if any(digest(Path(path)) != sha for path, sha in sources.items()):
        raise RuntimeError('Original source changed')
    def aggregate(rows):
        return dict(cases=len(rows), baseline=sum(r['baseline_direction_candidates'] for r in rows),
            all_partial_direction_structures=sum(r['partial_evidence_direction_structures'] for r in rows),
            anchored_direction_group_proposals=sum(r['existing_direction_group_proposals'] for r in rows),
            proposed_shared_identities=sum(r['shared_identity_proposals'] for r in rows))
    known = [r for r in controls if r['expected_direction']]
    weak = [r for r in signals if r['expected_direction']]
    negative = [r for r in controls if not r['expected_direction']]
    summary = dict(real=aggregate(real), frozen_controls=aggregate(controls), signal_fixtures=aggregate(signals),
        known_baseline_direction_walls=sum(r['expected_baseline_walls'] for r in known),
        known_partial_direction_walls=sum(r['expected_partial_walls'] for r in known),
        signal_baseline_direction_walls=sum(r['expected_baseline_walls'] for r in weak),
        signal_partial_direction_walls=sum(r['expected_partial_walls'] for r in weak),
        original_negative_baseline=sum(r['baseline_direction_candidates'] for r in negative),
        original_negative_anchored_proposals=sum(r['existing_direction_group_proposals'] for r in negative),
        sources_unchanged=True, all_original_candidates_preserved=True, filtered_arrays_identical=True,
        formal_event_count=None, adoption='sandbox_only_no_validated_patient_counts')
    save(args.output/'summary.json', summary)
    export_real_tables(args.output, real)
    save(args.output/'protocol.json', dict(source_files=sources, A0_changed=False,
        new_rejection_threshold=False, new_boundary_information='Support only; no automatic new independent events',
        ownership='Unique exact phase-node coverage; multiple or opposite-direction mappings remain unresolved',
        code_sha256={name:digest(ROOT/name) for name in ('experiments/partial_phase_events.py',
                                                       'experiments/test_partial_phase_cohort.py')}))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-root', type=Path, required=True)
    parser.add_argument('--controls-root', type=Path, required=True)
    parser.add_argument('--noise-reference', type=Path, required=True)
    parser.add_argument('--prior-controls', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
