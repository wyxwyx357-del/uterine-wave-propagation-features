"""Fixed 87 controls: provenance-only primary eligibility and identity ranges."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.provenance_count_trial import provenance_nodes, trial_count
from experiments.global_event_ownership import reconcile
from experiments.normalized_activity import normalize_magnitude, score_salience
from experiments.activity_strength import filtered_runs
from experiments.patient_activity_aggregation import observation_union
from experiments.audit_wave_evidence import control_arrays
from experiments.test_double_event_signals import double_fixture, raw_truth, phase_identity
from experiments.test_weak_wave_signals import radial_fixture
from experiments.test_activity_strength_trial import read, save


def truth_evaluation(result, truth):
    """Attach filter-independent signed-phase truth after the trial completes.

    Known length maxima require + then -, without imposing this role on the
    detector. Count representations and timing error, not a tuned accuracy.
    """
    nodes = {n['node_id']: n for n in result['provenance_nodes']}

    def describe(activities):
        rows = []
        for activity in activities:
            if not activity['propagation_supported']:
                continue
            support = {}
            for block in activity['block_evidence']:
                if not block['propagation_supported']:
                    continue
                node = nodes[block['node_id']]
                for c in block['sections']:
                    phases = node['profile'][c]
                    labels = [phase_identity(c, p['sign'], p['peak_s'], truth) for p in phases]
                    paired = node['evidence_kind'] == 'paired_phase'
                    role = not paired or [p['sign'] for p in phases] == [1, -1]
                    if not labels or None in labels or len(set(labels)) != 1 or not role:
                        continue
                    identity = labels[0]
                    marker = ('arrival_s' if paired else
                              'positive_peak_s' if phases[0]['sign'] > 0 else 'negative_peak_s')
                    expected = truth['columns'][c][identity][marker]
                    support.setdefault(identity, []).append(dict(
                        section=c, error_s=node['arrival'][c]-expected))
            labels = []
            for identity, points in support.items():
                ids = np.array(sorted({p['section'] for p in points}), int)
                blocks = np.split(ids, np.flatnonzero(np.diff(ids) != 1)+1)
                if max((len(b) for b in blocks), default=0) >= 3:
                    labels.append(dict(truth_id=identity, native_section_fraction=len(ids)/len(truth['columns']),
                        arrival_RMSE_s=float(np.sqrt(np.mean([p['error_s']**2 for p in points])))))
            rows.append(dict(activity_id=activity['activity_id'], source_nodes=activity['source_nodes'],
                             truth_support=labels, has_observed_boundary_phase=any(
                                 nodes[n].get('observed_boundary_phase_nodes') for n in activity['source_nodes'])))
        return rows

    before = describe(result['baseline']['activities'])
    after = describe(result['primary']['activities'])
    represented = lambda items, i: any(any(p['truth_id'] == i for p in r['truth_support']) for r in items)
    lost = [i for i in range(truth['uniform_observable_count'])
            if represented(before, i) and not represented(after, i)]
    return dict(observable_truth_count=truth['uniform_observable_count'],
                before=before, after=after,
                before_representations={str(i): sum(any(x['truth_id'] == i for x in r['truth_support']) for r in before)
                                        for i in range(truth['uniform_observable_count'])},
                after_representations={str(i): sum(any(x['truth_id'] == i for x in r['truth_support']) for r in after)
                                       for i in range(truth['uniform_observable_count'])},
                previously_supported_truth_ids_lost=lost,
                lost_truth_ids_with_boundary_evidence=[i for i in lost if any(
                    r['has_observed_boundary_phase'] and any(p['truth_id'] == i for p in r['truth_support'])
                    for r in before)],
                evaluation_only=True)


def run():
    root = ROOT/'outputs'
    output = root/'provenance_count_trial_20261119'
    output.mkdir(exist_ok=True)
    paths = sorted((root/'global_ownership_trial_20261119_v1').glob('texture*.json'))
    if len(paths) != 87:
        raise ValueError('Expected the existing 87 controls')
    frozen = ['experiments/activity_counting.py', 'experiments/global_event_ownership.py',
              'experiments/normalized_activity.py', 'uterine_wave/radial_events.py']
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    code_hashes = {p: digest(ROOT/p) for p in frozen}
    rows, source_hashes = [], {}
    for path in paths:
        name = path.stem
        parent_path = root/'event_input_smoothing_20261118_v1'/path.name
        local_path = root/'candidate_split_trial_20261119_v1'/path.name
        old_path = root/'significant_activity_counts_20261119_v2'/path.name
        old, parent, local = read(path), read(parent_path), read(local_path)
        for source in (path, parent_path, local_path, old_path):
            source_hashes[str(source.relative_to(ROOT))] = digest(source)
        texture, family = parent['texture'], old['family']
        control = root/'p3_outer_ideal_prior_20261117_v2'/(name+'.npz' if family == 'image_control' else texture+'_rigid.npz')
        source_hashes[str(control.relative_to(ROOT))] = digest(control)
        a, v, t, s, span = control_arrays(control)
        truth = None
        if family == 'double':
            clean = double_fixture(t, s, span, parent['expected_direction'], parent['gap_s'], parent['second_amplitude'])[0]
            a = a + clean[:, None]
            truth = raw_truth(clean, t)
        elif family == 'single_or_negative':
            kind, amp = name[len(texture+'_single_'):].rsplit('_', 1)
            clean = radial_fixture(t, s, span, kind, float(amp))[0]
            a = a + clean[:, None]
            if parent['expected_direction']:
                truth = raw_truth(clean, t)
        normalized, metadata = normalize_magnitude(filtered_runs(a, v, t), v)
        ownership, selected = dict(walls=[]), dict(walls=[])
        for w in (0, 1):
            report = parent['methods']['periodic_frozen']['walls'][w]['report']
            splits = local['walls'][w]['versions']['3']['proposals']
            rec = reconcile(report, splits, t, v[:, w], measurement_span_s=span)
            ownership['walls'].append(dict(reconciliation=rec))
            selected['walls'].append(dict(wall=w, scores=[score_salience(
                e, rec['source_evidence'], normalized[:, w], v[:, w], t) for e in rec['event_proposals']]))
        nodes = provenance_nodes(selected, ownership)
        for node in nodes:
            report = parent['methods']['periodic_frozen']['walls'][node['wall']]['report']
            node['observed_boundary_phase_nodes'] = [deepcopy(b) for b in report.get('observed_boundary_nodes', [])
                if b['column'] in node['profile'] and any(
                    p['peak_s'] == b['peak_time_s'] for p in node['profile'][b['column']])]
            node['boundary_annotation_scope'] = 'saved periodic boundary nodes; not a dedicated truncated-wave test'
        seconds = observation_union(dict(A=a, valid_mask=v, interval_start_s=t-span/2,
                                         interval_end_s=t+span/2))['seconds']
        result = trial_count(nodes, s, span, float(np.median(np.diff(t))), seconds)
        prior = read(old_path)
        if result['baseline'] != {k: prior[k] for k in result['baseline']}:
            raise ValueError('Baseline replay differs from the frozen formal result: '+name)
        result.update(source=name, normalization=metadata)
        result['evaluation_only'] = truth_evaluation(result, truth) if truth else None
        result['deferred_boundary_phase_evidence'] = [dict(node_id=n['node_id'], phase=b)
            for n in result['deferred_evidence'] for b in n['observed_boundary_phase_nodes']]
        save(output/path.name, result)
        rows.append(dict(source=name, old_count=prior['significant_propagation_activity_count'],
            primary_rows=result['primary_candidate_row_count'],
            conditional_range=result['identity_range']['conditional_candidate_identity_count_range'],
            unresolved_identity=result['identity_range']['identity_remains_unresolved'],
            deferred_rows=len(result['deferred_evidence']),
            deferred_old_propagating=sum(n['old_propagation_supported'] for n in result['deferred_evidence']),
            no_shared_section_pairs=len(result['identity_range']['no_shared_section_pairs']),
            evaluation=result['evaluation_only']))
    categories = dict(
        local_once=lambda n: n.endswith('_local_once'),
        single_strong=lambda n: '_single_' in n and n.endswith(('C2F_0.04', 'F2C_0.04')),
        double_10_strong=lambda n: '_double_' in n and n.endswith('_0.04_gap10'),
        double_15_strong=lambda n: '_double_' in n and n.endswith('_0.04_gap15'),
        double_20_strong=lambda n: '_double_' in n and n.endswith('_0.04_gap20'),
        double_15_weak=lambda n: '_double_' in n and n.endswith('_0.01_gap15'),
        double_20_weak=lambda n: '_double_' in n and n.endswith('_0.01_gap20'),
        synchronous=lambda n: 'synchronous' in n,
        rigid=lambda n: n.endswith('_rigid') or 'rigid_only' in n)
    aggregates = []
    for condition, choose in categories.items():
        subset = [r for r in rows if choose(r['source'])]
        row = dict(condition=condition, conditions=len(subset),
                   **{k: sum(r[k] for r in subset) for k in ('old_count', 'primary_rows', 'deferred_rows', 'deferred_old_propagating')},
                   conditional_range=[sum(r['conditional_range'][i] for r in subset) for i in (0, 1)])
        aggregates.append(row)
        print(row, flush=True)
    if any(digest(ROOT/p) != h for p, h in code_hashes.items()):
        raise ValueError('Frozen algorithm changed during trial')
    if any(digest(ROOT/p) != h for p, h in source_hashes.items()):
        raise ValueError('Saved source results changed during trial')
    save(output/'summary.json', dict(conditions=len(rows), aggregates=aggregates, rows=rows,
        frozen_code_sha256=code_hashes, source_sha256=source_hashes,
        formal_results_changed=False, parameters_changed=False,
        real_patients_recomputed=False, event_count=None, event_frequency=None))


if __name__ == '__main__':
    run()

