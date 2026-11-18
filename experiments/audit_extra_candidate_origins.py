"""Trace extra structures in the fixed-noise signal fixtures, not patient truth.

Uses the known injected waveform to distinguish its main phase from filtered
side lobes. A targeted one-node restoration is a causal diagnostic, not a fix.
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
from experiments.audit_wave_evidence import control_arrays
from experiments.test_weak_wave_signals import radial_fixture
from uterine_wave.amplitude import band_limited_signal
from uterine_wave import radial_events
from uterine_wave.io import jsonable


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')


def phase_bounds(signal, peak):
    edges = np.r_[0, np.flatnonzero(np.diff(signal >= 0))+1, len(signal)]
    return next((int(a), int(b), sequence) for sequence, (a, b)
                in enumerate(zip(edges[:-1], edges[1:])) if a <= peak < b)


def disjoint_pair_patterns(candidates):
    """Potential phase ownership conflicts, not confirmed duplicate waves."""
    records = []
    for wall in candidates['walls']:
        paired = [e for e in wall['events'] if e['evidence_kind'] == 'paired_phase']
        for event in wall['events']:
            if event['evidence_kind'] != 'single_phase':
                continue
            ridge = event['phase_ridge_ids'][0]
            for edge in wall['alternative_phase_pairs']:
                if edge['selected'] or ridge not in edge['phase_ridge_ids']:
                    continue
                opposite = next(k for k in edge['phase_ridge_ids'] if k != ridge)
                for main in paired:
                    if opposite not in main['phase_ridge_ids'] or set(edge['sections'])&set(main['sections']):
                        continue
                    records.append(dict(wall=wall['wall'], single_event=event['event_id'],
                        paired_event=main['event_id'], shared_phase=opposite,
                        same_phase_order=edge['leading_polarity'] == main['leading_polarity'],
                        direction_match=edge['classification'] == main['classification'],
                        complementary_pair_direction=edge['classification'],
                        single_sections=edge['sections'], paired_sections=main['sections']))
    return records


def restore_known_gap(signal, valid, t, s, threshold, column, polarity, truth_peak):
    """Restore one observable peak in a known injected phase touching the edge.

    Knowing which peak belongs to the injected wave is an oracle intervention.
    Nothing here is a general patient rule or permission to fill tracking gaps.
    """
    original = radial_events._lobes
    intervention = {}
    def build(y, supported, times, floor):
        filtered, columns = original(y, supported, times, floor)
        if not supported[:, column].all():
            raise ValueError('This diagnostic must not restore missing tracking data')
        a, b, sequence = phase_bounds(filtered[:, column], truth_peak)
        peak = a+int(np.argmax(abs(filtered[a:b, column])))
        if not (a == 0 or b == len(times)) or peak in (a, b-1):
            raise ValueError('Expected an interior peak in an edge-censored phase')
        if (1 if filtered[peak, column] > 0 else -1) != polarity:
            raise ValueError('Known phase polarity differs')
        if abs(filtered[peak, column]) <= floor[column]:
            raise ValueError('Restored node must satisfy original amplitude threshold')
        if any(n['first'] == a and n['last'] == b-1 for n in columns[column]):
            raise ValueError('Target node already present')
        node = dict(column=column, run_id=0, sequence=sequence, sign=polarity,
                    peak_index=peak, peak_time_s=float(times[peak]),
                    amplitude=float(abs(filtered[peak, column])), first=a, last=b-1,
                    start_s=float(times[a]), end_s=float(times[b-1]))
        columns[column].append(node)
        columns[column].sort(key=lambda n:n['first'])
        intervention.update(column=column, phase_start_s=float(times[a]),
            phase_end_s=float(times[b-1]), peak_s=float(times[peak]),
            censored_start=a == 0, censored_end=b == len(times),
            censored_flag_retained_in_audit=True)
        return filtered, columns
    with patch.object(radial_events, '_lobes', build):
        result = radial_events.detect_radial_events(signal, valid, t, s, noise_floor=threshold)
    return result, intervention


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    conditions = read(args.signal_root/'all_conditions.json')
    results, interventions, sources = [], [], {}
    cache = {}
    for condition in conditions:
        saved_path = args.signal_root/(condition['source']+'.json')
        saved = read(saved_path)
        sources[str(saved_path)] = hashlib.sha256(saved_path.read_bytes()).hexdigest()
        primary = {m['event_id'] for m in condition['matches_to_injected_structure'] if m}
        extras = [(w, e) for w in saved['walls'] for e in w['events']
                  if e['classification'] in ('C2F', 'F2C') and e['event_id'] not in primary]
        if not extras:
            continue
        texture = condition['texture']
        path = args.controls_root/(texture+'_rigid.npz')
        if texture not in cache:
            cache[texture] = control_arrays(path)
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        noise, valid, t, s, dt = cache[texture]
        clean, _, _ = radial_fixture(t, s, dt, condition['condition'],
                                     condition['injected_relative_length_peak'])
        clean_filtered = band_limited_signal(clean, t)
        for wall, event in extras:
            w = wall['wall']; noisy = clean+noise[:, w]
            noise_filtered = band_limited_signal(noise[:, w], t)
            total = band_limited_signal(noisy, t)
            np.testing.assert_allclose(total, clean_filtered+noise_filtered, atol=1e-14, rtol=0)
            ridge = next(r for r in wall['phase_ridges'] if r['ridge_id'] == event['phase_ridge_ids'][0])
            polarity = ridge['sign']
            components, main_phase_votes = [], []
            for col, time in zip(event['sections'], event['arrival_time_s']):
                peak = int(np.argmin(abs(t-time)))
                a, b, _ = phase_bounds(clean_filtered[:, col], peak)
                truth_peak = int(np.argmax(polarity*clean[:, col]))
                main_phase_votes.append(a <= truth_peak < b)
                raw_peak = float(np.max(abs(clean[:, col])))
                components.append(dict(section=col, peak_s=time,
                    raw_injection_at_peak=float(clean[peak, col]),
                    raw_relative_to_injected_peak=float(abs(clean[peak, col])/raw_peak),
                    clean_band_at_peak=float(clean_filtered[peak, col]),
                    residual_band_at_peak=float(noise_filtered[peak, col]),
                    total_band_at_peak=float(total[peak, col]),
                    contains_true_signed_peak=a <= truth_peak < b))
            is_main_fragment = all(main_phase_votes)
            if any(main_phase_votes) and not is_main_fragment:
                raise ValueError('Mixed phase origin needs explicit review')
            origin = 'main_injected_phase_fragment' if is_main_fragment else 'filtered_extra_lobe_plus_residual'
            record = dict(source=condition['source'], wall=w, event_id=event['event_id'],
                origin_in_known_fixture=origin, sections=event['sections'],
                original_phase_peak=event['peak_absolute_rate_per_phase'],
                threshold=condition['fixed_noise_floor_rate'], components=components,
                raw_injection_relative_peak_median=float(np.median([x['raw_relative_to_injected_peak'] for x in components])))
            results.append(record)
            if is_main_fragment:
                # Check whether a common opposite ridge was allocated to the
                # larger disjoint part of this same known injected phase.
                options = [edge for edge in wall['alternative_phase_pairs']
                           if event['phase_ridge_ids'][0] in edge['phase_ridge_ids']]
                if len(options) != 1:
                    raise ValueError('Expected one blocked complementary phase pair')
                edge = options[0]
                opposite = next(k for k in edge['phase_ridge_ids'] if k != ridge['ridge_id'])
                assigned = next(e for e in wall['events'] if opposite in e['phase_ridge_ids']
                                and e['evidence_kind'] == 'paired_phase')
                other_id = next(k for k in assigned['phase_ridge_ids'] if k != opposite)
                other = next(r for r in wall['phase_ridges'] if r['ridge_id'] == other_id)
                union = sorted(set(event['sections']) | set(other['sections']))
                gap = sorted(set(range(min(union), max(union)+1))-set(union))
                if len(gap) != 1:
                    raise ValueError('Expected exactly one censored spatial phase node')
                col = gap[0]
                truth_peak = int(np.argmax(polarity*clean[:, col]))
                after, detail = restore_known_gap(noisy, valid[:, w], t, s,
                    condition['fixed_noise_floor_rate'], col, polarity, truth_peak)
                # Column-wise and batched FFTs can differ at ~1e-18 rounding.
                np.testing.assert_allclose(after.pop('filtered_A'), total, atol=1e-14, rtol=0)
                before_count = sum(e['classification'] in ('C2F', 'F2C') for e in wall['events'])
                intervention = dict(source=condition['source'], wall=w,
                    common_opposite_ridge=opposite, unmatched_pair_sections=edge['sections'],
                    assigned_pair_sections=assigned['sections'],
                    disjoint_spatial_support=not bool(set(edge['sections'])&set(assigned['sections'])),
                    before_direction_candidates=before_count,
                    after_restoring_one_known_node=after['traveling_candidate_count'],
                    actual_waveform_changed=False, oracle_diagnostic_not_a_general_fix=True,
                    restored_node=detail)
                interventions.append(intervention)
                record['pairing_mechanism'] = intervention
            print(json.dumps(dict(source=record['source'], wall=w, event=record['event_id'],
                                  origin=origin, raw_ratio=record['raw_injection_relative_peak_median'])), flush=True)
    if len(results) != 8:
        raise ValueError('Expected the same eight additional structures')
    summary = dict(extra_structures=len(results),
        filtered_extra_lobe_plus_residual=sum(r['origin_in_known_fixture'] == 'filtered_extra_lobe_plus_residual' for r in results),
        main_injected_phase_fragments=sum(r['origin_in_known_fixture'] == 'main_injected_phase_fragment' for r in results),
        targeted_node_restorations=interventions, sources_unchanged=True,
        A0_changed=False, patient_results_changed=False, diagnosis_uses_known_injection=True)
    if args.review_root is not None:
        patterns = []
        cases = read(args.review_root/'protocol.json')['selected']
        for case in cases:
            path = args.review_root/case['review_id']/'candidates.json'
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            patterns.extend(dict(source=case['review_id'], **row) for row in disjoint_pair_patterns(read(path)))
        narrower = [r for r in patterns if r['same_phase_order'] and r['direction_match']
                    and r['complementary_pair_direction'] in ('C2F', 'F2C')]
        summary['real_pattern_audit'] = dict(cases=len(cases), potential_relations=len(patterns),
            same_phase_order_and_direction_relations=len(narrower),
            cases_with_same_order_direction_relation=len({r['source'] for r in narrower}),
            confirmed_duplicate_wave_count=None)
        save(args.output/'real_disjoint_pair_patterns.json', patterns)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha for path, sha in sources.items()):
        raise RuntimeError('Original source changed')
    save(args.output/'extra_origins.json', results)
    save(args.output/'summary.json', summary)
    save(args.output/'protocol.json', dict(source_files=sources,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        core_code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'uterine_wave/radial_events.py', 'uterine_wave/amplitude.py',
            'experiments/test_weak_wave_signals.py')},
        scope='Known A-signal-layer fixtures only; not a patient rejection rule'))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--signal-root', type=Path, required=True)
    parser.add_argument('--controls-root', type=Path, required=True)
    parser.add_argument('--review-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
