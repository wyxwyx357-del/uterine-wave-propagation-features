"""Apply the existing significant-activity counter to saved A0 maps.

No outcome access, tracking rerun, new detector, or threshold selection. These
remain algorithm-defined significant propagation counts, not adjudicated waves.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from experiments.activity_counting import count_activities
from experiments.activity_strength import filtered_runs
from experiments.candidate_split_verifier import prepare_parent, decide_split
from experiments.global_event_ownership import reconcile
from experiments.normalized_activity import normalize_magnitude, score_salience
from experiments.partial_phase_events import detect_partial
from experiments.patient_activity_aggregation import candidate_nodes, observation_union
from uterine_wave.io import jsonable

ROOT = Path(__file__).resolve().parents[1]
COUNT = 'significant_propagation_activity_count'
RATE = 'significant_propagation_activity_frequency_cpm'
CODE = ['experiments/batch_wave_counts.py', 'experiments/activity_counting.py',
        'experiments/activity_strength.py', 'experiments/candidate_split_verifier.py',
        'experiments/event_input_smoothing.py', 'experiments/global_event_ownership.py',
        'experiments/normalized_activity.py', 'experiments/partial_phase_events.py',
        'experiments/patient_activity_aggregation.py', 'experiments/direction_identity_trial.py',
        'uterine_wave/radial_events.py', 'uterine_wave/amplitude.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    text = json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def count_map(maps, noise_floor):
    a = np.asarray(maps['A'], float)
    valid = np.asarray(maps['valid_mask'], bool) & np.isfinite(a)
    t = np.asarray(maps['interval_center_s'], float)
    s = np.asarray(maps['s'], float)
    if a.shape != (len(t), 2, len(s)) or valid.shape != a.shape:
        raise ValueError('Expected native time x two walls x section maps')
    span = float(np.median(maps['interval_dt_s']))
    dt = float(np.median(np.diff(t)))
    filtered = filtered_runs(a, valid, t)
    filtered_valid = np.isfinite(filtered)
    observable = (filtered_valid[:, :, :-2] & filtered_valid[:, :, 1:-1] &
                  filtered_valid[:, :, 2:]).any() if len(s) >= 3 else False
    exposure = observation_union(dict(maps, valid_mask=valid))
    if not observable or exposure['seconds'] <= 0:
        return dict(status='no_analyzable_event_support', **{COUNT: None, RATE: None},
                    effective_observation_seconds=exposure['seconds'],
                    failure_reason='No three adjacent native sections with finite filtered support',
                    observation=exposure)
    normalized, metadata = normalize_magnitude(filtered, valid)
    ownership = dict(walls=[])
    selected = dict(walls=[])
    evidence = []
    for w in (0, 1):
        report = detect_partial(a[:, w], valid[:, w], t, s, noise_floor=noise_floor)
        report.pop('filtered_A')  # Arrays remain in the source NPZ, not JSON.
        proposals = [decide_split(prepare_parent(a[:, w], valid[:, w], t, s, parent,
                     noise_floor=noise_floor, measurement_span_s=span),
                     minimum_adjacent_support=3) for parent in report['events']]
        result = reconcile(report, proposals, t, valid[:, w], measurement_span_s=span)
        scores = [score_salience(event, result['source_evidence'], normalized[:, w],
                  valid[:, w], t) for event in result['event_proposals']]
        ownership['walls'].append(dict(reconciliation=result))
        selected['walls'].append(dict(wall=w, scores=scores))
        evidence.append(dict(wall=w, parent_report=report, split_proposals=proposals,
                             reconciliation=result, salience=scores))
    nodes = candidate_nodes(selected, ownership)
    result = count_activities(nodes, s, span, dt, exposure['seconds'])
    result.update(status='completed', failure_reason=None, normalization=metadata,
                  observation=exposure, measurement_span_s=span, sample_dt_s=dt,
                  native_section_count=len(s), sampled_s_range=[float(s[0]), float(s[-1])],
                  requested_s_range=[0., 1.], walls=evidence,
                  raw_valid_fraction=float(valid.mean()),
                  unresolved_activity_rows=sum(bool(e['unresolved_related_nodes'])
                                               for e in result['activities']),
                  reviewed=False, frozen_count_algorithm=False)
    return result


def process_case(task):
    case_id, source, destination, floor = task
    begin = time.monotonic()
    try:
        before = digest(source)
        with np.load(source, allow_pickle=False) as z:
            maps = dict(z)
        result = count_map(maps, floor)
        if digest(source) != before:
            raise RuntimeError('Source spacetime changed during counting')
        result.update(case_id=case_id, source_spacetime=str(source),
                      source_sha256=before, elapsed_s=time.monotonic()-begin)
        save(destination, result)
        keys = [COUNT, RATE, 'other_significant_activity_count',
                'effective_observation_seconds', 'unresolved_activity_rows',
                'raw_valid_fraction', 'sampled_s_range', 'native_section_count']
        return dict(case_id=case_id, processing_status=result['status'],
                    failure_reason=result.get('failure_reason'), result_path=str(destination),
                    **{k: result.get(k) for k in keys})
    except Exception as error:
        # Per-case batch boundary only: retain the actual error, never zero.
        failed = dict(case_id=case_id, processing_status='failed',
                      failure_reason=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc(), **{COUNT: None, RATE: None})
        save(destination, failed)
        return {k: v for k, v in failed.items() if k != 'traceback'}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = pd.read_csv(args.features/'patient_results.csv')
    if args.case_id:
        inputs = inputs.loc[inputs['case_id'].isin(args.case_id)]
    if not len(inputs) or inputs['case_id'].duplicated().any():
        raise ValueError('Expected unique, nonempty examination IDs')
    protocol_source = json.loads(args.noise_protocol.read_text(encoding='utf-8'))
    floor = float(protocol_source['noise_floor_rate'])
    protocol = dict(noise_floor_rate=floor, noise_protocol=str(args.noise_protocol),
        noise_protocol_sha256=digest(args.noise_protocol), cutoff=.6, minimum_native_support=3,
        parents='periodic FFT + observable partial phase',
        split='existing measurement_span_mean, support=3, within parents only',
        ownership='existing global ownership + direction-independent cross-wall identity',
        propagation='existing arrival classifier with restored phase fits',
        observation='union of measurement intervals with >=3 adjacent valid native sections in either wall',
        spatial_scope='all existing native sections, no resampling, no best-region selection',
        direction_used_for_counting=False, outcomes_read=False, thresholds_changed=False,
        algorithm_frozen=False, count_interpretation='significant propagation activity, not adjudicated total waves',
        known_limitations=['local_once false positives', 'adjacent-event duplicate identity',
                           'fixed .6 salience misses weak events'],
        code_sha256={p: digest(ROOT/p) for p in CODE}, requested_cases=len(inputs))
    save(args.output/'protocol_before_counting.json', protocol)
    tasks = [(row.case_id, args.features/row.case_id/'spacetime_2s.npz',
              args.output/(row.case_id+'.json'), floor) for row in inputs.itertuples()]
    rows = []
    if args.workers == 1:
        rows = [process_case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_case, task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
                save(args.output/'progress.json', dict(finished=len(rows), total=len(tasks)))
    table = pd.DataFrame(rows).set_index('case_id').reindex(inputs['case_id']).reset_index()
    table.to_csv(args.output/'wave_counts.csv', index=False, encoding='utf-8-sig')
    table.to_excel(args.output/'wave_counts.xlsx', index=False)
    if any(digest(ROOT/p) != h for p, h in protocol['code_sha256'].items()):
        raise RuntimeError('Counting implementation changed during the batch')
    summary = dict(total=len(table), status_counts=table['processing_status'].value_counts().to_dict(),
        total_algorithm_defined_activities=int(table[COUNT].sum()),
        nonzero_exams=int((table[COUNT] > 0).sum()),
        unavailable_exams=int(table[COUNT].isna().sum()),
        frozen_algorithms_changed=False, thresholds_changed=False)
    save(args.output/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--noise-protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--case-id', action='append')
    run(parser.parse_args())
