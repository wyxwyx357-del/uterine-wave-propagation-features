"""Read existing frozen A0 caches; export only the requested examination table/summary.

No tracking, image, clinical-table or outcome imports. M1 is the unchanged frozen
context applied to the existing production count_map, including its local split.
"""
import argparse
import csv
import hashlib
import json
import zipfile
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'uterine-wave-propagation-features' / 'outputs'
OUTPUT = ROOT / 'outputs/m1_cohort_frozen_v1'
BENCH = ROOT / 'experiments/step2_salience_sandbox/benchmark_run'
VALIDATION = BENCH / 'm1_unseen_texture_validation_v2'
VERSION = 'm1-global-pairing-unseen-v1'
MAP_KEYS = ('A', 'valid_mask', 's', 'interval_start_s', 'interval_end_s',
            'interval_center_s', 'interval_dt_s')
CACHE_ROOTS = (
    DATA / 'frozen_features_223_20261119',
    DATA / 'short_duration_holdout_validation_20261120/measurement',
    DATA / 'existing_data_completion_20261124/dynamic_append/measurement',
    DATA / 'screened_feature_auc_20261124/dynamic_append/measurement',
)
ID_CSVS = (DATA / 'frozen_feature_pipeline_20261119/input_inventory.csv',
           CACHE_ROOTS[0] / 'manifest.csv', CACHE_ROOTS[1] / 'manifest.csv')
ID_JSONS = tuple(p.parent / 'READY_INPUTS_LOCAL.json' for p in CACHE_ROOTS[2:])
NUMERIC_FIELDS = (
    'm1_prop_count_min', 'm1_prop_count_max', 'effective_observation_seconds',
    'effective_observation_minutes', 'm1_prop_rate_min_per_min',
    'm1_prop_rate_max_per_min', 'm1_other_significant_activity_count',
    'ambiguous_cross_wall_prop_pairs', 'raw_valid_fraction',
    'm1_minmax_physical_cutoff',
)
FIELDS = ('case_id', 'm1_status', 'failure_reason', *NUMERIC_FIELDS,
          'algorithm_version', 'reviewed')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def freeze_check():
    freeze_path = VALIDATION / 'M1_CANDIDATE_FREEZE.json'
    candidate = load_json(freeze_path)
    assert candidate['candidate_version_id'] == VERSION
    for name, digest in candidate['source_hashes'].items():
        assert sha(ROOT / name) == digest, 'Frozen source mismatch: ' + name
    assert sha(VALIDATION / candidate['code_snapshot']) == candidate['code_snapshot_sha256']
    prelock_path = VALIDATION / 'M1_UNSEEN_TEXTURE_PRELOCK.json'
    lock = load_json(prelock_path)
    assert sha(prelock_path) == candidate['prelock_sha256']
    assert lock['formal_validation_eligible'] is True
    assert load_json(VALIDATION / 'M1_UNSEEN_TEXTURE_GATE_DECISION.json')['M1_UNSEEN_VALIDATION'] == 'PASS'
    manifest = load_json(VALIDATION / 'M1_UNSEEN_TEXTURE_INPUT_MANIFEST.json')
    assert len(manifest['sources']) == 8
    attempt_path = BENCH / 'pairing_filter_2x2/attempt_002/PAIRING_FILTER_2X2_PRELOCK.json'
    attempt = load_json(attempt_path)
    import scipy
    assert scipy.__version__ == attempt['scipy_version'], 'Frozen SciPy version differs'
    mechanism = ROOT / 'experiments/step2_salience_sandbox/mechanism_2x2/mechanisms.py'
    assert sha(mechanism) == candidate['mechanisms_sha256'] == lock['M1_mechanism_file_hash']
    assert sha(mechanism) == attempt['experimental_source_sha256']['mechanisms.py']
    from experiments.step2_salience_sandbox.benchmark_data import verify
    baseline = verify()
    source_hashes = dict(candidate['source_hashes'])
    for name, digest in baseline['benchmark_source_sha256'].items():
        source_hashes['experiments/step2_salience_sandbox/' + name] = digest
    source_hashes['experiments/m1_cohort_frozen.py'] = sha(Path(__file__))
    config = dict(algorithm_version=VERSION, band_cpm=[.5, 5.], cutoff=.6,
        runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
        normalization='whole_video_both_walls', min_continuous_run_s=12,
        minimum_adjacent_native_sections=3, noise_floor=float(lock['noise_floor']),
        main_analysis_scalar='m1_prop_rate_min_per_min',
        M1_scope='unchanged global detector only; local candidate_split_verifier bypasses M1 weighting',
        count='unchanged Step1 v2 identity ambiguity min/max; unchanged filtered observation union',
        cache_roots=[str(p.relative_to(DATA)) for p in CACHE_ROOTS],
        inventory='Union of technical case-ID registries and canonical per-examination A0 caches; no patient deduplication or outcome selection.',
        duplicate_policy='Static root order, only after equal detector-input arrays are verified; no output-driven selection.',
        registry_projection='case_id only; no clinical/outcome tables',
        missing_policy='Keep failure rows; no tracking, cache reconstruction or imputation',
        extra_feature_definitions=dict(ambiguous_cross_wall_prop_pairs='existing ambiguous_cross_wall_propagation_pairs maximum matching size',
            m1_other_significant_activity_count='existing other_significant_activity_count',
            raw_valid_fraction='mean(valid_mask & isfinite(A))',
            m1_minmax_physical_cutoff='existing normalization.physical_cutoff'))
    hashes = dict(source_files=source_hashes, fixed_configuration=value_hash(config),
        candidate_freeze=sha(freeze_path), unseen_validation_prelock=sha(prelock_path),
        attempt_002_prelock=sha(attempt_path), candidate_code_snapshot=candidate['code_snapshot_sha256'])
    return config, hashes


def read_maps(path):
    with np.load(path, allow_pickle=False) as z:
        maps = {k: z[k] for k in MAP_KEYS}
        if 'nominal_interval_s' in z.files and float(z['nominal_interval_s']) != 2.:
            raise ValueError('Existing cache nominal interval is not the frozen 2 s definition')
    a, valid, s, t = (maps[k] for k in ('A', 'valid_mask', 's', 'interval_center_s'))
    if t.ndim != 1 or s.ndim != 1 or len(t) < 2 or a.shape != (len(t), 2, len(s)) or valid.shape != a.shape:
        raise ValueError('Invalid native time/wall/section array dimensions')
    if not (np.isfinite(t).all() and (np.diff(t) > 0).all() and np.isfinite(s).all()):
        raise ValueError('Invalid native time or section coordinates')
    for k in ('interval_start_s', 'interval_end_s', 'interval_dt_s'):
        if maps[k].shape != t.shape or not np.isfinite(maps[k]).all():
            raise ValueError('Invalid measurement interval vector: ' + k)
    if not ((maps['interval_dt_s'] > 0).all() and
            (maps['interval_end_s'] > maps['interval_start_s']).all()):
        raise ValueError('Invalid measurement interval duration')
    return maps


def error_text(prefix, error):
    return prefix + ': ' + type(error).__name__ + ': ' + str(error)


def inventory():
    ids = set()
    registry_hashes = {}
    for path in ID_CSVS:
        with path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            col = next(reader).index('case_id')
            projected = [row[col] for row in reader]
        ids.update(projected)
        registry_hashes[str(path.relative_to(DATA))] = value_hash(projected)
    for path in ID_JSONS:
        projected = [r['case_id'] for r in load_json(path)]
        ids.update(projected)
        registry_hashes[str(path.relative_to(DATA))] = value_hash(projected)
    candidates = defaultdict(list)
    for root in CACHE_ROOTS:
        for p in sorted(root.glob('*/spacetime_2s.npz')):
            ids.add(p.parent.name)
            candidates[p.parent.name].append(p)
    entries = []
    for case in sorted(ids):
        paths = candidates[case]
        entry = dict(case_id=case, paths=[str(p) for p in paths],
                     input_sha256=[], preflight_failure='')
        if not paths:
            entry['preflight_failure'] = 'missing_A0_input'
        else:
            try:
                maps = read_maps(paths[0])
                entry['input_sha256'].append(sha(paths[0]))
                for p in paths[1:]:
                    other = read_maps(p)
                    if any(not np.array_equal(maps[k], other[k], equal_nan=True) for k in MAP_KEYS):
                        raise ValueError('Conflicting duplicate frozen A0 detector inputs')
                    entry['input_sha256'].append(sha(p))
            except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as error:
                entry['preflight_failure'] = error_text('unreadable_or_invalid_A0_input', error)
        entries.append(entry)
    assert len(entries) == len({e['case_id'] for e in entries}) and len(entries) > 0
    return entries, registry_hashes


def empty_row(case, status, reason):
    return dict(case_id=case, m1_status=status, failure_reason=reason,
                **{f: None for f in NUMERIC_FIELDS}, algorithm_version=VERSION, reviewed=False)


def result_row(case, result, maps):
    row = empty_row(case, 'completed', '')
    seconds = float(result['effective_observation_seconds'])
    row.update(effective_observation_seconds=seconds, effective_observation_minutes=seconds / 60,
               raw_valid_fraction=float((np.asarray(maps['valid_mask'], bool) & np.isfinite(maps['A'])).mean()))
    if seconds <= 0 or result['status'] == 'no_analyzable_event_support':
        row.update(m1_status='no_effective_support', failure_reason='no_effective_observation_support')
        return row
    row.update(m1_prop_count_min=int(result['significant_propagation_activity_count_min']),
               m1_prop_count_max=int(result['significant_propagation_activity_count_max']),
               m1_other_significant_activity_count=int(result['other_significant_activity_count']),
               ambiguous_cross_wall_prop_pairs=int(result['ambiguous_cross_wall_propagation_pairs']),
               m1_minmax_physical_cutoff=result['normalization']['physical_cutoff'])
    row['m1_prop_rate_min_per_min'] = row['m1_prop_count_min'] / row['effective_observation_minutes']
    row['m1_prop_rate_max_per_min'] = row['m1_prop_count_max'] / row['effective_observation_minutes']
    assert 0 <= row['m1_prop_count_min'] <= row['m1_prop_count_max']
    return row


def process_case(entry, noise_floor):
    case = entry['case_id']
    if entry['preflight_failure']:
        status = 'missing_A0_input' if not entry['paths'] else 'invalid_A0_input'
        return empty_row(case, status, entry['preflight_failure'])
    from analysis.mse_pregnancy_association.code.batch_wave_counts import count_map
    from experiments.step2_salience_sandbox.mechanism_2x2.mechanisms import intervention
    try:
        path = Path(entry['paths'][0])
        if sha(path) != entry['input_sha256'][0]:
            raise RuntimeError('Frozen A0 input changed after preflight')
        maps = read_maps(path)
        with intervention('M1'):
            result = count_map(maps, noise_floor)
        if sha(path) != entry['input_sha256'][0]:
            raise RuntimeError('Frozen A0 input changed during calculation')
        return result_row(case, result, maps)
    except Exception as error:
        # One examination's failure remains explicit; never substitute a model,
        # regenerate a cache, or turn missing evidence into a zero count/rate.
        return empty_row(case, 'failed', error_text('M1_calculation_failed', error))


def run(workers):
    config, hashes = freeze_check()
    entries, registry_hashes = inventory()
    hashes.update(input_inventory=value_hash(entries), technical_case_id_registries=registry_hashes)
    print(json.dumps(dict(preflight_cases=len(entries), existing_A0_cases=sum(bool(e['paths']) for e in entries),
        missing_A0_cases=sum(not e['paths'] for e in entries), invalid_A0_cases=sum(bool(e['paths']) and bool(e['preflight_failure']) for e in entries))), flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if any(OUTPUT.iterdir()):
        raise RuntimeError('Output directory is nonempty; do not overwrite a frozen batch')
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_case, e, config['noise_floor']) for e in entries]
        for f in as_completed(futures):
            row = f.result()
            results[row['case_id']] = row
            if len(results) % 25 == 0 or len(results) == len(entries):
                print('M1 completed examinations:', len(results), '/', len(entries), flush=True)
    _, after_hashes = freeze_check()
    assert hashes['source_files'] == after_hashes['source_files'], 'Frozen sources changed during the batch'
    assert hashes['fixed_configuration'] == after_hashes['fixed_configuration']
    assert all(sha(Path(p)) == h for e in entries for p, h in zip(e['paths'], e['input_sha256']))
    table_path = OUTPUT / 'm1_cohort_features_frozen.csv'
    ordered = [results[e['case_id']] for e in entries]
    with table_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in ordered:
            writer.writerow({k: 'NaN' if row[k] is None else row[k] for k in FIELDS})
    rates = np.array([r['m1_prop_rate_min_per_min'] for r in ordered
                      if r['m1_prop_rate_min_per_min'] is not None], dtype=float)
    assert np.isfinite(rates).all()
    success = sum(r['m1_status'] == 'completed' for r in ordered)
    no_support = sum(r['m1_status'] == 'no_effective_support' for r in ordered)
    summary = dict(input_case_count=len(ordered), success_count=success,
        no_effective_support_count=no_support, other_failure_count=len(ordered) - success - no_support,
        failure_reason_counts=dict(Counter(r['failure_reason'] for r in ordered if r['failure_reason'])),
        m1_prop_rate_min_per_min=dict(valid_n=len(rates), median=float(np.median(rates)) if len(rates) else None,
            Q1=float(np.quantile(rates, .25)) if len(rates) else None,
            Q3=float(np.quantile(rates, .75)) if len(rates) else None,
            minimum=float(np.min(rates)) if len(rates) else None,
            maximum=float(np.max(rates)) if len(rates) else None),
        algorithm_version=VERSION, source_and_configuration_sha256=hashes,
        main_table_sha256=sha(table_path), pregnancy_outcome_read=False,
        tracking_rerun=False, production_core_modified=False)
    (OUTPUT / 'm1_cohort_frozen_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'source_and_configuration_sha256'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    run(args.workers)
