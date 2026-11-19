"""Replay cached maps with old/new exposure arithmetic; never rerun tracking."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uterine_wave import patient
from uterine_wave.io import jsonable


def legacy_center_duration(mask, interval_start_s, interval_end_s):
    center = (interval_start_s+interval_end_s)/2
    return float(np.diff(center)[mask[:-1] & mask[1:]].sum())


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in args.maps:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            maps = dict(data)
        # Replay the former arithmetic on exactly the same maps, masks, and
        # feature implementation. Only the private duration helper is swapped.
        with patch.object(patient, '_duration', legacy_center_duration):
            before = patient.summarize_patient(maps, s_range=args.s_range)
        after = patient.summarize_patient(maps, s_range=args.s_range)
        for old, new in zip(before['walls'], after['walls']):
            assert {k: v for k, v in old.items() if k != 'common_region_duration_s'} == {
                k: v for k, v in new.items() if k != 'common_region_duration_s'}
        assert before['modes'] == after['modes']
        assert before['events'] == after['events']
        assert before['event_summary'] == after['event_summary']
        assert after['event_summary']['event_count'] is None
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        row = dict(source=path.name, source_sha256=digest,
            old_common_region_duration_s=[w['common_region_duration_s'] for w in before['walls']],
            new_common_region_duration_s=[w['common_region_duration_s'] for w in after['walls']],
            unchanged_amplitude=[w['radial_stretch_activity_amplitude'] for w in after['walls']],
            all_other_wall_and_mode_values_equal=True, reviewed_events_supplied=False,
            event_count=None, event_frequency_cpm=None)
        rows.append(row)
        (args.output/(path.stem+'_features.json')).write_text(json.dumps(
            jsonable(after), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (args.output/'summary.json').write_text(json.dumps(jsonable(dict(
        rows=rows, s_range=args.s_range, tracking_rerun=False,
        baseline='old center-span helper replayed on same cached maps',
        exposure_definition='union_of_supported_measurement_intervals')),
        ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(jsonable(rows), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', type=Path, nargs='+', required=True)
    parser.add_argument('--s-range', type=float, nargs=2, default=(.2, .8))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())

