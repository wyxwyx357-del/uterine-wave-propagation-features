"""Strict label-free record-MSE boundary for future measurement, not historical blinding."""
import argparse
import json
import math
from numbers import Real
import os
from pathlib import Path

from .record_mse_core import summarize_paired_windows

EXPECTED_METADATA = {
    'definition_version': 'radial_wave_features_v2',
    'requested_s_range': [.2, .8],
    'nominal_interval_s': 2.,
    'nominal_window_s': 20.,
    'nominal_step_s': 1.,
    'band_cpm': [.5, 5.],
}
OPTIONAL_METADATA = {'actual_interval_s_range', 'actual_window_sample_span_s', 'actual_step_s'}
REQUIRED_WINDOW = {'window_start_s', 'window_end_s', 'window_center_s', 'first_section',
                   'last_section', 'anterior_ER', 'posterior_ER'}
OPTIONAL_WINDOW = {'s_start', 's_end', 'center_span'}


def _keys(value, required, optional, label):
    if not isinstance(value, dict):
        raise ValueError(label + ' must be an object')
    if set(value) - required - optional or required - set(value):
        raise ValueError(label + ' has missing or forbidden fields; clinical/label fields are not accepted')


def _numeric(value):
    # Protocol identity, not a physiological support/quality threshold.
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError('Measurement fields must use finite numeric values')


def validate_metadata(metadata):
    """Refuse an unverified/different protocol instead of silently mixing MSE definitions."""
    _keys(metadata, set(EXPECTED_METADATA), OPTIONAL_METADATA, 'measurement_metadata')
    for key, expected in EXPECTED_METADATA.items():
        value = metadata[key]
        if isinstance(expected, list):
            if not isinstance(value, (list, tuple)) or len(value) != len(expected):
                raise ValueError('Protocol mismatch: ' + key)
            for item in value:
                _numeric(item)
            equal = list(value) == expected
        elif isinstance(expected, str):
            equal = isinstance(value, str) and value == expected
        else:
            _numeric(value)
            equal = value == expected
        if not equal:
            raise ValueError('Protocol mismatch: ' + key)
    if 'actual_interval_s_range' in metadata:
        values = metadata['actual_interval_s_range']
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError('actual_interval_s_range must have two values')
        for item in values:
            _numeric(item)
    for key in ['actual_window_sample_span_s', 'actual_step_s']:
        if key in metadata:
            _numeric(metadata[key])
    return dict(metadata)


def measure_record(payload):
    """Only case_id, paired ER windows and a verified measurement protocol enter here."""
    _keys(payload, {'case_id', 'paired_ER_windows', 'measurement_metadata'}, set(), 'measurement input')
    if not isinstance(payload['case_id'], str) or not payload['case_id'].strip():
        raise ValueError('case_id must be a nonempty local record key')
    metadata = validate_metadata(payload['measurement_metadata'])
    paired = payload['paired_ER_windows']
    if not isinstance(paired, list):
        raise ValueError('paired_ER_windows must be a list')
    for window in paired:
        _keys(window, REQUIRED_WINDOW, OPTIONAL_WINDOW, 'paired ER window')
        for value in window.values():
            _numeric(value)
    # Verbatim historical numeric core; no labels, imputation or association calls.
    record = summarize_paired_windows(paired)
    return dict(case_id=payload['case_id'], measurement_metadata=metadata, record=record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=os.environ.get('INPUT_PATH'), required='INPUT_PATH' not in os.environ)
    parser.add_argument('--output', default=os.environ.get('OUTPUT_PATH'), required='OUTPUT_PATH' not in os.environ)
    args = parser.parse_args()
    result = measure_record(json.loads(Path(args.input).read_text(encoding='utf-8-sig')))
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError('Use a new private measurement output; never overwrite frozen results')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                           encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
