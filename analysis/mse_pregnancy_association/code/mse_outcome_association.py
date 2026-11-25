"""Phase2 private join only; labels enter after measurement files have been frozen.

This prepares inputs for a separately specified association/prediction analysis.
It does not fit a model, calculate AUC/OR or choose clinical variables.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from .record_mse_measurement import validate_metadata


CLINICAL9 = ['age', 'bmi', 'infertility_years', 'transfer_embryo_count_component',
             'blastocyst_transfer', 'endometrial_thickness_mm', 'endometrial_volume_ml',
             'endometrial_pattern_pure_C', 'days_before_transfer']


def join_measurements(measurements, clinical):
    """Left-join exact case keys; preserve missing measurements and unmatched records."""
    rows = []
    for measurement in measurements:
        if set(measurement) != {'case_id', 'measurement_metadata', 'record'}:
            raise ValueError('Use the strict label-free measurement output, not legacy mixed cohort files')
        validate_metadata(measurement['measurement_metadata'])
        record = measurement['record']
        rows.append(dict(case_id=measurement['case_id'],
            record_MSE_time_equal=record['record_MSE_time_equal'],
            record_CC_time_equal=record['record_CC_time_equal']))
    frame = pd.DataFrame(rows, columns=['case_id', 'record_MSE_time_equal', 'record_CC_time_equal'])
    required = {'case_id', 'patient_id', 'outcome'}
    if not required.issubset(clinical.columns):
        raise ValueError('Clinical phase requires case_id, patient_id and frozen outcome')
    if any(not isinstance(key, str) or not key.strip() for key in [*frame.case_id, *clinical.case_id]):
        raise ValueError('Use nonempty exact case keys; missing identities are not matched')
    if frame.case_id.duplicated().any() or clinical.case_id.duplicated().any():
        raise ValueError('case_id must identify one examination per input table')
    if not clinical.outcome.dropna().isin([0, 1]).all():
        raise ValueError('Use the frozen source label policy; this join does not recode outcomes')
    columns = ['case_id', 'patient_id', 'outcome'] + [c for c in CLINICAL9 if c in clinical]
    return frame.merge(clinical[columns], on='case_id', how='left', validate='one_to_one', indicator='clinical_join')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements', nargs='+', required=True, help='Previously frozen strict measurement JSON files')
    parser.add_argument('--clinical', required=True, help='Private Clinical9/outcome table; exact case_id joins only')
    parser.add_argument('--output', required=True, help='New private joined CSV, not a public result')
    args = parser.parse_args()
    measurements = [json.loads(Path(p).read_text(encoding='utf-8-sig')) for p in args.measurements]
    clinical = pd.read_csv(args.clinical, dtype={'case_id': str, 'patient_id': str})
    joined = join_measurements(measurements, clinical)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError('Use a new private phase2 output')
    destination.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(destination, index=False)


if __name__ == '__main__':
    main()
