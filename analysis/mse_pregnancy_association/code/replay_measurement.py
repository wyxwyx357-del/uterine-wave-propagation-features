"""Portable saved-tracks → frozen A/ER/record-MSE entry; private inputs required."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .measurement.deformation import compute_spacetime
from .measurement.patient import summarize_patient
from .measurement.io import jsonable
from .record_mse_measurement import measure_record, EXPECTED_METADATA


def replay(tracks,grid,*,case_id):
    maps=compute_spacetime(tracks,grid,interval_s=2.)
    patient=summarize_patient(maps,s_range=(.2,.8),include_mode_velocity=False)
    modes=patient['modes']
    metadata=dict(definition_version=patient['definition_version'],
        requested_s_range=patient['requested_s_range'],nominal_interval_s=patient['nominal_interval_s'],
        nominal_window_s=modes['nominal_window_s'],nominal_step_s=modes['nominal_step_s'],
        band_cpm=list(EXPECTED_METADATA['band_cpm']),actual_interval_s_range=patient['actual_interval_s_range'],
        actual_window_sample_span_s=modes['actual_window_sample_span_s'],actual_step_s=modes['actual_step_s'])
    for window in modes['windows']:
        if window['band_cpm']!=metadata['band_cpm']:
            raise ValueError('ER window band differs from the frozen measurement protocol')
    measurement=measure_record(dict(case_id=case_id,paired_ER_windows=modes['paired_ER_windows'],measurement_metadata=metadata))
    return dict(modes=modes,record=measurement['record'],measurement=measurement,nominal_interval_s=2.,
                actual_interval_s_range=patient['actual_interval_s_range'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-id',required=True,help='Local examination key; no outcome or clinical fields')
    parser.add_argument('--tracks',default=os.environ.get('INPUT_PATH'),required='INPUT_PATH' not in os.environ)
    parser.add_argument('--grid',required=True,help='Original saved grid/layout JSON or numeric NPZ; no re-annotation')
    parser.add_argument('--output',default=os.environ.get('OUTPUT_PATH'),required='OUTPUT_PATH' not in os.environ)
    args=parser.parse_args()
    with np.load(args.tracks,allow_pickle=False) as z:tracks={key:z[key] for key in z.files}
    if Path(args.grid).suffix.lower()=='.json':
        grid=json.loads(Path(args.grid).read_text(encoding='utf-8-sig'))
    else:
        with np.load(args.grid,allow_pickle=False) as z:grid={key:z[key] for key in z.files}
    result=replay(tracks,grid,case_id=args.case_id)
    destination=Path(args.output)
    if destination.exists():raise FileExistsError('Use a new private output path; do not overwrite frozen results.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(jsonable(result['measurement']),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
