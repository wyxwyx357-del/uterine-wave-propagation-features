"""CLI for V0 event detection and patient-level candidate features."""
import argparse
import json
from pathlib import Path
import numpy as np
from .io import jsonable
from .event_detection import EventDetectionConfig, detect_events
from .event_features import summarize_events
from .coordination import match_bilateral_events
from .patient_features import aggregate_patient_features


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', required=True)
    parser.add_argument('--route', choices=('A', 'B'), default='A')
    parser.add_argument('--z-threshold', type=float, default=3.0)
    parser.add_argument('--min-component-cells', type=int, default=6)
    parser.add_argument('--min-space-points', type=int, default=4)
    parser.add_argument('--min-path-fraction', type=float, default=.15)
    parser.add_argument('--min-fit-r2', type=float, default=.50)
    parser.add_argument('--synchronous-tolerance-s', type=float, default=.50)
    parser.add_argument('--max-midpoint-lag-s', type=float, default=2.0)
    parser.add_argument('--min-spatial-support-fraction', type=float, default=.50)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)

    with np.load(args.maps, allow_pickle=False) as data:
        maps = dict(data)
    route = args.route
    signal = np.asarray(maps[route], float)
    valid = np.asarray(maps[route+'_valid'], bool)
    time_s = np.asarray(maps['time_s'], float)
    arcs = np.asarray(maps[route+'_arc_positions'], float)
    if signal.ndim != 3 or valid.shape != signal.shape:
        raise ValueError(f'{route} map must be time x plane x space with matching validity')
    if arcs.shape != signal.shape[1:]:
        raise ValueError(f'{route} arc positions must be plane x space')

    config = EventDetectionConfig(
        z_threshold=args.z_threshold,
        min_component_cells=args.min_component_cells,
        min_space_points=args.min_space_points,
        min_path_fraction=args.min_path_fraction,
        min_fit_r2=args.min_fit_r2,
        synchronous_tolerance_s=args.synchronous_tolerance_s,
    ).validate()

    planes = []
    for i in range(signal.shape[1]):
        detected = detect_events(signal[:, i], valid[:, i], time_s, arcs[i], config)
        planes.append(dict(
            plane_index=i,
            event_count=detected['event_count'],
            events=detected['events'],
            summary=summarize_events(detected['events']),
        ))

    report = dict(
        status='V0_candidate_events_not_clinically_validated',
        route=route,
        config=detected['config'] if planes else None,
        planes=planes,
    )

    if route == 'A':
        if signal.shape[1] != 2:
            raise ValueError('A route must contain exactly anterior and posterior wall planes')
        coordination = match_bilateral_events(
            planes[0]['events'], planes[1]['events'],
            max_midpoint_lag_s=args.max_midpoint_lag_s,
        )
        report['coordination'] = coordination
        report['patient_features'] = aggregate_patient_features(
            planes[0]['events'], planes[1]['events'],
            valid[:, 0], valid[:, 1], time_s, coordination,
            min_spatial_fraction=args.min_spatial_support_fraction,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(jsonable(report), indent=2, allow_nan=False),
        encoding='utf-8'
    )


if __name__ == '__main__':
    main()
