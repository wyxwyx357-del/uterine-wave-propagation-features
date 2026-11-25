"""Small command interface; measurement input deliberately contains no outcomes."""
import argparse
import json
from pathlib import Path
import numpy as np
from .geometry import CONFIG, read_labelme, layout
from .io import jsonable
from .deformation import DEFAULT_PROPAGATION_INTERVAL_S


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    grid_cmd = commands.add_parser('layout', help='Create A/B seed topology from first-frame LabelMe JSON')
    grid_cmd.add_argument('--annotation', required=True)
    grid_cmd.add_argument('--scale', type=float, default=.5)
    grid_cmd.add_argument('--output', required=True)
    control_cmd = commands.add_parser('controls', help='Known-motion controls; defaults to generated texture')
    control_cmd.add_argument('--image', help='Optional local grayscale first-frame image, already at processed resolution')
    control_cmd.add_argument('--annotation', help='Matching native-resolution first-frame LabelMe JSON')
    control_cmd.add_argument('--scale', type=float, default=.5)
    control_cmd.add_argument('--output', required=True)
    track_cmd = commands.add_parser('track', help='Track a video using explicitly supplied background poses')
    track_cmd.add_argument('--video', required=True)
    track_cmd.add_argument('--annotation', required=True)
    track_cmd.add_argument('--poses', required=True, help='NPZ with time_s, poses, pose_epoch, matching decoded frames')
    track_cmd.add_argument('--time-factor', type=float, required=True)
    track_cmd.add_argument('--scale', type=float, default=.5)
    track_cmd.add_argument('--output', required=True)
    maps_cmd = commands.add_parser('features', help='Compute physical-interval radial maps; explicit --hop keeps legacy A/B output')
    maps_cmd.add_argument('--tracks', required=True)
    maps_cmd.add_argument('--layout', required=True)
    interval_group = maps_cmd.add_mutually_exclusive_group()
    interval_group.add_argument('--interval-s', type=float, default=DEFAULT_PROPAGATION_INTERVAL_S,
                                help='Physical interval in seconds (default: 2.0; sensitivity: 0.96)')
    interval_group.add_argument('--hop', type=int, help='Explicit legacy frame interval and A/B output')
    maps_cmd.add_argument('--output', required=True)
    patient_cmd = commands.add_parser('summarize', help='Fixed-region radial wave features and reviewed event measurements')
    patient_cmd.add_argument('--maps', required=True, help='Physical-interval spacetime NPZ')
    patient_cmd.add_argument('--s-range', type=float, nargs=2, required=True, metavar=('START', 'END'),
                             help='Preselected normalized anatomical range, applied to both walls')
    patient_cmd.add_argument('--review', help='JSON with complete, intervals, events; omitted means unreviewed, not zero waves')
    patient_cmd.add_argument('--time-range', type=float, nargs=2, metavar=('START', 'END'),
                             help='Acquisition-time bounds; only complete measurement intervals inside these bounds are used')
    patient_cmd.add_argument('--refine-mode-speed', action='store_true', help='Enable the existing continuous mode-speed candidate')
    patient_cmd.add_argument('--output', required=True)
    wave_cmd = commands.add_parser('wave-candidates', help='Retain reviewable paired and single-phase propagation evidence on radial maps')
    wave_cmd.add_argument('--maps', required=True)
    wave_cmd.add_argument('--noise-floor', type=float, required=True,
                          help='Absolute A threshold in s^-1 from an explicitly documented noise reference')
    wave_cmd.add_argument('--s-range', type=float, nargs=2, default=(0.,1.))
    wave_cmd.add_argument('--output', required=True, help='JSON report; companion NPZ stores filtered maps')
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == 'layout':
        grid = layout(read_labelme(args.annotation, coordinate_scale=args.scale))
        np.savez_compressed(output, **grid)
    elif args.command == 'controls':
        import cv2
        from .controls import run_controls, synthetic_geometry, synthetic_texture
        if bool(args.image) != bool(args.annotation):
            parser.error('--image and --annotation must be supplied together')
        if args.image:
            gray = cv2.imdecode(np.frombuffer(Path(args.image).read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError('Cannot decode control texture')
            shapes = read_labelme(args.annotation, coordinate_scale=args.scale)
        else:
            gray, shapes = synthetic_texture(), synthetic_geometry()
        results = run_controls(gray, shapes, progress=lambda name: print('control:', name, flush=True))
        output.write_text(json.dumps(jsonable(dict(config=CONFIG, results=results)), indent=2, allow_nan=False), encoding='utf-8')
    elif args.command == 'track':
        from .io import read_video
        from .tracking import track_segments
        grid = layout(read_labelme(args.annotation, coordinate_scale=args.scale))
        frames, t = read_video(args.video, time_factor=args.time_factor, coordinate_scale=args.scale)
        with np.load(args.poses, allow_pickle=False) as data:
            pose_time, poses, epoch = data['time_s'], data['poses'], data['pose_epoch']
            pose_valid = data['pose_valid'] if 'pose_valid' in data else None
        if pose_time.shape != t.shape or not np.allclose(pose_time, t, atol=1e-6, rtol=0):
            raise ValueError('Pose timestamps differ from the decoded acquisition timeline')
        if epoch.shape != t.shape:
            raise ValueError('pose_epoch must match acquisition timestamps')
        tracks = track_segments(frames, np.vstack((grid['A_points'], grid['B_points'])), t, poses,
                                pose_epoch=epoch, pose_valid=pose_valid)
        np.savez_compressed(output, **tracks)
        np.savez_compressed(output.with_name(output.stem+'_layout.npz'), **grid)
    elif args.command == 'features':
        from .deformation import compute_maps, compute_spacetime
        with np.load(args.tracks, allow_pickle=False) as data:
            tracks = dict(data)
        with np.load(args.layout, allow_pickle=False) as data:
            grid = dict(data)
        result = (compute_maps(tracks, grid, args.hop) if args.hop is not None else
                  compute_spacetime(tracks, grid, interval_s=args.interval_s))
        np.savez_compressed(output, **result)
    elif args.command == 'summarize':
        from .patient import summarize_patient
        with np.load(args.maps, allow_pickle=False) as data:
            spacetime = dict(data)
        review = json.loads(Path(args.review).read_text(encoding='utf-8')) if args.review else None
        result = summarize_patient(spacetime, s_range=args.s_range, time_range=args.time_range, review=review,
                                   refine_mode_speed=args.refine_mode_speed)
        output.write_text(json.dumps(jsonable(result), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    else:
        from .radial_events import radial_candidates_from_maps
        if output.suffix.lower() != '.json':parser.error('wave-candidates --output must end in .json')
        with np.load(args.maps, allow_pickle=False) as data:
            spacetime = dict(data)
        report, arrays = radial_candidates_from_maps(spacetime, noise_floor=args.noise_floor, s_range=args.s_range)
        report['source_maps'] = str(Path(args.maps))
        report['filtered_maps'] = str(output.with_suffix('.npz'))
        output.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        np.savez_compressed(output.with_suffix('.npz'), **arrays)


if __name__ == '__main__':
    main()
