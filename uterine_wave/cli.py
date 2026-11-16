"""Small command interface; measurement input deliberately contains no outcomes."""
import argparse
import json
from pathlib import Path
import numpy as np
from .geometry import CONFIG, read_labelme, layout
from .io import jsonable


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
    maps_cmd = commands.add_parser('features', help='Compute A/B maps from saved tracks and topology')
    maps_cmd.add_argument('--tracks', required=True)
    maps_cmd.add_argument('--layout', required=True)
    maps_cmd.add_argument('--hop', type=int, default=8)
    maps_cmd.add_argument('--output', required=True)
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
        if pose_time.shape != t.shape or not np.allclose(pose_time, t, atol=1e-6, rtol=0):
            raise ValueError('Pose timestamps differ from the decoded acquisition timeline')
        if epoch.shape != t.shape:
            raise ValueError('pose_epoch must match acquisition timestamps')
        tracks = track_segments(frames, np.vstack((grid['A_points'], grid['B_points'])), t, poses, pose_epoch=epoch)
        np.savez_compressed(output, **tracks)
        np.savez_compressed(output.with_name(output.stem+'_layout.npz'), **grid)
    else:
        from .deformation import compute_maps
        with np.load(args.tracks, allow_pickle=False) as data:
            tracks = dict(data)
        with np.load(args.layout, allow_pickle=False) as data:
            grid = dict(data)
        np.savez_compressed(output, **compute_maps(tracks, grid, args.hop))


if __name__ == '__main__':
    main()
