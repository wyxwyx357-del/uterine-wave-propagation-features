"""Local entry point: audit first, then an explicitly limited end-to-end run."""
import argparse
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opencv-dll-dir', help='Windows CUDA OpenCV DLL directory, if required')
    commands = parser.add_subparsers(dest='command', required=True)
    audit = commands.add_parser('audit', help='Inventory only; no tracking or features')
    audit.add_argument('--cohort', type=Path, required=True)
    audit.add_argument('--data-root', type=Path, required=True)
    audit.add_argument('--annotation-root', type=Path, action='append', required=True)
    audit.add_argument('--pose-root', type=Path, action='append', default=[])
    audit.add_argument('--pose-index', type=Path, help='Explicit case_id/cache_path CSV registry')
    audit.add_argument('--allow-legacy-epoch', action='store_true', help='Copy legacy epoch to pose_epoch without changing values')
    audit.add_argument('--time-factor', type=float, required=True)
    audit.add_argument('--output', type=Path, required=True)
    run = commands.add_parser('run', help='A0 -> 2s spacetime -> frozen features')
    run.add_argument('--manifest', type=Path, required=True)
    run.add_argument('--s-range', type=float, nargs=2, required=True)
    batch = run.add_mutually_exclusive_group(required=True)
    batch.add_argument('--limit', type=int, help='Maximum eligible cases to attempt for technical testing')
    batch.add_argument('--all', action='store_true', help='Explicit full-cohort request')
    run.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    handle = os.add_dll_directory(args.opencv_dll_dir) if args.opencv_dll_dir else None
    if args.command == 'audit':
        import numpy as np
        if not np.isfinite(args.time_factor) or args.time_factor <= 0:
            parser.error('time-factor must be finite and positive')
        from uterine_wave.input_inventory import audit_inputs
        audit_inputs(args.cohort, data_root=args.data_root, annotation_roots=args.annotation_root,
            pose_roots=args.pose_root, pose_index=args.pose_index, output=args.output,
            time_factor=args.time_factor, allow_legacy_epoch=args.allow_legacy_epoch)
    else:
        from uterine_wave.frozen_pipeline import run_manifest
        run_manifest(args.manifest, s_range=args.s_range, output=args.output,
                     limit=args.limit, all_cases=args.all)
    if handle is not None:
        handle.close()


if __name__ == '__main__':
    main()

