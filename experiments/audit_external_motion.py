"""Distributed external-point motion audit, without changing saved A0 results."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uterine_wave.motion_consensus import rigid_consensus
from uterine_wave.io import jsonable


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def select_external_points(gray, radial, cv2):
    """Up to eight corners in each of eight sectors outside the radial ROI."""
    mask = np.uint8(gray > 12)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        raise ValueError('No ultrasound image region')
    tissue = np.uint8(labels == 1+np.argmax(stats[1:, cv2.CC_STAT_AREA]))*255
    tissue = cv2.erode(tissue, np.ones((31, 31), np.uint8))
    hull = np.zeros_like(gray)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.rint(radial).astype(np.int32)), 255)
    distance = cv2.distanceTransform(255-hull, cv2.DIST_L2, 5)
    ring = (tissue > 0) & (distance >= 28) & (distance <= 135)
    yy, xx = np.indices(gray.shape)
    center = radial.mean(0)
    sectors = np.floor((np.arctan2(yy-center[1], xx-center[0])+np.pi)*8/(2*np.pi)).astype(int)%8
    points, region = [], []
    for sector in range(8):
        allowed = np.uint8(ring & (sectors == sector))*255
        corners = cv2.goodFeaturesToTrack(gray, 8, .01, 10, mask=allowed, blockSize=7)
        if corners is not None:
            points.extend(corners[:, 0].tolist())
            region.extend([sector]*len(corners))
    if len(points) < 3:
        raise ValueError('Too few external reference points')
    return np.asarray(points, np.float32), np.asarray(region), tissue, ring


def run(args):
    dll = os.add_dll_directory(str(args.opencv_bin)) if args.opencv_bin else None
    import cv2
    from uterine_wave._lk import match, local_contrast
    protocol = json.loads((args.source/'protocol.json').read_text(encoding='utf-8'))
    row = next(x for x in protocol['selected'] if x['review_id'] == args.review_id)
    source = args.source/args.review_id
    tracks, grid, maps = [load(source/(name+'.npz')) for name in ('tracks', 'grid', 'maps')]
    video = Path(row['video'])
    if not video.is_absolute():
        video = ROOT.parent/video
    args.output.mkdir(parents=True, exist_ok=True)
    all_t = tracks['time_s']
    indices = np.flatnonzero((all_t >= args.start_s) & (all_t <= args.end_s))
    if len(indices) < 3:
        raise ValueError('Requested time window too short')
    with tempfile.TemporaryDirectory(prefix='wave_external_motion_') as td:
        staged = Path(td)/'source.mp4'
        shutil.copyfile(video, staged)
        cap = cv2.VideoCapture(str(staged))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
        frames = []
        for index in indices:
            ok, image = cap.read()
            if not ok:
                raise ValueError('Source video ended before requested interval')
            frames.append(cv2.resize(image, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
        cap.release()
        if not np.allclose(all_t[indices], indices/fps, atol=.003, rtol=0):
            raise ValueError('Native frame times differ from saved tracks')
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        radial = tracks['positions'][indices[0], :len(grid['A_points'])]
        seeds, regions, tissue, ring = select_external_points(gray[0], radial, cv2)
        n, k = len(indices), len(seeds)
        positions = np.full((n, k, 2), np.nan, np.float32)
        valid = np.zeros((n, k), bool)
        agree = np.zeros_like(valid)
        positions[0], valid[0] = seeds, True
        transforms = np.full((n, 3, 3), np.nan)
        cumulative = np.full_like(transforms, np.nan)
        cumulative[0] = np.eye(3)
        old_relative = tracks['poses'][indices[0]]
        center = radial.mean(0)
        common_step = np.full((n, 2), np.nan)
        old_step = np.full_like(common_step, np.nan)
        residual_median = np.full(n, np.nan)
        old_prediction_residual = np.full(n, np.nan)
        external_prediction_residual = np.full(n, np.nan)
        fractions = np.full(n, np.nan)
        region_counts = np.zeros(n, int)
        majority = np.zeros(n, bool)
        previous_enhanced = local_contrast(gray[0])
        for i in range(1, n):
            enhanced = local_contrast(gray[i])
            active = np.flatnonzero(valid[i-1])
            if len(active):
                q, good, _, _, _ = match(previous_enhanced, enhanced, gray[i-1], gray[i],
                                         positions[i-1, active], 21)
                keep = active[good]
                positions[i, keep] = q[good]
                valid[i, keep] = True
                a, b = positions[i-1, keep], positions[i, keep]
                result = rigid_consensus(a, b, args.agreement_px)
                transforms[i] = result['matrix']
                agree[i, keep] = result['inliers']
                fractions[i] = result['inlier_fraction']
                majority[i] = result['majority_found']
                region_counts[i] = len(np.unique(regions[keep[result['inliers']]]))
                if len(keep):
                    old = np.linalg.inv(tracks['poses'][indices[i]]) @ tracks['poses'][indices[i-1]]
                    old_step[i] = old[:2, :2] @ center + old[:2, 2] - center
                    old_prediction_residual[i] = np.median(np.linalg.norm(a @ old[:2, :2].T + old[:2, 2] - b, axis=1))
                if np.isfinite(transforms[i]).all():
                    transform = transforms[i]
                    external_prediction_residual[i] = np.median(result['residual_px'])
                    common_step[i] = transform[:2, :2] @ center + transform[:2, 2] - center
                    if result['inlier_points']:
                        residual_median[i] = np.median(result['residual_px'][result['inliers']])
                    if majority[i] and np.isfinite(cumulative[i-1]).all():
                        cumulative[i] = cumulative[i-1] @ np.linalg.inv(transform)
            previous_enhanced = enhanced
        # Native increments alone can look unanimous because displacement is tiny.
        # Independently test the same actual ~2 s support as the radial measure.
        native = all_t[indices]
        hop = int(np.floor(2/np.median(np.diff(native))+.5))
        if n <= hop:
            raise ValueError('External motion audit needs more than two seconds')
        span_inlier = np.zeros_like(valid)
        span_fraction = np.full(n, np.nan)
        span_majority = np.zeros(n, bool)
        span_regions = np.zeros(n, int)
        span_transform = np.full_like(transforms, np.nan)
        for i in range(hop, n):
            supported = valid[i-hop:i+1].all(0)
            ids = np.flatnonzero(supported)
            result = rigid_consensus(positions[i-hop, ids], positions[i, ids], args.agreement_px)
            span_transform[i] = result['matrix']
            span_inlier[i, ids] = result['inliers']
            span_fraction[i] = result['inlier_fraction']
            span_majority[i] = result['majority_found']
            span_regions[i] = len(np.unique(regions[ids[result['inliers']]]))
        # Agreement never deletes a QC-supported point from the next fit.
        saved = dict(time_s=all_t[indices], frame_indices=indices, positions=positions,
                     target_valid=valid, consensus_inlier=agree, region_id=regions,
                     source_to_target_transform=transforms, image_to_clip_reference=cumulative,
                     majority_found=majority, inlier_fraction=fractions,
                     inlier_region_count=region_counts, common_step_analysis_px=common_step,
                     existing_step_analysis_px=old_step, fit_residual_analysis_px=residual_median,
                     external_prediction_residual_analysis_px=external_prediction_residual,
                     existing_prediction_residual_analysis_px=old_prediction_residual,
                     interval_hop=np.array(hop), interval_consensus_inlier=span_inlier,
                     interval_inlier_fraction=span_fraction, interval_majority_found=span_majority,
                     interval_inlier_region_count=span_regions, interval_transform=span_transform)
        np.savez_compressed(args.output/'external_motion.npz', **saved)
        native = all_t[indices]
        # Draw image content only; exclude scanner headers from the review export.
        y, x = np.nonzero(tissue)
        bounds = [int(x.min()), int(x.max()+1), int(y.min()), int(y.max()+1)]
        x0, x1, y0, y1 = bounds
        panel_w, panel_h = 760, int(round(760*(y1-y0)/(x1-x0)))
        height = ((panel_h+130+1)//2)*2
        writer_path = Path(td)/'comparison.mp4'
        writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*'avc1'), fps, (1520, height))
        if not writer.isOpened():
            raise RuntimeError('H264 video writer unavailable')
        preview = None
        for i, frame in enumerate(frames):
            left = frame.copy()
            roi_now = tracks['positions'][indices[i], :len(grid['A_points'])]
            roi_now = roi_now[np.isfinite(roi_now).all(1)]
            if len(roi_now) >= 3:
                cv2.polylines(left, [cv2.convexHull(np.rint(roi_now).astype(np.int32))], True, (255, 255, 255), 1)
            for j in np.flatnonzero(valid[i]):
                color = ((200, 180, 100) if i < hop else
                         (80, 230, 70) if span_inlier[i, j] else (0, 165, 255))
                cv2.circle(left, tuple(np.rint(positions[i, j]).astype(int)), 3, color, -1)
            stable = np.isfinite(cumulative[i]).all()
            right = cv2.warpAffine(frame, cumulative[i, :2], (frame.shape[1], frame.shape[0])) if stable else frame.copy()
            canvas = np.full((height, 1520, 3), 22, np.uint8)
            for panel, image in enumerate((left, right)):
                crop = cv2.resize(image[y0:y1, x0:x1], (panel_w, panel_h))
                canvas[90:90+panel_h, panel*760:(panel+1)*760] = crop
            cv2.putText(canvas, f'Native time {native[i]:.2f}s | original + external reference points', (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .62, (245, 245, 245), 1)
            label = ('Incremental preview; ~2s majority supported' if span_majority[i] else
                     'Incremental preview; ~2s common motion UNCONFIRMED') if stable else 'No continuous reference'
            cv2.putText(canvas, label, (775, 27), cv2.FONT_HERSHEY_SIMPLEX, .60, (245, 245, 245), 1)
            text = f'QC valid {valid[i].sum()}/{k}; ~2s agreement {span_inlier[i].sum()}/{valid[i].sum()}; sectors {span_regions[i]}/8'
            cv2.putText(canvas, text, (12, 61), cv2.FONT_HERSHEY_SIMPLEX, .58, (245, 245, 245), 1)
            cv2.putText(canvas, 'Green: ~2s rigid agreement. Orange: other QC-supported points. Early blue: no ~2s comparison yet. Playback 1x.', (12, height-14), cv2.FONT_HERSHEY_SIMPLEX, .54, (245, 245, 245), 1)
            writer.write(canvas)
            if abs(native[i]-args.preview_s) < .5/fps:
                preview = canvas.copy()
        writer.release()
        check = cv2.VideoCapture(str(writer_path))
        encoded_count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
        encoded_fps = check.get(cv2.CAP_PROP_FPS)
        check.release()
        if encoded_count != n or abs(encoded_fps-fps) > .001:
            raise ValueError('Review video lost native frames/time')
        shutil.copyfile(writer_path, args.output/'comparison_1x.mp4')
        from PIL import Image
        if preview is not None:
            Image.fromarray(preview[:, :, ::-1]).save(args.output/'preview.png')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for component, color in enumerate(('tab:blue', 'tab:orange')):
        axes[0].plot(native, common_step[:, component]*2*fps, color=color, label=f'external {"xy"[component]}')
        axes[0].plot(native, old_step[:, component]*2*fps, '--', color=color, alpha=.5, label=f'existing {"xy"[component]}')
    axes[0].set_ylabel('px/s at ROI center'); axes[0].legend(ncol=4)
    axes[1].plot(native, valid.sum(1), label='QC supported')
    axes[1].plot(native, span_inlier.sum(1), label='~2 s rigid agreement')
    axes[1].plot(native, span_regions, label='~2 s supporting sectors / 8')
    axes[1].set_ylabel('Reference point count'); axes[1].legend(ncol=3)
    axes[2].plot(native, 2*residual_median, label='external fit: inliers only')
    axes[2].plot(native, 2*external_prediction_residual, label='external fit: all QC reference points')
    axes[2].plot(native, 2*old_prediction_residual, label='existing pose: all QC reference points')
    axes[2].set_ylabel('Residual (native px)'); axes[2].legend()
    mt = maps['interval_center_s']
    a = np.where(maps['valid_mask'], maps['A'], np.nan)
    for wall in (0, 1):
        rms = np.sqrt(np.nanmean(a[:, wall]**2, axis=1))
        axes[3].plot(mt, rms, label=('anterior' if wall == 0 else 'posterior'))
    axes[3].set_ylabel('Saved A0 radial RMS (1/s)'); axes[3].legend()
    axes[3].set_xlim(native[0], native[-1]); axes[3].set_xlabel('Native acquisition time (s)')
    for ax in axes:
        ax.axvline(args.preview_s, color='.4', ls=':'); ax.grid(alpha=.2)
    fig.savefig(args.output/'motion_diagnostics.png', dpi=130)
    plt.close(fig)
    summary = dict(review_id=args.review_id, native_time_range_s=[native[0], native[-1]], frames=n,
                   fps=fps, initial_reference_points=k, initial_sectors=len(np.unique(regions)),
                   selection='8 angular sectors; max 8 corners/sector; outside radial hull 28..135 analysis px',
                   LK_window=21, agreement_tolerance_analysis_px=args.agreement_px,
                   tolerance_source='Exploratory rigid-model agreement, not a physiological exclusion threshold',
                   majority_step_fraction=float(majority[1:].mean()),
                   actual_consensus_interval_s=float(np.median(native[hop:]-native[:-hop])),
                   majority_interval_fraction=float(span_majority[hop:].mean()),
                   interval_inlier_fraction_median=float(np.nanmedian(span_fraction[hop:])),
                   interval_supporting_sectors_median=float(np.median(span_regions[hop:])),
                   valid_points_median=float(np.median(valid[1:].sum(1))),
                   inlier_fraction_median=float(np.nanmedian(fractions[1:])),
                   supporting_sectors_median=float(np.median(region_counts[1:])),
                   rigid_fit_inlier_residual_native_px_median=float(np.nanmedian(residual_median)*2),
                   external_pose_all_reference_residual_native_px_median=float(np.nanmedian(external_prediction_residual)*2),
                   existing_pose_all_reference_residual_native_px_median=float(np.nanmedian(old_prediction_residual)*2),
                   residuals_are_not_independent_error_comparison=True,
                   video='comparison_1x.mp4', saved_A0_changed=False, wave_detector_changed=False,
                   limitations=['External tissue motion is not independent probe pose truth.',
                                'Majority of points does not guarantee correct tissue reference.',
                                'Rigid stabilization cannot repair OOP motion or differential localization error.',
                                'Rigid translation/rotation preserve radial pair distance; no amplitude subtraction performed.'])
    if args.check_end_s is not None:
        i, j = [int(np.argmin(abs(native-value))) for value in (args.preview_s, args.check_end_s)]
        if j <= i:
            raise ValueError('Check end must follow preview/check start within the clip')
        keep = valid[i:j+1].all(0)
        delta = 2*(positions[j, keep]-positions[i, keep])
        if not len(delta):
            raise ValueError('No continuous external points in check interval')
        summary['endpoint_check'] = dict(time_range_s=native[[i, j]], supported_points=len(delta),
            moving_right=int((delta[:, 0] > 0).sum()), moving_left=int((delta[:, 0] < 0).sum()),
            median_displacement_native_px=np.median(delta, axis=0),
            horizontal_displacement_p10_p90_native_px=np.quantile(delta[:, 0], [.1, .9]),
            sector_median_displacement_native_px={str(int(region)): np.median(delta[regions[keep] == region], axis=0)
                                                for region in np.unique(regions[keep])},
            signs_are_descriptive_not_a_probe_motion_test=True)
    (args.output/'summary.json').write_text(json.dumps(jsonable(summary), indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(jsonable(summary), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--review-id', required=True)
    parser.add_argument('--start-s', type=float, required=True)
    parser.add_argument('--end-s', type=float, required=True)
    parser.add_argument('--preview-s', type=float, required=True)
    parser.add_argument('--check-end-s', type=float)
    parser.add_argument('--agreement-px', type=float, default=1.)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--opencv-bin', type=Path)
    run(parser.parse_args())
