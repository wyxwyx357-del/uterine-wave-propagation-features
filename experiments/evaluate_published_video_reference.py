"""Fixed-pipeline feasibility check on author-described public ultrasound.

Encoded time and anatomical endpoint identity are NOT independent truth.
The two timing hypotheses are reported separately, never selected by outcome.
No formal frequency, accuracy, or event-count estimate is produced.
"""
import argparse
from collections import Counter
import hashlib
from html import escape
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uterine_wave.geometry import layout
from uterine_wave.deformation import compute_spacetime
from uterine_wave.io import jsonable
from experiments.partial_phase_events import detect_partial
from experiments.candidate_split_verifier import prepare_parent, decide_split
from experiments.global_event_ownership import reconcile
from experiments.apply_global_ownership_real import export_missing_observations


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retime(tracks, factor):
    """Reuse exactly the same image measurements; hypothesis changes only time."""
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('Expected positive time hypothesis')
    result = dict(tracks)
    result['time_s'] = tracks['time_s'] * factor
    return result


def image_direction(direction):
    # Layout's increasing s is explicitly RIGHT -> LEFT for these inputs.
    return {'C2F': 'right_to_left', 'F2C': 'left_to_right'}.get(direction)


def tracking_audit(frames, tracks, grid, fps):
    """Describe support and the largest frame change without rejecting frames."""
    difference = np.array([np.mean(abs(b.astype(float)-a.astype(float)))
                           for a, b in zip(frames[:-1], frames[1:])])
    index = int(np.argmax(difference))+1
    ids = grid['A_pair_ids']
    pairs = tracks['target_valid'][:, ids].all(3)
    radial_ids = ids.ravel()
    accepted = tracks['step_valid'][:, radial_ids]
    scores = tracks['target_ncc'][:, radial_ids][accepted]
    reasons, counts = np.unique(tracks['reject_reason'][index], return_counts=True)
    return dict(largest_frame_change_index=index, largest_frame_change_file_time_s=index/fps,
        mean_absolute_image_change_median=float(np.median(difference)),
        mean_absolute_image_change_max=float(difference[index-1]),
        pair_valid_fraction_before_change=float(pairs[index-1].mean()),
        pair_valid_fraction_at_change=float(pairs[index].mean()),
        reasons_at_change=dict(zip(reasons.tolist(), counts.tolist())),
        ncc_mean_on_accepted_radial_steps=float(np.mean(scores)) if len(scores) else None,
        ncc_semantics='Conditional on QC-accepted radial steps, NOT whole-video literature PCC',
        edit_or_cut_confirmed=False)


def render_tracks(path, frames, tracks, grid, fps, cv2):
    h, w = frames[0].shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'avc1'), fps, (2*w, 2*h))
    if not writer.isOpened():
        raise RuntimeError('H264 writer unavailable')
    ids = grid['A_pair_ids']
    for i, frame in enumerate(frames):
        canvas = cv2.cvtColor(cv2.resize(frame, (2*w, 2*h)), cv2.COLOR_GRAY2BGR)
        p = tracks['positions'][i]*2
        valid = tracks['target_valid'][i]
        for a, b in ids.reshape(-1, 2):
            if valid[a] and valid[b]:
                cv2.line(canvas, tuple(np.rint(p[a]).astype(int)), tuple(np.rint(p[b]).astype(int)),
                         (180, 200, 30), 1, cv2.LINE_AA)
            if valid[a]:
                cv2.circle(canvas, tuple(np.rint(p[a]).astype(int)), 3, (40, 240, 70), -1)
            if valid[b]:
                cv2.circle(canvas, tuple(np.rint(p[b]).astype(int)), 4, (20, 160, 255), 1)
        cv2.putText(canvas, f'File time {i/fps:.2f}s | original acquisition multiplier UNKNOWN',
                    (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'A0: green inner / orange outer | provisional image rails, not expert anatomy',
                    (12, canvas.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, .43, (240, 240, 240), 1, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()


def plot_case(path, hypotheses, reports, limit, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharex=True, sharey=True)
    for k, factor in enumerate((4., 8.)):
        maps = hypotheses[factor]
        file_t = maps['interval_center_s']/factor
        for wall in (0, 1):
            for col, values, label in ((2*k, maps['A'][:, wall], 'Raw'),
                                       (2*k+1, reports[factor][wall]['filtered_A'], 'Periodic + proposals')):
                ax = axes[wall, col]
                im = ax.pcolormesh(file_t, maps['s'], values.T, shading='nearest',
                                   cmap='RdBu_r', vmin=-limit, vmax=limit)
                ax.set_facecolor('#bbb'); ax.set_ylim(0, 1)
                ax.set_title(f'{factor:g}x ASSUMED / {label}', fontsize=10)
                ax.set_xlabel('Encoded file time (s)')
                if col == 0:
                    ax.set_ylabel(('Image upper' if wall == 0 else 'Image lower')+' / s: right -> left')
            ax = axes[wall, 2*k+1]
            for event in reports[factor][wall]['reconciliation']['event_proposals']:
                ids = event['supporting_sections']
                color = '#763d88' if event['ownership_status'].startswith('ambiguous') else '#078627'
                # Do not connect a curve across absent native sections.
                for block in np.split(np.arange(len(ids)), np.flatnonzero(np.diff(ids) != 1)+1):
                    if len(block):
                        ax.plot(np.asarray(event['arrival_time_s'])[block]/factor,
                                maps['s'][np.asarray(ids)[block]], '.-', color=color, lw=.8, ms=3)
    fig.suptitle(title+' | hypothesis check, NOT validated wave identities', fontsize=12)
    fig.subplots_adjust(left=.065, right=.93, bottom=.12, top=.86, wspace=.12, hspace=.3)
    fig.colorbar(im, cax=fig.add_axes([.945, .17, .012, .65]), label='A (s^-1), assumed time')
    fig.text(.04, .025, 'Shared color scale across ALL videos/hypotheses; gray = missing. '
             'Green = retained proposal; purple = unresolved ownership. '
             'No anatomy or timing chosen to match author labels.', fontsize=8)
    fig.savefig(path, dpi=140); plt.close(fig)


def run(args):
    args.reference = args.reference.resolve(); args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    geometry = read(args.geometry)
    source = read(args.reference/'published_reference_manifest.json')
    floor = read(args.noise_protocol)['noise_floor_rate']
    # Source module hashes protect the unchanged A0 / feature / event chain.
    names = ['uterine_wave/'+n+'.py' for n in ('tracking', '_lk', 'geometry', 'deformation',
             'amplitude', 'spectrum', 'radial_events')]
    names += ['experiments/'+n+'.py' for n in ('partial_phase_events', 'candidate_split_verifier',
              'global_event_ownership')]
    frozen = {n: digest(ROOT/n) for n in names}
    protocol = dict(status='feasibility_only', time_factors=[4., 8.],
        time_factor_source='Article describes four-to-eight speed-up; exact per-file multiplier unknown',
        time_hypotheses_not_selected_by_results=True, anatomical_endpoints_verified=False,
        geometry_sha256=digest(args.geometry), geometry_source=geometry['source'],
        source_manifest_sha256=digest(args.reference/'published_reference_manifest.json'),
        noise_protocol_sha256=digest(args.noise_protocol), noise_floor_rate=floor,
        coordinate_scale=.5, measurement_interval_s=2., frozen_code_sha256=frozen,
        pose_policy='Explicit identity: no background pose supplied. Radial lengths are rigid-invariant; '
                    'recovery therefore has no independently measured global-motion prior.',
        axis_policy='Increasing s is image right to left; API cervix/fundus names are computational '
                    'placeholders, NOT independently verified anatomical labels.',
        event_count=None, event_frequency=None, accuracy=None, patient_outcomes_read=False)
    save(args.output/'protocol.json', protocol)
    dll = os.add_dll_directory(str(args.opencv_bin)) if args.opencv_bin else None
    import cv2
    from uterine_wave.tracking import track_segments
    cv2.setNumThreads(2)
    results = []; all_values = []; plots = []
    for asset in source['assets']:
        if not asset['file'].endswith('.mp4'):
            continue
        name = Path(asset['file']).stem; destination = args.output/name
        destination.mkdir(exist_ok=True)
        shapes = {k: np.asarray(v, float)*.5 for k, v in geometry['videos'][name]['shapes'].items()}
        grid = layout(shapes)
        if not np.all(np.diff(grid['wall'][..., 0], axis=1) < 0):
            raise ValueError('Reference rails must increase right to left')
        previous = Path.cwd()
        try:
            os.chdir(args.reference)
            cap = cv2.VideoCapture(asset['file']); fps = cap.get(cv2.CAP_PROP_FPS)
            frames = []
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(cv2.resize(image, None, fx=.5, fy=.5,
                              interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
            cap.release()
        finally:
            os.chdir(previous)
        if len(frames) != asset['frames'] or fps != asset['encoded_fps']:
            raise ValueError('Decoded media differs from reference manifest')
        if digest(args.reference/asset['file']) != asset['sha256']:
            raise ValueError('Published media changed')
        t = np.arange(len(frames))/fps
        seeds = np.vstack((grid['A_points'], grid['B_points']))
        print('TRACK', name, len(t), len(seeds), flush=True)
        tracks_path = destination/'tracks_encoded_time.npz'
        if tracks_path.exists():
            cached = read(destination/'tracking_inputs.json')
            if cached != dict(video_sha256=asset['sha256'], geometry_sha256=digest(args.geometry),
                              frozen_code_sha256=frozen):
                raise ValueError('Cached tracking inputs differ')
            with np.load(tracks_path, allow_pickle=False) as z:
                tracks = dict(z)
        else:
            tracks = track_segments(frames, seeds, t, np.broadcast_to(np.eye(3), (len(t), 3, 3)).copy())
            np.savez_compressed(tracks_path, **tracks)
            save(destination/'tracking_inputs.json', dict(video_sha256=asset['sha256'],
                 geometry_sha256=digest(args.geometry), frozen_code_sha256=frozen))
        np.savez_compressed(destination/'grid.npz', **grid)
        audit = tracking_audit(frames, tracks, grid, fps)
        save(destination/'tracking_audit.json', audit)
        # Relative ASCII output avoids Windows OpenCV Unicode filename failures.
        previous = Path.cwd()
        try:
            os.chdir(destination)
            render_tracks(Path('tracking.mp4'), frames, tracks, grid, fps, cv2)
        finally:
            os.chdir(previous)
        hypotheses = {}; reports = {}
        for factor in (4., 8.):
            maps = compute_spacetime(retime(tracks, factor), grid, interval_s=2.)
            hypotheses[factor] = maps; walls = []; wall_summary = []
            for w in (0, 1):
                parent = detect_partial(maps['A'][:, w], maps['valid_mask'][:, w],
                                        maps['interval_center_s'], maps['s'], noise_floor=floor)
                filtered = parent.pop('filtered_A')
                span = float(np.median(maps['interval_dt_s']))
                local = [decide_split(prepare_parent(maps['A'][:, w], maps['valid_mask'][:, w],
                         maps['interval_center_s'], maps['s'], p, noise_floor=floor, measurement_span_s=span),
                         minimum_adjacent_support=3, include_signals=True) for p in parent['events']]
                reconciled = reconcile(parent, local, maps['interval_center_s'],
                                       maps['valid_mask'][:, w], measurement_span_s=span)
                export_missing_observations(local)
                records = reconciled['event_proposals']
                for event in records:
                    event['image_direction'] = image_direction(event['direction'])
                    event['anatomical_direction_verified'] = False
                measured = maps['A'][:, w][maps['valid_mask'][:, w]]
                wall_summary.append(dict(image_wall=('upper', 'lower')[w],
                    pair_valid_fraction=float(maps['valid_mask'][:, w].mean()),
                    raw_A_rms=float(np.sqrt(np.mean(measured**2))) if len(measured) else None,
                    proposal_rows=len(records),
                    image_directions=dict(Counter(e['image_direction'] or 'unresolved' for e in records)),
                    ambiguous_rows=sum(e['ownership_status'].startswith('ambiguous') for e in records),
                    directional_parent_evidence_kinds=dict(Counter(e['evidence_kind'] for e in parent['events']
                        if e['classification'] in ('C2F', 'F2C'))),
                    local_split_statuses=dict(Counter(p['split_status'] for p in local)),
                    global_split_action_count=len(reconciled['split_actions'])))
                walls.append(dict(parent_report=parent, local_split_proposals=local,
                                  reconciliation=reconciled, filtered_A=filtered))
                all_values.extend(abs(maps['A'][:, w][maps['valid_mask'][:, w]]).tolist())
            reports[factor] = walls
            np.savez_compressed(destination/f'maps_assumed_{factor:g}x.npz', **maps,
                                filtered_A=np.stack([w['filtered_A'] for w in walls], axis=1))
            save(destination/f'proposals_assumed_{factor:g}x.json', dict(
                time_factor_hypothesis=factor, acquisition_timing_verified=False,
                walls=[{k: v for k, v in w.items() if k != 'filtered_A'} for w in walls],
                event_count=None, event_frequency=None, accuracy=None))
            summary = dict(video=name, author_reference=asset['reference_kind'],
                assumed_time_factor=factor, encoded_duration_s=asset['encoded_duration_s'],
                hypothesis_duration_s=len(t)/fps*factor, native_sections=len(grid['u']),
                actual_measurement_span_s=float(np.median(maps['interval_dt_s'])),
                walls=wall_summary, tracking_audit=audit, anatomical_direction_agreement=None,
                event_count=None, event_frequency=None, accuracy=None)
            results.append(summary); print(json.dumps(summary), flush=True)
        plots.append((destination, hypotheses, reports, name))
    limit = float(np.percentile(all_values, 98)) if all_values else .001
    for destination, hypotheses, reports, name in plots:
        plot_case(destination/'spacetime.png', hypotheses, reports, limit, name)
    if {n: digest(ROOT/n) for n in names} != frozen:
        raise RuntimeError('Frozen measurement modules changed')
    save(args.output/'summary.json', dict(results=results, display_limit_A=limit,
        status='preliminary image/timing feasibility, not clinical validation',
        known_time_factor=False, anatomical_endpoints_verified=False,
        event_count=None, event_frequency=None, accuracy=None))
    cards = []
    for destination, _, _, name in plots:
        cards.append('<section><h2>'+escape(name)+'</h2><video controls src="'+name+
                     '/tracking.mp4"></video><p>绿点inner，橙圈outer；文件播放时间。图像上下壁与左右端点为初步取样，未经解剖专家确认。</p><a href="'+name+
                     '/spacetime.png"><img src="'+name+'/spacetime.png"></a></section>')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>公开视频初步对照</title>'
        '<style>body{font:17px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:15px}video{width:80%}img{width:100%}</style>'
        '<h1>期刊示例：A0追踪与径向图初步对照</h1><p>无需用户标注。作者传播/静息类别是外部参考；'
        '4/8倍均是假设，不能择优解释。尚无独立解剖端点标注；本页不报告准确率、波次数或次/分。</p>'
        '<p><a href="summary.json">测量摘要</a> · <a href="protocol.json">固定方案</a></p>'+''.join(cards), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--noise-protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--opencv-bin', type=Path)
    run(parser.parse_args())
