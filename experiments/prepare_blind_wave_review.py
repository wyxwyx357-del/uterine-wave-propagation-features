"""Prepare local original-video review without exposing candidate scores.

This does not rerun tracking, change candidates, or establish reference truth.
The private review key is deliberately not loaded by the browser.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name('blind_wave_review.html')
LABELS = ('unreviewed', 'definite_propagation', 'definite_local_activity',
          'no_credible_visible_activity', 'uncertain')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def review_items(cases, seed=20261118):
    """Random order independent of magnitude/clinical data; explicit key only."""
    rng = random.Random(seed)
    cases = list(cases)
    rng.shuffle(cases)
    items, key = [], []
    for number, (review_id, candidates) in enumerate(cases, 1):
        alias = f'P{number:02d}'
        for wall in candidates['walls']:
            for event in wall['events']:
                if event['classification'] not in ('C2F', 'F2C'):
                    continue
                items.append(dict(case_alias=alias, wall=int(wall['wall']),
                                  sections=[int(x) for x in event['sections']],
                                  start_s=float(event['start_s']),
                                  end_s=float(event['end_s'])))
                key.append(dict(review_id=review_id, event_id=event['event_id']))
    order = list(range(len(items)))
    rng.shuffle(order)
    shuffled, private = [], []
    for number, index in enumerate(order, 1):
        token = f'R{number:03d}'
        shuffled.append(dict(token=token, **items[index]))
        private.append(dict(token=token, **key[index], **items[index]))
    return shuffled, private


def display_tracks(tracks, grid, image_scale):
    """Image coordinates only. Missing points stay null; no corrected overlay."""
    positions = tracks['positions'][:, :len(grid['A_points'])] / image_scale
    valid = tracks['target_valid'][:, :positions.shape[1]]
    if positions.shape[:2] != valid.shape or len(positions) != len(tracks['time_s']):
        raise ValueError('Tracking dimensions differ')
    if len(tracks['time_s']) < 2 or np.any(np.diff(tracks['time_s']) <= 0):
        raise ValueError('Expected increasing native frame times')
    frames = [[[round(float(p[0]), 1), round(float(p[1]), 1)]
               if good and np.isfinite(p).all() else None
               for p, good in zip(frame, mask)]
              for frame, mask in zip(positions, valid)]
    return dict(frames=frames, time_s=tracks['time_s'].tolist(),
                pair_ids=grid['A_pair_ids'].tolist())


def build(args):
    # Keep this handle alive until OpenCV has finished reading video headers.
    dll = os.add_dll_directory(str(args.opencv_bin)) if args.opencv_bin else None
    import cv2
    protocol = read_json(args.source / 'protocol.json')
    cases = [(row['review_id'], read_json(args.source / row['review_id'] / 'candidates.json'))
             for row in protocol['selected']]
    items, key = review_items(cases, args.seed)
    signature = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    args.output.mkdir(parents=True, exist_ok=True)
    metadata, sources = {}, []
    aliases = {entry['review_id']: entry['case_alias'] for entry in key}
    for row in protocol['selected']:
        rid = row['review_id']
        # A case without directional candidates has no item in this review.
        if rid not in aliases:
            continue
        alias = aliases[rid]
        folder = args.source / rid
        with np.load(folder / 'tracks.npz', allow_pickle=False) as z:
            tracks = dict(z)
        with np.load(folder / 'grid.npz', allow_pickle=False) as z:
            grid = dict(z)
        video = Path(row['video'])
        if not video.is_absolute():
            video = ROOT.parent / video
        dest = args.output / alias
        dest.mkdir(exist_ok=True)
        copy = dest / 'source_1x.mp4'
        video_hash = digest(video)
        if not copy.exists() or digest(copy) != video_hash:
            shutil.copyfile(video, copy)
        if digest(copy) != video_hash:
            raise ValueError('Copied original video differs')
        with tempfile.TemporaryDirectory(prefix='blind_wave_header_') as td:
            staged = Path(td) / 'source.mp4'
            shutil.copyfile(copy, staged)
            cap = cv2.VideoCapture(str(staged))
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, first = cap.read()
            cap.release()
        if not ok or count != len(tracks['time_s']):
            raise ValueError('Original frames and saved tracking differ')
        # These cohort caches were explicitly produced at half native size.
        frame_times = np.arange(count) / fps
        if not np.allclose(tracks['time_s'], frame_times, atol=.003, rtol=0):
            raise ValueError('Original video and tracking timelines differ')
        data = display_tracks(tracks, grid, args.image_scale)
        from PIL import Image
        Image.fromarray(first[:, :, ::-1]).save(dest / 'first_frame.jpg')
        # ROI includes both walls plus context, never the candidate alone.
        finite = np.asarray(tracks['initial_points'][:len(grid['A_points'])]) / args.image_scale
        height, width = first.shape[:2]
        lo = np.maximum(np.floor(finite.min(0) - 65), [0, 0]).astype(int)
        hi = np.minimum(np.ceil(finite.max(0) + 65), [width, height]).astype(int)
        data.update(width=width, height=height,
                    roi=[int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])])
        (dest / 'display_data.js').write_text(
            'window.blindFrameData=window.blindFrameData||{};window.blindFrameData['
            + json.dumps(alias) + ']=' + json.dumps(data, separators=(',', ':'),
                                                    allow_nan=False) + ';', encoding='utf-8')
        metadata[alias] = dict(duration_s=count / fps, fps=fps,
                               video=f'{alias}/source_1x.mp4',
                               poster=f'{alias}/first_frame.jpg',
                               data=f'{alias}/display_data.js')
        sources.append(dict(case_alias=alias, review_id=rid, frames=count,
                            source_video_sha256=video_hash,
                            tracks_sha256=digest(folder / 'tracks.npz'),
                            candidates_sha256=digest(folder / 'candidates.json')))
    # Viewer data whitelist intentionally omits scores, inferred direction and flags.
    viewer = dict(bundle_id=signature, cases=metadata, items=items, labels=LABELS)
    (args.output / 'review_data.js').write_text('window.blindReview=' + json.dumps(
        viewer, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + ';', encoding='utf-8')
    shutil.copyfile(TEMPLATE, args.output / 'index.html')
    write_json(args.output / 'review_key_private.json', dict(bundle_id=signature, items=key,
                                                           sources=sources))
    write_json(args.output / 'protocol.json', dict(
        scope='Existing directional candidates only; not full-recording sensitivity validation',
        original_video_preserved=True, review_order_seed=args.seed,
        source_image_scale=args.image_scale, cases=len(metadata), items=len(items),
        display_padding_s=2, padding_is='Viewing context only, not event definition',
        blinded_fields=['candidate direction', 'magnitude', 'heatmap', 'QC scores', 'clinical values'],
        limitations=['Candidate locations/times remain visible; review is not fully blinded.',
                     'Original scan may contain embedded identifiers; keep this folder local.',
                     'No credible visible activity is a human observation, not proof of noise.',
                     'These cases already participated in development; no independent validation claim.'],
        source_verification=sources))
    print(json.dumps(dict(cases=len(metadata), items=len(items),
                          index=str(args.output / 'index.html')), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-scale', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=20261118)
    parser.add_argument('--opencv-bin', type=Path)
    args = parser.parse_args()
    if not 0 < args.image_scale <= 1:
        parser.error('--image-scale must be in (0,1]')
    build(args)
