"""Sequential A0 -> 2 s spacetime -> frozen patient features, local only."""
import csv
import json
from pathlib import Path
import time
import traceback
import numpy as np

from .input_inventory import BASE_COMMIT, local_video_name, save_json, sha256
from .io import jsonable


class ProgressFrames(list):
    def __getitem__(self, index):
        if isinstance(index, int) and index > getattr(self, '_last', 0)+300:
            self._last = index
            print('TRACK_FRAME', index, '/', len(self), flush=True)
        return super().__getitem__(index)


def _empty_result(row):
    keys = ('case_id', 'video_path', 'annotation_path', 'pose_path', 'video_duration_s')
    return dict({k: row.get(k) for k in keys}, processing_status='not_run', failure_reason=None,
        tracking_success=None, spacetime_success=None, feature_success=None,
        nominal_interval_s=2., actual_interval_s_range=None,
        anterior_radial_stretch_activity_amplitude=None, posterior_radial_stretch_activity_amplitude=None,
        anterior_valid_fraction=None, posterior_valid_fraction=None,
        anterior_common_region_duration_s=None, posterior_common_region_duration_s=None,
        anterior_amplitude_status=None, posterior_amplitude_status=None,
        mode_features_status=None, feature_file=None, stage_history=[])


def process_case(row, *, s_range, output):
    """Catch failure once at the case boundary; save the exact stage/traceback."""
    from .geometry import read_labelme, layout
    from .io import read_video
    from .tracking import track_segments
    from .deformation import compute_spacetime
    from .patient import summarize_patient

    result = _empty_result(row)
    folder = Path(output).resolve()/row['case_id']
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stage = 'input_validation'
    try:
        if row['processing_status'] != 'ready_for_tracking':
            raise ValueError(row.get('failure_reason') or 'inputs_not_ready')
        for key in ('annotation', 'pose'):
            if sha256(row[key+'_path']) != row[key+'_sha256']:
                raise ValueError(key+'_changed_after_input_audit')
        video = Path(row['video_path'])
        if video.stat().st_size != row['video_size_bytes']:
            raise ValueError('video_size_changed_after_input_audit')
        grid = layout(read_labelme(row['annotation_path'], coordinate_scale=.5))
        stage = 'video_decode'
        print('DECODE', row['case_id'], flush=True)
        with local_video_name(video) as name:
            frames, t = read_video(name, time_factor=row['time_factor'], coordinate_scale=.5)
        with np.load(row['pose_path'], allow_pickle=False) as z:
            pose = dict(z)
        if (len(frames) != row['video_frame_count'] or pose['time_s'].shape != t.shape or
                not np.allclose(pose['time_s'], t, atol=1e-6, rtol=0)):
            raise ValueError('full_decoded_video_pose_timeline_mismatch')
        result['full_decode_checked'] = True
        result['stage_history'].append('video_decode_success')
        stage = 'tracking'
        result['tracking_success'] = False
        print('A0', row['case_id'], len(t), 'frames', flush=True)
        tracks = track_segments(ProgressFrames(frames), np.vstack((grid['A_points'], grid['B_points'])),
            t, pose['poses'], pose_epoch=pose['pose_epoch'], pose_valid=pose.get('pose_valid'))
        np.savez_compressed(folder/'tracks.npz', **tracks)
        np.savez_compressed(folder/'layout.npz', **grid)
        result.update(tracking_success=True)
        result['stage_history'].append('tracking_success')
        del frames
        stage = 'spacetime'
        result['spacetime_success'] = False
        print('SPACETIME', row['case_id'], flush=True)
        maps = compute_spacetime(tracks, grid, interval_s=2.)
        np.savez_compressed(folder/'spacetime_2s.npz', **maps)
        result['spacetime_success'] = True
        result['stage_history'].append('spacetime_success')
        stage = 'features'
        result['feature_success'] = False
        print('FROZEN_FEATURES', row['case_id'], flush=True)
        features = summarize_patient(maps, s_range=s_range, include_mode_velocity=False)
        # No review is supplied; no detector, event count or velocity is run.
        features['pipeline_scope'] = 'frozen_amplitude_mode_similarity_and_support_only'
        features['mode_velocity_computed'] = False
        features['event_velocity_units'] = None
        save_json(folder/'features.json', features)
        save_json(folder/'mode_windows.json', features['modes'])
        save_json(folder/'section_support.json', dict(requested_s_range=features['requested_s_range'],
            sampled_s_range=features['sampled_s_range'], duration_definition=features['duration_definition'],
            walls=features['walls']))
        for name, wall in zip(('anterior', 'posterior'), features['walls']):
            for source, target in (('radial_stretch_activity_amplitude', 'radial_stretch_activity_amplitude'),
                                   ('valid_fraction', 'valid_fraction'),
                                   ('common_region_duration_s', 'common_region_duration_s'),
                                   ('amplitude_status', 'amplitude_status'),
                                   ('requested_sections', 'requested_sections'),
                                   ('analyzed_sections', 'analyzed_sections'),
                                   ('amplitude_covers_all_requested_sections', 'complete_roi_coverage')):
                result[name+'_'+target] = wall[source]
        modes = features['modes']
        result.update(feature_success=True, processing_status='completed',
            requested_s_range=features['requested_s_range'], sampled_s_range=features['sampled_s_range'],
            actual_interval_s_range=features['actual_interval_s_range'],
            duration_definition=features['duration_definition'],
            mode_features_status='available' if modes['windows'] else 'no_supported_mode_windows',
            mode_window_count=len(modes['windows']), paired_ER_window_count=len(modes['paired_ER_windows']),
            coordination_block_count=len(modes['coordination']),
            CC_available_blocks=sum(r['zero_lag_ER_Pearson_CC'] is not None for r in modes['coordination']),
            MSE_available_blocks=sum(r['ER_MSE'] is not None for r in modes['coordination']),
            feature_file=str(folder/'features.json'))
        result['stage_history'].append('feature_success')
    except Exception as error:
        # Explicit per-case isolation, no fallback algorithm or zero features.
        result.update(processing_status=stage+'_failed', failure_reason=str(error),
                      error_type=type(error).__name__)
        save_json(folder/'failure.json', dict(stage=stage, exception=type(error).__name__,
                  reason=str(error), traceback=traceback.format_exc()))
        print('FAILED', row['case_id'], stage, str(error), flush=True)
    result['elapsed_seconds'] = time.monotonic()-started
    save_json(folder/'processing.json', result)
    return result


def write_manifest(output, rows, config):
    output = Path(output)
    save_json(output/'manifest.json', dict(config=config, cases=rows))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output/'manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(jsonable(v), ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def run_manifest(manifest, *, s_range, output, limit=None, all_cases=False):
    if not (len(s_range) == 2 and np.isfinite(s_range).all() and 0 <= s_range[0] < s_range[1] <= 1):
        raise ValueError('An explicit ordered s_range inside 0..1 is required')
    if not all_cases and (limit is None or limit < 1):
        raise ValueError('Specify a technical-test limit or explicitly request all cases')
    manifest = Path(manifest).resolve()
    source = json.loads(manifest.read_text(encoding='utf-8'))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    config = dict(baseline_commit=BASE_COMMIT, s_range=list(s_range), nominal_interval_s=2.,
                  source_manifest=str(manifest), source_manifest_sha256=sha256(manifest),
                  limit=limit, all_cases=all_cases, mode_velocity=False, event_detection=False,
                  features_entry='uterine_wave.patient.summarize_patient', outcomes_read=False)
    processed = 0
    for raw in source['cases']:
        row = dict(raw)
        for key in ('video_path', 'annotation_path', 'pose_path'):
            if row.get(key) and not Path(row[key]).is_absolute():
                row[key] = str((manifest.parent/row[key]).resolve())
        if row['processing_status'] != 'ready_for_tracking':
            result = _empty_result(row)
            result.update(processing_status='input_invalid', failure_reason=row.get('failure_reason'))
        elif not all_cases and processed >= limit:
            result = _empty_result(row)
            result['processing_status'] = 'not_run_technical_batch_limit'
        else:
            processed += 1
            result = process_case(row, s_range=s_range, output=output)
        rows.append(result)
        write_manifest(output, rows, config)
    return rows
