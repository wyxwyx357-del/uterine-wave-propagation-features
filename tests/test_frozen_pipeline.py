import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import numpy as np

from uterine_wave.frozen_pipeline import run_manifest
from uterine_wave.input_inventory import match_annotation, read_cohort, save_json, sha256
from uterine_wave.patient import summarize_patient
from tests.test_patient import maps


def prepared(root, case):
    folder = root/case
    folder.mkdir()
    video = folder/'video.mp4'
    video.write_bytes(b'test video handled by mocked decoder')
    shapes = dict(ant_wall=[[60, 140], [620, 140]], post_wall=[[60, 240], [620, 240]],
                  cervix_point=[[60, 190]], fundus_point=[[620, 190]])
    annotation = folder/'first_frame.json'
    save_json(annotation, dict(imageWidth=700, imageHeight=400,
              shapes=[dict(label=k, points=v) for k, v in shapes.items()]))
    t = np.arange(61, dtype=float)
    pose = folder/'poses.npz'
    np.savez(pose, time_s=t, poses=np.repeat(np.eye(3)[None], len(t), axis=0),
             pose_epoch=np.zeros(len(t), int))
    return dict(case_id=case, video_path=str(video), annotation_path=str(annotation), pose_path=str(pose),
        annotation_sha256=sha256(annotation), pose_sha256=sha256(pose), video_size_bytes=video.stat().st_size,
        video_frame_count=len(t), video_duration_s=61., processing_status='ready_for_tracking',
        time_factor=1.)


def decoded(*args, **kwargs):
    return [np.zeros((200, 350), np.uint8) for _ in range(61)], np.arange(61, dtype=float)


def tracked(frames, initial, t, poses, **kwargs):
    p = np.repeat(np.asarray(initial, float)[None], len(t), axis=0)
    p[:, :, 1] *= 1+.01*np.sin(2*np.pi*t[:, None]/30)
    valid = np.ones(p.shape[:2], bool)
    steps = valid.copy()
    steps[0] = False
    return dict(corrected=p, positions=p, time_s=t, target_valid=valid, step_valid=steps,
                segment_id=np.zeros_like(valid, int), pose_epoch=np.zeros(len(t), int))


class FrozenPipelineTests(unittest.TestCase):
    def test_tracking_automatically_continues_to_real_spacetime_and_features(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = prepared(root, '1-A-20240101')
            save_json(root/'input.json', dict(cases=[row]))
            with patch('uterine_wave.io.read_video', side_effect=decoded), \
                 patch('uterine_wave.tracking.track_segments', side_effect=tracked), \
                 patch('uterine_wave.modes.mode_velocity', side_effect=AssertionError('speed must not run')):
                result = run_manifest(root/'input.json', s_range=(.2, .8), output=root/'out', limit=1)[0]
            self.assertEqual(result['processing_status'], 'completed')
            self.assertEqual(result['stage_history'], ['video_decode_success', 'tracking_success',
                                                     'spacetime_success', 'feature_success'])
            self.assertEqual(result['duration_definition'], 'union_of_supported_measurement_intervals')
            folder = root/'out'/row['case_id']
            with np.load(folder/'spacetime_2s.npz') as data:
                self.assertEqual(int(data['hop']), 2)
                self.assertEqual(float(data['nominal_interval_s']), 2.)
                self.assertIn('interval_start_s', data.files)
            features = json.loads((folder/'features.json').read_text())
            self.assertIsNone(features['event_summary']['event_count'])
            self.assertFalse(features['mode_velocity_computed'])
            self.assertTrue(features['modes']['windows'])
            self.assertNotIn('C2F', features['modes']['windows'][0])
            self.assertIn('positive_frequency_bins', features['walls'][0]['per_pair'][0]['blocks'][0])

    def test_tracking_failure_does_not_stop_next_case_or_write_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [prepared(root, name) for name in ('1-A-20240101', '2-B-20240102')]
            save_json(root/'input.json', dict(cases=rows))
            calls = []
            def run(*args, **kwargs):
                calls.append(1)
                if len(calls) == 1:
                    raise RuntimeError('expected test failure')
                return tracked(*args, **kwargs)
            with patch('uterine_wave.io.read_video', side_effect=decoded), \
                 patch('uterine_wave.tracking.track_segments', side_effect=run):
                results = run_manifest(root/'input.json', s_range=(.2, .8), output=root/'out', limit=2)
            self.assertEqual(results[0]['processing_status'], 'tracking_failed')
            self.assertFalse(results[0]['tracking_success'])
            self.assertIsNone(results[0]['spacetime_success'])
            self.assertIsNone(results[0]['anterior_radial_stretch_activity_amplitude'])
            self.assertEqual(results[1]['processing_status'], 'completed')

    def test_missing_section_keeps_primary_amplitude_null_and_partial_qc(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = prepared(root, '1-A-20240101')
            save_json(root/'input.json', dict(cases=[row]))
            def missing(*args, **kwargs):
                tracks = tracked(*args, **kwargs)
                n = tracks['corrected'].shape[1]//7
                tracks['target_valid'][:, 2*(n//2)] = False
                return tracks
            with patch('uterine_wave.io.read_video', side_effect=decoded), \
                 patch('uterine_wave.tracking.track_segments', side_effect=missing):
                result = run_manifest(root/'input.json', s_range=(.2, .8), output=root/'out', limit=1)[0]
            self.assertTrue(result['feature_success'])
            self.assertIsNone(result['anterior_radial_stretch_activity_amplitude'])
            self.assertEqual(result['anterior_amplitude_status'], 'insufficient_section_support')
            feature = json.loads((root/'out'/row['case_id']/'features.json').read_text())
            self.assertIsNotNone(feature['walls'][0]['partial_region_activity_amplitude_QC'])

    def test_disabling_speed_retains_frozen_mode_and_wall_values(self):
        data = maps()
        old = summarize_patient(data, s_range=(.2, .8))
        with patch('uterine_wave.modes.mode_velocity', side_effect=AssertionError('speed called')):
            new = summarize_patient(data, s_range=(.2, .8), include_mode_velocity=False)
        self.assertEqual(old['walls'], new['walls'])
        self.assertEqual(old['modes']['coordination'], new['modes']['coordination'])
        self.assertEqual(old['modes']['paired_ER_windows'], new['modes']['paired_ER_windows'])
        for a, b in zip(old['modes']['windows'], new['modes']['windows']):
            self.assertEqual({k: a[k] for k in b}, b)

    def test_spreadsheet_does_not_resolve_outcome_values(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'cohort.xlsx'
            xml = '''<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
            <row r="1"><c r="A1" t="inlineStr"><is><t>视频研究编号</t></is></c>
            <c r="B1" t="inlineStr"><is><t>视频完整路径</t></is></c>
            <c r="C1" t="inlineStr"><is><t>临床_临床妊娠</t></is></c></row>
            <row r="2"><c r="A2"><v>1</v></c><c r="B2" t="inlineStr"><is><t>/data/1-A-20240101/video.mp4</t></is></c>
            <c r="C2" t="s"><v>999999</v></c></row></sheetData></worksheet>'''
            with ZipFile(p, 'w') as z:
                z.writestr('xl/worksheets/sheet1.xml', xml)
            result = read_cohort(p)
            self.assertFalse(result['outcomes_read'])
            self.assertEqual(result['source_rows'], 1)
            self.assertNotIn('临床_临床妊娠', result['read_columns'])

    def test_existing_date_pinyin_annotation_matches_without_redrawing(self):
        row = dict(case_id='101-EXAMPLE-20250124', subject_name_for_matching='测试姓名')
        document = dict(shapes=[dict(label='ant_wall', points=[[1, 1], [2, 1]])])
        catalog = [dict(match_key='PATIENTS20250124EXAMPLE', document=document)]
        chosen, aliases = match_annotation(row, catalog)
        self.assertIs(chosen, catalog[0])
        self.assertEqual(aliases, 1)


if __name__ == '__main__':
    unittest.main()

