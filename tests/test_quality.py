"""Quality summaries must not hide gaps or treat recovery jumps as native PCC."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from uterine_wave.quality import native_pcc, tracking_quality, radial_quality_maps, candidate_quality


def fixture():
    n, k = 7, 4
    tracks = dict(time_s=np.arange(n, dtype=float), target_ncc=np.full((n, k), .9),
                  target_valid=np.ones((n, k), bool), step_valid=np.ones((n, k), bool),
                  segment_id=np.zeros((n, k), int), reanchor=np.zeros((n, k), bool),
                  pose_epoch=np.zeros(n, int), reject_reason=np.full((n, k), 'accepted', dtype='U28'))
    tracks['step_valid'][0] = False
    grid = dict(A_pair_ids=np.array([[[0, 1]], [[2, 3]]]))
    maps = dict(hop=2, s=np.array([.4]), valid_mask=np.ones((5, 2, 1), bool),
                interval_start_s=np.arange(5.), interval_end_s=np.arange(2., 7.),
                interval_center_s=np.arange(1., 6.))
    return tracks, grid, maps


class Quality(unittest.TestCase):
    def test_seed_gap_reanchor_and_epoch_are_not_native_checks(self):
        tr, grid, maps = fixture()
        tr['target_ncc'][0] = 1.  # a seed has no preceding measurement
        tr['step_valid'][2, 0] = False
        tr['target_ncc'][2, 0] = -1.  # uncomputed sentinel
        tr['reanchor'][3, 0] = True
        tr['segment_id'][3:, 0] = 1
        tr['target_ncc'][3, 0] = .999  # recovery compared nonadjacent frames
        tr['pose_epoch'][5:] = 1
        before = deepcopy(tr)
        q = native_pcc(tr)
        self.assertTrue(np.isnan(q[0]).all())
        self.assertTrue(np.isnan(q[2, 0]))
        self.assertTrue(np.isnan(q[3, 0]))
        self.assertEqual(q[4, 0], .9)  # confirmation after reanchor is native
        self.assertTrue(np.isnan(q[5]).all())
        report = tracking_quality(tr, grid)
        self.assertLess(report['all_points']['pcc_coverage_fraction'], 1.)
        self.assertIsNone(report['independent_recording_quality_pass'])
        for key in before: np.testing.assert_array_equal(tr[key], before[key])

    def test_equal_point_and_pooled_means_differ_with_missing_checks(self):
        tr, grid, maps = fixture()
        tr['target_ncc'][1:, 0] = .8
        tr['step_valid'][2:, 0] = False
        report = tracking_quality(tr, grid)['walls'][0]['radial']
        self.assertAlmostEqual(report['point_mean_average'], .85)
        self.assertAlmostEqual(report['mean'], (6*.9+.8)/7)
        self.assertAlmostEqual(report['pcc_coverage_fraction'], 7/12)
        tr['step_valid'][:] = False
        empty = tracking_quality(tr, grid)['all_points']
        self.assertIsNone(empty['mean'])
        self.assertEqual(empty['pcc_coverage_fraction'], 0.)

    def test_exact_interval_steps_both_pair_members_and_missing_coverage(self):
        tr, grid, maps = fixture()
        tr['target_ncc'][1, 0] = .8
        tr['target_ncc'][2, 1] = .76
        tr['step_valid'][3, 0] = False
        maps['valid_mask'][1, 0, 0] = False
        before = deepcopy(maps)
        q = radial_quality_maps(tr, grid, maps)
        self.assertAlmostEqual(q['interval_pcc_mean'][0, 0, 0], (.8+.9+.9+.76)/4)
        self.assertAlmostEqual(q['interval_pcc_min'][0, 0, 0], .76)
        self.assertEqual(q['interval_pcc_coverage_fraction'][1, 0, 0], .75)
        self.assertFalse(q['deformation_valid_mask'][1, 0, 0])
        np.testing.assert_allclose(q['interval_pcc_mean'][:, 1], .9)
        for key in before: np.testing.assert_array_equal(maps[key], before[key])
        maps['interval_start_s'] = maps['interval_start_s']+.5
        with self.assertRaises(ValueError): radial_quality_maps(tr, grid, maps)

    def test_candidate_qc_uses_its_wall_time_and_sections_without_confirming(self):
        tr, grid, maps = fixture()
        tr['target_ncc'][1:, 2:] = .78
        quality = radial_quality_maps(tr, grid, maps)
        event = dict(event_id='candidate_demo', sections=[0], start_s=2., end_s=3.,
                     classification='F2C', quality_flags=['single_phase_event_identity_unresolved'])
        candidates = dict(walls=[dict(wall=1, events=[event])])
        before = deepcopy(candidates)
        row = candidate_quality(quality, candidates)[0]
        self.assertAlmostEqual(row['interval_pcc']['mean'], .78)
        self.assertEqual(row['analyzed_interval_samples'], 2)
        self.assertEqual(row['source_start_s'], 1.)
        self.assertEqual(row['source_end_s'], 4.)
        self.assertFalse(row['confirmed_event'])
        self.assertAlmostEqual(row['interval_minimum_pcc']['minimum'], .78)
        self.assertEqual(candidates, before)

    def test_audit_reports_missing_and_unsupported_scores_without_modifying_inputs(self):
        from experiments.audit_tracking_quality import run
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)/'review'
            root.mkdir()
            rows = [dict(review_id='demo_missing'), dict(review_id='demo_unsupported')]
            (root/'screening_summary.json').write_text(json.dumps(rows), encoding='utf-8')
            original = {}
            for row in rows:
                folder = root/row['review_id']
                folder.mkdir()
                tr, grid, maps = fixture()
                if row['review_id'] == 'demo_missing': del tr['target_ncc']
                else: tr['step_valid'][:] = False
                for name, data in [('tracks', tr), ('grid', grid), ('maps', maps)]:
                    np.savez(folder/(name+'.npz'), **data)
                (folder/'candidates.json').write_text(json.dumps(dict(walls=[])), encoding='utf-8')
                for file in folder.iterdir(): original[file] = file.read_bytes()
            output = Path(td)/'audit'
            with redirect_stdout(io.StringIO()):
                run(SimpleNamespace(review_root=root, output=output))
            summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary[0]['status'], 'saved_native_pcc_unavailable')
            self.assertIsNone(summary[1]['radial_points']['mean'])
            self.assertEqual(summary[1]['radial_points']['pcc_coverage_fraction'], 0.)
            self.assertIn('无有效PCC', (output/'REPORT.md').read_text(encoding='utf-8'))
            for file, content in original.items(): self.assertEqual(file.read_bytes(), content)


if __name__ == '__main__': unittest.main()
