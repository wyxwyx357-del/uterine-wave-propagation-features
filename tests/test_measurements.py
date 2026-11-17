import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from uterine_wave.geometry import CONFIG, layout, read_labelme
from uterine_wave.controls import synthetic_geometry, reference_tracks
from uterine_wave.deformation import curvature, compute_maps, interval_support, rate_A, rate_B
from uterine_wave.spectrum import direction
from uterine_wave.cli import main


class Measurements(unittest.TestCase):
    def setUp(self):
        self.shapes = synthetic_geometry()
        self.grid = layout(self.shapes)
        self.t = np.arange(41)*.12
        self.p = np.vstack((self.grid['A_points'], self.grid['B_points']))
        self.truth = np.repeat(self.p[None], len(self.t), axis=0)

    def test_topology_reversal_and_baseline(self):
        reversed_shapes = {k: (v[::-1] if k.endswith('wall') else v) for k, v in self.shapes.items()}
        backwards = layout(reversed_shapes)
        for key in ('A_points', 'A_pair_ids', 'B_points', 'B_curve_ids'):
            np.testing.assert_allclose(self.grid[key], backwards[key], atol=1e-12)
        radial = self.grid['radial']
        np.testing.assert_allclose(np.linalg.norm(radial[:, :, 1]-radial[:, :, 0], axis=-1), 21.)
        self.assertLess(self.grid['tangent_normal_error'], 1e-12)
        self.assertEqual(np.unique(self.grid['A_pair_ids']).size, len(self.grid['A_points']))

    def test_rigid_geometry_equivariance(self):
        angle = .7
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        move = np.array([15., -20.])
        transformed = layout({k: v@rotation.T+move for k, v in self.shapes.items()})
        for key in ('A_points', 'B_points'):
            np.testing.assert_allclose(transformed[key], self.grid[key]@rotation.T+move, atol=1e-10)

    def test_static_maps_are_zero(self):
        result = compute_maps(reference_tracks(self.truth, self.t), self.grid)
        self.assertTrue(result['A_valid'].all())
        self.assertTrue(result['B_valid'].all())
        np.testing.assert_allclose(result['A'], 0, atol=1e-14)
        np.testing.assert_allclose(result['B'], 0, atol=1e-14)

    def test_known_isotropic_extension_and_nonuniform_time(self):
        # Independently known length scale; no tracking-derived reference.
        t = self.t + np.arange(len(self.t))**2*.0001
        scale = 1 + .02*t
        p = self.truth*scale[:, None, None]
        a = rate_A(p[:, :len(self.grid['A_points'])], self.grid, t)
        expected = ((scale[8:]-scale[:-8])/scale[:-8]/(t[8:]-t[:-8]))[:, None, None]
        np.testing.assert_allclose(a, np.broadcast_to(expected, a.shape), atol=1e-13)

    def test_curvature_rigid_invariance_and_scaling(self):
        arc = self.grid['B_arc_positions'][0]
        ids = self.grid['B_curve_ids'][0]
        points = self.grid['B_points'][ids][None]
        angle = .4
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        k = curvature(points, arc)
        np.testing.assert_allclose(curvature(points@rotation.T+[5., 8.], arc), k, atol=1e-14)
        np.testing.assert_allclose(curvature(points*1.2, arc), k/1.2, atol=1e-14)

    def test_gap_invalidates_only_affected_intervals(self):
        track = reference_tracks(self.truth, self.t)
        track['step_valid'][12, 0] = False
        support = interval_support(track, 8)
        expected = np.ones(len(self.t)-8, bool)
        expected[4:12] = False  # interval end frames 12..19 cross step 12
        np.testing.assert_array_equal(support[:, 0], expected)
        self.assertTrue(support[:, 1:].all())

    def test_segment_and_pose_epoch_boundaries(self):
        track = reference_tracks(self.truth, self.t)
        track['segment_id'][12:, 0] = 1
        support = interval_support(track, 8)
        self.assertFalse(support[4:12, 0].any())
        self.assertTrue(support[12:, 0].all())
        track['pose_epoch'] = np.r_[np.zeros(20, int), np.ones(len(self.t)-20, int)]
        support = interval_support(track, 8)
        self.assertFalse(support[12:20].any())

    def test_missing_neighbor_invalidates_curvature(self):
        track = reference_tracks(self.truth.copy(), self.t)
        offset = len(self.grid['A_points'])
        point = offset + self.grid['B_curve_ids'][0, 4]
        track['target_valid'][12, point] = False
        track['corrected'][12, point] = np.nan
        maps = compute_maps(track, self.grid)
        self.assertTrue(maps['A_valid'].all())
        self.assertFalse(maps['B_valid'][4, 0, :5].any())
        self.assertTrue(np.isnan(maps['B'][4, 0, :5]).all())

    def test_reject_invalid_time_and_point_count(self):
        track = reference_tracks(self.truth, self.t)
        track['time_s'] = self.t*0
        with self.assertRaises(ValueError):
            compute_maps(track, self.grid)
        track = reference_tracks(self.truth[:, :-1], self.t)
        with self.assertRaises(ValueError):
            compute_maps(track, self.grid)

    def test_interior_nan_is_missing_even_if_legacy_flag_is_true(self):
        track = reference_tracks(self.truth.copy(), self.t)
        track['corrected'][12, 0] = np.nan
        maps = compute_maps(track, self.grid)
        self.assertFalse(maps['A_valid'][4:13, 0, 0].any())
        self.assertTrue(maps['A_valid'][:, 1].all())

    def test_direction_forward_reverse_static_and_standing(self):
        t = np.arange(500)*.12
        s = np.arange(32)*10.5
        phase_t = 2*np.pi*2*t[:, None]/60
        phase_s = 2*np.pi*np.arange(32)[None, :]/32
        forward = direction(np.sin(phase_t-phase_s), t, s)
        reverse = direction(np.sin(phase_t+phase_s), t, s)
        self.assertGreater(forward['ER'], .99)
        self.assertLess(reverse['ER'], -.99)
        self.assertAlmostEqual(direction(np.sin(phase_t)*np.cos(phase_s), t, s)['ER'], 0, places=12)
        self.assertIsNone(direction(np.zeros((len(t), len(s))), t, s)['ER'])
        self.assertIsNone(forward['event_detected'])

    def test_fft_rejects_irregular_time(self):
        with self.assertRaises(ValueError):
            direction(np.ones((4, 3)), [0, 1, 2.1, 3], [0, 1, 2])

    def test_band_arcs_are_actual_distances(self):
        for curve, arc in zip(self.grid['band'], self.grid['B_arc_positions']):
            np.testing.assert_allclose(np.diff(arc), np.linalg.norm(np.diff(curve, axis=0), axis=1))

    def test_labelme_and_cli_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = {'shapes': [dict(label=k, points=(v/.5).tolist()) for k, v in self.shapes.items()]}
            annotation = root/'first.json'
            annotation.write_text(json.dumps(labels), encoding='utf-8')
            grid_path, track_path, out = root/'grid.npz', root/'tracks.npz', root/'maps.npz'
            main(['layout', '--annotation', str(annotation), '--output', str(grid_path)])
            np.savez(track_path, **reference_tracks(self.truth, self.t))
            main(['features', '--layout', str(grid_path), '--tracks', str(track_path), '--hop', '8', '--output', str(out)])
            with np.load(out, allow_pickle=False) as maps:
                self.assertTrue(maps['A_valid'].all())
                self.assertTrue(maps['B_valid'].all())


if __name__ == '__main__':
    unittest.main()
