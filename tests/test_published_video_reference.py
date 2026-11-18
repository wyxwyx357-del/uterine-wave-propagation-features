"""Physical-unit invariance for the public-video timing-hypothesis adapter."""
import unittest
import numpy as np
from uterine_wave.controls import synthetic_geometry, reference_tracks
from uterine_wave.geometry import layout
from uterine_wave.deformation import compute_spacetime
from experiments.evaluate_published_video_reference import retime


class PublicVideoTiming(unittest.TestCase):
    def test_clock_conversion_preserves_strain_and_missing_segment_boundaries(self):
        grid = layout(synthetic_geometry())
        t = np.arange(100)/25
        seeds = np.vstack((grid['A_points'], grid['B_points']))
        points = seeds[None]*(1+.01*t[:, None, None])
        tracks = reference_tracks(points, t)
        tracks['step_valid'][38] = False
        tracks['target_valid'][38] = False
        tracks['segment_id'][39:] += 1
        before = t.copy()
        original = compute_spacetime(tracks, grid, interval_s=.48)
        converted = compute_spacetime(retime(tracks, 4.), grid, interval_s=1.92)
        self.assertEqual(original['hop'], converted['hop'])
        np.testing.assert_allclose(converted['epsilon'], original['epsilon'], equal_nan=True)
        np.testing.assert_allclose(converted['A'], original['A']/4, equal_nan=True)
        np.testing.assert_array_equal(converted['valid_mask'], original['valid_mask'])
        self.assertFalse(converted['valid_mask'][26:39].any())
        np.testing.assert_array_equal(tracks['time_s'], before)


if __name__ == '__main__':
    unittest.main()
