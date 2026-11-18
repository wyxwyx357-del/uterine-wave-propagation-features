import unittest
import numpy as np
from uterine_wave.motion_consensus import rigid_consensus


class MotionConsensus(unittest.TestCase):
    def setUp(self):
        self.p = np.random.default_rng(30).uniform(-70, 70, (64, 2))
        angle = .04
        self.r = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
        self.q = self.p @ self.r.T + [3., -2.]

    def test_recovers_majority_without_following_large_local_motion(self):
        q = self.q.copy()
        q[:20] += np.random.default_rng(9).normal(0, 20, (20, 2))
        result = rigid_consensus(self.p, q)
        self.assertTrue(result['majority_found'])
        self.assertEqual(result['inlier_points'], 44)
        np.testing.assert_allclose(result['matrix'][:2, :2], self.r, atol=1e-12)
        np.testing.assert_allclose(result['matrix'][:2, 2], [3, -2], atol=1e-12)

    def test_equal_competing_regions_do_not_claim_majority(self):
        q = self.p.copy()
        q[:32] += [8, 0]
        q[32:] += [-8, 0]
        result = rigid_consensus(self.p, q, tolerance_px=.1)
        self.assertFalse(result['majority_found'])
        self.assertEqual(result['inlier_points'], 32)

    def test_rigid_correction_preserves_pair_lengths(self):
        result = rigid_consensus(self.p, self.q)
        m = result['matrix']
        corrected = (self.q-m[:2, 2]) @ m[:2, :2]
        np.testing.assert_allclose(np.linalg.norm(corrected[1::2]-corrected[::2], axis=1),
                                   np.linalg.norm(self.q[1::2]-self.q[::2], axis=1), atol=1e-12)

    def test_too_few_points_stays_unresolved(self):
        result = rigid_consensus(self.p[:2], self.q[:2])
        self.assertFalse(result['majority_found'])
        self.assertTrue(np.isnan(result['matrix']).all())


if __name__ == '__main__':
    unittest.main()
