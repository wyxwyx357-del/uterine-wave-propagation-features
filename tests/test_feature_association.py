import unittest
import numpy as np
from experiments.feature_association import bh_adjust, fit_cluster_logit


class AssociationTests(unittest.TestCase):
    def test_logit_recovers_exact_two_by_two_odds_ratio(self):
        # x=0: odds=20/80; x=1: odds=40/60; known OR=8/3.
        x = np.r_[np.zeros(100), np.ones(100)]
        y = np.r_[np.ones(20), np.zeros(80), np.ones(40), np.zeros(60)]
        design = np.column_stack([np.ones(200), x])
        result = fit_cluster_logit(design, y, np.arange(200))
        self.assertAlmostEqual(np.exp(result['beta'][1]), 8/3, places=7)
        correction = 200/199*199/198
        expected_variance = (1/20+1/80+1/40+1/60)*correction
        self.assertAlmostEqual(result['covariance'][1, 1], expected_variance, places=8)

    def test_repeating_patient_records_does_not_double_precision(self):
        x = np.r_[np.zeros(100), np.ones(100)]
        y = np.r_[np.ones(20), np.zeros(80), np.ones(40), np.zeros(60)]
        design = np.column_stack([np.ones(200), x])
        original = fit_cluster_logit(design, y, np.arange(200))
        repeated = fit_cluster_logit(np.repeat(design, 2, axis=0),
            np.repeat(y, 2), np.repeat(np.arange(200), 2))
        np.testing.assert_allclose(original['beta'], repeated['beta'], atol=1e-8)
        left = original['covariance']/original['covariance_CR1_factor']
        right = repeated['covariance']/repeated['covariance_CR1_factor']
        np.testing.assert_allclose(left, right, atol=1e-8)

    def test_BH_matches_known_order_independent_values(self):
        np.testing.assert_allclose(bh_adjust([.04, .001, .03]), [.04, .003, .04])


if __name__ == '__main__':
    unittest.main()
