import unittest
import numpy as np
from analysis.mse_pregnancy_association.code.complete_record_coordination import summarize_paired_windows
from analysis.tests.fixtures import paired


class MSEFormula(unittest.TestCase):
    def test_exact_squared_difference_and_missing(self):
        result=summarize_paired_windows([paired(30,.2,.1),paired(31,-.1,.1),paired(32,.4,0.)])
        self.assertAlmostEqual(result['record_MSE_time_equal'],(.01+.04+.16)/3)
        self.assertIsNone(summarize_paired_windows([])['record_MSE_time_equal'])

    def test_equal_walls_and_extreme_ER(self):
        self.assertEqual(summarize_paired_windows([paired(30,-1.,-1.)])['record_MSE_time_equal'],0.)
        self.assertEqual(summarize_paired_windows([paired(30,-1.,1.)])['record_MSE_time_equal'],4.)

    def test_nonfinite_is_rejected_not_silently_zeroed(self):
        with self.assertRaises(ValueError):summarize_paired_windows([paired(30,np.nan,1.)])
