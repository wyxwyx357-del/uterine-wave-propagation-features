import unittest
from analysis.mse_pregnancy_association.code.complete_record_coordination import summarize_paired_windows
from analysis.tests.fixtures import paired


class PatientAggregation(unittest.TestCase):
    def test_equal_time_is_not_equal_window_block_count(self):
        r=summarize_paired_windows([paired(30,1.,0.),paired(30,0.,0.,7,10),paired(31,1.,-1.)])
        self.assertAlmostEqual(r['record_MSE_time_equal'],(.5+4)/2)
        self.assertAlmostEqual(r['record_MSE_pooled'],5/3)

    def test_square_errors_before_spatial_average(self):
        r=summarize_paired_windows([paired(30,1.,0.),paired(30,-1.,0.,7,10)])
        self.assertEqual(r['record_MSE_time_equal'],1.)
        # Squaring the average ER difference would incorrectly return zero.
        self.assertNotEqual(r['record_MSE_time_equal'],0.)

    def test_longer_segment_contributes_more_valid_window_times(self):
        r=summarize_paired_windows([paired(20,0.,0.)]+[paired(50+i,1.,-1.) for i in range(3)])
        self.assertEqual(r['temporal_segment_count'],2)
        self.assertEqual(r['record_MSE_time_equal'],3.)
