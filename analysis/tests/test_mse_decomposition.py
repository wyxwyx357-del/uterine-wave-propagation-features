import unittest
from analysis.mse_pregnancy_association.code.complete_record_coordination import summarize_paired_windows
from analysis.tests.fixtures import paired


class MSEDecomposition(unittest.TestCase):
    def test_population_identity_and_standardization(self):
        r=summarize_paired_windows([paired(30,.2,-.1),paired(31,-.4,.2),paired(32,.7,.1),paired(33,.1,-.6)])
        self.assertAlmostEqual(r['record_MSE_time_equal'],r['record_mean_bias_sq']+r['record_sd_scale_sq']+r['record_time_shape_term'])
        self.assertAlmostEqual(r['record_MSE_z'],2*(1-r['record_CC_time_equal']))

    def test_constant_wall_does_not_get_fake_standardized_value(self):
        r=summarize_paired_windows([paired(30+i,.2,float(i)/3) for i in range(4)])
        self.assertIsNone(r['record_MSE_z']);self.assertIsNone(r['record_CC_time_equal'])

    def test_multiple_blocks_have_no_substitute_decomposition(self):
        r=summarize_paired_windows([paired(30,1.,0.),paired(30,-1.,0.,lo=7,hi=10)])
        self.assertIsNone(r['record_mean_bias_sq'])
        self.assertEqual(r['decomposition_status'],'not_defined_for_multiple_blocks_per_time')
