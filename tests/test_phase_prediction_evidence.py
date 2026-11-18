import unittest
import numpy as np
from experiments.phase_prediction_evidence import predict_phase


class PhasePredictionEvidence(unittest.TestCase):
    def example(self, middle_shift=0., amplitude=.01):
        t = np.arange(601)*.1
        s = np.array([.2, .4, .6])
        peaks = np.array([20., 23.+middle_shift, 26.])
        y = amplitude*np.exp(-.5*((t[:, None]-peaks)/2)**2)
        nodes = {j:(int(np.rint((p-6)/.1)), int(np.rint((p+6)/.1))+1, 0, len(t))
                 for j, p in enumerate(peaks)}
        return y, t, s, nodes, peaks

    def test_same_delayed_shape_predicts_held_out_position_at_low_amplitude(self):
        for amplitude in (.01, .001):
            y, t, s, nodes, peaks = self.example(amplitude=amplitude)
            result = predict_phase(y, t, s, [0, 1, 2], nodes, peaks)
            self.assertLess(result['arrival_loo_RMSE_s'], 1e-12)
            self.assertGreater(result['predicted_shape_cosine_mean'], .999)

    def test_peak_alignment_does_not_hide_a_bad_arrival_sequence(self):
        y, t, s, nodes, peaks = self.example(middle_shift=-4.)
        result = predict_phase(y, t, s, [0, 1, 2], nodes, peaks)
        self.assertGreater(result['peak_aligned_shape_cosine_mean'], .999)
        self.assertGreater(result['arrival_loo_RMSE_s'], 3.)
        self.assertLess(result['predicted_shape_cosine_mean'], .5)

    def test_prediction_stays_inside_held_out_run_and_reports_clipped_support(self):
        y, t, s, nodes, peaks = self.example()
        # The observed phase must be inside its finite run as real recovered nodes are.
        nodes[1] = (190, 265, 190, 265)
        result = predict_phase(y, t, s, [0, 1, 2], nodes, peaks)
        held = result['positions'][1]
        self.assertLess(held['retained_prediction_support_fraction'], 1.)
        self.assertTrue(held['prediction_inside_continuous_run'])
        self.assertLessEqual(abs(held['rounding_error_s']), .05)


if __name__ == '__main__':
    unittest.main()
