import unittest
import numpy as np
from experiments.batch_wave_counts import count_map, COUNT, RATE


class WaveBatchTests(unittest.TestCase):
    def maps(self, supported):
        t = np.arange(121)*.25+1.
        return dict(A=np.zeros((121, 2, 4)),
                    valid_mask=np.full((121, 2, 4), supported, bool),
                    s=np.linspace(.2, .8, 4), interval_center_s=t,
                    interval_start_s=t-1., interval_end_s=t+1.,
                    interval_dt_s=np.full(121, 2.))

    def test_missing_support_is_not_zero_activity(self):
        result = count_map(self.maps(False), .0007509593468916695)
        self.assertIsNone(result[COUNT])
        self.assertIsNone(result[RATE])
        self.assertEqual(result['status'], 'no_analyzable_event_support')

    def test_observable_constant_map_has_zero_detected_activities(self):
        result = count_map(self.maps(True), .0007509593468916695)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result[COUNT], 0)
        self.assertEqual(result[RATE], 0.)
        self.assertEqual(result['effective_observation_seconds'], 32.)


if __name__ == '__main__':
    unittest.main()
