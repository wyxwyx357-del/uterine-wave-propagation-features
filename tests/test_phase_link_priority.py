import unittest
import numpy as np
from experiments.phase_link_priority import detect_priority, phase_similarity
from uterine_wave import radial_events


class PhaseLinkPriority(unittest.TestCase):
    def test_full_phase_similarity_penalizes_unmatched_tails_not_low_amplitude(self):
        y = np.zeros((25, 3))
        y[9:12, 0] = [1, 2, 1]
        y[9:12, 1] = [1, 2, 1]
        y[5:16, 1] += .5
        y[:, 2] = .1*y[:, 0]
        a = dict(first=9, last=11, peak_index=10, column=0)
        b = dict(first=5, last=15, peak_index=10, column=1)
        c = dict(first=9, last=11, peak_index=10, column=2)
        self.assertLess(phase_similarity(a, b, y), phase_similarity(a, c, y))
        self.assertAlmostEqual(phase_similarity(a, c, y), 1.)

    def test_known_propagation_at_two_amplitudes_and_both_directions_survives(self):
        t = np.arange(751)*.12
        s = np.linspace(.2, .8, 17)
        original_builder = radial_events._ridges
        for direction in (1, -1):
            for scale in (1., .5):
                x = t[:, None]-(25+direction*(s-.5)/.04)[None]
                y = scale*(-.03*x/9)*np.exp(-.5*(x/3)**2)
                before = y.copy()
                r = detect_priority(y, np.ones_like(y, bool), t, s, noise_floor=.001)
                self.assertEqual(r['traveling_candidate_count'], 1)
                self.assertEqual(r['events'][0]['classification'], 'C2F' if direction == 1 else 'F2C')
                self.assertAlmostEqual(r['events'][0]['velocity_paths_per_s'], direction*.04, places=5)
                self.assertIsNone(r['event_count'])
                self.assertIs(radial_events._ridges, original_builder)
                np.testing.assert_array_equal(y, before)

    def test_missing_time_and_space_are_not_joined(self):
        t = np.arange(751)*.12
        s = np.linspace(.2, .8, 17)
        x = t[:, None]-(25+(s-.5)/.04)[None]
        y = (-.03*x/9)*np.exp(-.5*(x/3)**2)
        valid = np.ones_like(y, bool)
        gap = (t >= 22)&(t <= 27)
        valid[gap] = False; valid[:, 8] = False
        r = detect_priority(y, valid, t, s, noise_floor=.001)
        self.assertTrue(np.isnan(r['filtered_A'][gap]).all())
        for e in r['events']:
            self.assertFalse(e['start_s'] < 22 and e['end_s'] > 27)
            self.assertFalse(min(e['sections']) < 8 < max(e['sections']))


if __name__ == '__main__':
    unittest.main()
