import unittest
import numpy as np
from experiments.test_weak_wave_signals import radial_fixture, add_to_fixed_noise, match_known_structure


class WeakWaveSignals(unittest.TestCase):
    def test_matching_known_full_path_does_not_favor_tiny_incidental_fragment(self):
        short = dict(retained_path_fraction=.08, arrival_RMSE_to_injection_s=.05)
        full = dict(retained_path_fraction=.9, arrival_RMSE_to_injection_s=.4)
        self.assertIs(match_known_structure([short, full]), full)

    def test_amplitude_reduction_keeps_identical_noise(self):
        t = np.arange(1., 59., .12)
        s = np.linspace(.1, .9, 17)
        noise = np.random.default_rng(12).normal(0, .0005, (len(t), 2, len(s)))
        for stretch in (.04, .02, .01):
            signal, _, _ = radial_fixture(t, s, 2.04, 'C2F', stretch)
            mixed = add_to_fixed_noise(noise, signal[:, None])
            np.testing.assert_allclose(mixed-signal[:, None], noise, atol=1e-15, rtol=0)

    def test_local_activity_is_synchronous_and_preserves_inactive_positions(self):
        t = np.arange(1., 59., .12)
        s = np.linspace(.1, .9, 17)
        signal, arrival, active = radial_fixture(t, s, 2.04, 'local_synchronous')
        self.assertEqual(int(active.sum()), 5)
        self.assertTrue(np.all(arrival == 30))
        self.assertTrue(np.all(signal[:, active == 0] == 0))
        np.testing.assert_allclose(signal[:, active > 0], np.repeat(signal[:, active > 0][:, :1], 5, axis=1))

    def test_positive_and_negative_wave_paths_retain_known_arrival_order(self):
        t = np.arange(1., 59., .12)
        s = np.linspace(.1, .9, 17)
        for condition, sign in [('C2F', 1), ('F2C', -1)]:
            signal, arrival, active = radial_fixture(t, s, 2.04, condition)
            self.assertTrue(np.all(np.diff(arrival)*sign > 0))
            self.assertTrue(np.isfinite(signal).all())
            self.assertTrue(np.all(active == 1))


if __name__ == '__main__':
    unittest.main()
