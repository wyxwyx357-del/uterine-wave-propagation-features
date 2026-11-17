import unittest
import numpy as np
from uterine_wave.spectrum import mode_velocity


class ContinuousSpeed(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(167)*.12
        self.s = np.arange(17)/17

    def wave(self, cpm, wavenumber):
        return np.sin(2*np.pi*(cpm*self.t[:, None]/60+wavenumber*self.s[None, :]))

    def test_short_window_off_grid_speeds_and_unchanged_ER(self):
        for cpm, k in [(2., -1.), (4., -1.), (2.3, 1.4)]:
            with self.subTest(cpm=cpm, k=k):
                wave = self.wave(cpm, k)
                coarse = mode_velocity(wave, self.t, self.s, refine=False)
                result = mode_velocity(wave, self.t, self.s, refine=True)
                direction = 'C2F' if k < 0 else 'F2C'
                self.assertAlmostEqual(result[direction]['velocity'], -cpm/60/k, places=6)
                self.assertEqual(result['ER'], coarse['ER'])
                self.assertEqual(result[direction]['grid_peak']['velocity'], coarse[direction]['velocity'])
                self.assertIsNone(result['event_detected'])

    def test_counterpropagating_waves_with_noise_and_linear_drift(self):
        wave = self.wave(2.3, -1.4)+.7*self.wave(3.7, .9)
        wave += .1*np.random.default_rng(418).normal(size=wave.shape)
        wave += .08*self.t[:, None]*(1+self.s[None, :])
        result = mode_velocity(wave, self.t, self.s, refine=True)
        for name, truth in [('C2F', 2.3/60/1.4), ('F2C', -3.7/60/.9)]:
            self.assertLess(abs(result[name]['velocity']/truth-1), .02)

    def test_drift_does_not_get_resolved_propagation_speed(self):
        for wave in [self.t[:, None]*(1+self.s[None, :]),
                     self.t[:, None]**2*np.ones_like(self.s[None, :])]:
            result = mode_velocity(wave, self.t, self.s, refine=True)
            self.assertIsNone(result['C2F']['velocity'])
            self.assertIsNone(result['F2C']['velocity'])

    def test_translating_nonperiodic_pulse(self):
        truth = .05
        wave = np.exp(-((self.t[:, None]-5-self.s[None, :]/truth)/2)**2)
        result = mode_velocity(wave, self.t, self.s, refine=True)
        self.assertLess(abs(result['C2F']['velocity']/truth-1), .05)
        self.assertIsNone(result['event_detected'])
        self.assertGreater(result['velocity_fit']['residual_fraction'], .1)


if __name__ == '__main__':
    unittest.main()
