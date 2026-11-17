import unittest
import numpy as np

from uterine_wave.amplitude import band_limited_sd, band_limited_signal, summarize_interval_strain
from uterine_wave.spectrum import direction


class AmplitudeAndActivity(unittest.TestCase):
    def test_band_limited_interval_strain_sd(self):
        time_s=np.arange(500)*.12
        in_band=.02*np.sin(2*np.pi*2*time_s/60)
        out_band=.05*np.sin(2*np.pi*8*time_s/60)
        values=np.column_stack((in_band+out_band,2*in_band))
        result=summarize_interval_strain(values,time_s)
        np.testing.assert_allclose(result['band_SD_per_pair'],[.02/np.sqrt(2),.04/np.sqrt(2)],atol=1e-12)
        np.testing.assert_allclose(np.std(band_limited_signal(values,time_s),axis=0),result['band_SD_per_pair'])
        self.assertGreater(result['raw_SD_per_pair'][0],result['band_SD_per_pair'][0])
        self.assertEqual(result['positive_frequency_bins'],5)

    def test_missing_and_irregular_blocks_are_not_filled(self):
        time_s=np.arange(20)*.12;values=np.ones((20,3))
        values[4,1]=np.nan
        with self.assertRaises(ValueError):band_limited_sd(values,time_s)
        with self.assertRaises(ValueError):band_limited_sd(np.ones((20,3)),time_s+np.arange(20)**2*.001)

    def test_directional_rms_has_input_units_and_er_is_unchanged(self):
        time_s=np.arange(500)*.12;space=np.arange(20)/20
        wave=np.cos(2*np.pi*(2*time_s[:,None]/60-space[None]))
        result=direction(wave,time_s,space)
        self.assertGreater(result['ER'],.999999)
        self.assertAlmostEqual(result['directional_rms'],1/np.sqrt(2),places=12)
        self.assertAlmostEqual(result['directional_band_energy_fraction'],1,places=12)


if __name__=='__main__':unittest.main()
