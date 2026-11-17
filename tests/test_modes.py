import unittest
import numpy as np
from uterine_wave.spectrum import mode_velocity, direction
from uterine_wave.modes import analyze_modes


class Modes(unittest.TestCase):
    def test_known_signed_velocity_and_legacy_ER(self):
        t=np.arange(500)*.12;s=np.arange(21)/21
        for sign in (1,-1):
            wave=np.sin(2*np.pi*(2*t[:,None]/60-sign*s[None]))
            result=mode_velocity(wave,t,s)
            self.assertEqual(result['ER'],direction(wave,t,s)['ER'])
            direction_name='C2F' if sign==1 else 'F2C'
            self.assertAlmostEqual(result[direction_name]['velocity'],sign*2/60,places=12)
            self.assertIsNone(result['event_detected'])
        self.assertIsNone(mode_velocity(np.zeros((500,21)),t,s)['C2F']['velocity'])

    def test_no_gap_splicing_and_paired_same_region(self):
        t=np.arange(500)*.12;s=np.arange(9)/9
        wave=np.sin(2*np.pi*(2*t[:,None]/60-s[None]))
        values=np.stack((wave,wave),axis=1);mask=np.ones_like(values,bool)
        mask[:,0,3:]=False;mask[:,1,:3]=False
        result=analyze_modes(dict(A=values,valid_mask=mask,interval_center_s=t,s=s),refine_velocity=False)
        self.assertTrue(result['windows']);self.assertFalse(result['paired_ER_windows'])
        self.assertFalse(result['coordination'])
        mask[:]=True;mask[200]=False
        result=analyze_modes(dict(A=values,valid_mask=mask,interval_center_s=t,s=s),refine_velocity=False)
        gap=t[200]
        self.assertTrue(all(not(r['window_start_s']<=gap<=r['window_end_s']) for r in result['windows']))

    def test_same_mode_mse_and_coarse_short_window_resolution(self):
        t=np.arange(500)*.12;s=np.arange(9)/9
        wave=np.sin(2*np.pi*(2*t[:,None]/60-s[None]))
        values=np.stack((wave,wave),axis=1)
        result=analyze_modes(dict(A=values,valid_mask=np.ones_like(values,bool),interval_center_s=t,s=s),refine_velocity=False)
        self.assertEqual(result['coordination'][0]['ER_MSE'],0)
        self.assertIsNone(result['event_count'])
        self.assertEqual(result['windows'][0]['positive_temporal_band_bins'],1)
        self.assertEqual(result['windows'][0]['C2F']['available_spectral_cells'],4)
        self.assertTrue(result['windows'][0]['C2F']['lowest_nonzero_spatial_peak'])
        self.assertAlmostEqual(result['windows'][0]['time_resolution_cpm'],60/(round(20/.12)*.12))


if __name__=='__main__':unittest.main()
