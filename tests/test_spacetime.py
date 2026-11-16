import unittest
import numpy as np
from uterine_wave.controls import synthetic_geometry, reference_tracks
from uterine_wave.geometry import layout
from uterine_wave.deformation import compute_spacetime, interval_support


class Spacetime(unittest.TestCase):
    def test_physical_interval_at_31fps_and_known_extension(self):
        grid=layout(synthetic_geometry());t=np.arange(80)/31
        seeds=np.vstack((grid['A_points'],grid['B_points']))
        scale=1+.02*t
        points=seeds[None]*scale[:,None,None]
        result=compute_spacetime(reference_tracks(points,t),grid)
        self.assertEqual(result['hop'],30)
        np.testing.assert_allclose(result['interval_dt_s'],30/31)
        np.testing.assert_allclose(result['interval_center_s'],(t[:-30]+t[30:])/2)
        expected=.02*(30/31)/(1+.02*t[:-30])
        np.testing.assert_allclose(result['epsilon'],np.broadcast_to(expected[:,None,None],result['epsilon'].shape),atol=1e-13)
        np.testing.assert_array_equal(result['s'],grid['u'])
        self.assertGreater(result['s'][0],0)
        self.assertLess(result['s'][-1],1)

    def test_mid_interval_pose_failure_invalidates_without_mutating_tracks(self):
        grid=layout(synthetic_geometry());t=np.arange(45)*.12
        p=np.repeat(np.vstack((grid['A_points'],grid['B_points']))[None],len(t),axis=0)
        tracks=reference_tracks(p,t);before=tracks['target_valid'].copy()
        tracks['pose_valid']=np.ones(len(t),bool);tracks['pose_valid'][12]=False
        result=compute_spacetime(tracks,grid)
        self.assertFalse(result['valid_mask'][4:13].any())
        self.assertTrue(np.isnan(result['A'][4:13]).all())
        self.assertTrue(result['valid_mask'][:4].all())
        self.assertTrue(result['valid_mask'][13:].all())
        np.testing.assert_array_equal(tracks['target_valid'],before)
        tracks['pose_valid']=np.ones(2,bool)
        with self.assertRaises(ValueError):interval_support(tracks,8)


if __name__=='__main__':unittest.main()
