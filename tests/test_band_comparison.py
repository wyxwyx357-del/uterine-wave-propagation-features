"""Physical and interface checks for review-only tracked band curvature."""
import unittest
import numpy as np
from uterine_wave.geometry import layout
from experiments.compare_midline_radial import band_maps


class BandComparison(unittest.TestCase):
    def setUp(self):
        self.grid=layout(dict(ant_wall=np.array([[20.,80.],[220.,80.]]),
            post_wall=np.array([[20.,120.],[220.,120.]]),
            cervix_point=np.array([[20.,100.]]),fundus_point=np.array([[220.,100.]])))
        self.t=np.arange(901)*.1;self.hop=20
        self.initial=np.vstack((self.grid['A_points'],self.grid['B_points']))
        n=len(self.initial)
        self.tracks=dict(time_s=self.t,corrected=np.broadcast_to(self.initial,(len(self.t),n,2)).copy(),
            target_valid=np.ones((len(self.t),n),bool),step_valid=np.ones((len(self.t),n),bool),
            segment_id=np.zeros((len(self.t),n),int),pose_epoch=np.zeros(len(self.t),int))
        self.maps=dict(hop=self.hop,s=self.grid['u'],
            interval_center_s=(self.t[self.hop:]+self.t[:-self.hop])/2)

    def test_rigid_motion_does_not_create_bending(self):
        angle=.1*np.sin(self.t/9);c=np.cos(angle);s=np.sin(angle)
        rotation=np.stack((c,-s,s,c),axis=1).reshape(-1,2,2)
        self.tracks['corrected']=np.einsum('tij,nj->tni',rotation,self.initial)
        self.tracks['corrected']+=np.stack((3*np.sin(self.t/7),4*np.cos(self.t/8)),axis=1)[:,None,:]
        result=band_maps(self.tracks,self.grid,self.maps)
        self.assertTrue(result['valid_mask'].all())
        self.assertLess(np.nanmax(abs(result['B_rate'])),1e-12)

    def test_known_bending_keeps_native_interval_and_original_space(self):
        offset=len(self.grid['A_points']);ids=self.grid['B_curve_ids']+offset
        x=self.grid['u'][None,:]
        for curve in ids:
            self.tracks['corrected'][:,curve,1]+=3*np.sin(2*np.pi*(self.t[:,None]/30-x))
        before=self.tracks['corrected'].copy()
        result=band_maps(self.tracks,self.grid,self.maps)
        self.assertGreater(np.max(abs(result['B_rate'])),1e-6)
        np.testing.assert_array_equal(result['s'],self.grid['u'][2:-2])
        self.assertGreater(result['s'][0],0);self.assertLess(result['s'][-1],1)
        np.testing.assert_allclose(result['interval_dt_s'],2.)
        np.testing.assert_array_equal(result['interval_center_s'],self.maps['interval_center_s'])
        np.testing.assert_array_equal(before,self.tracks['corrected'])
        self.assertIsNone(result['event_count'])

    def test_local_missing_support_is_not_filled_or_spread_to_other_bands(self):
        col=8;point=self.grid['B_curve_ids'][1,col]+len(self.grid['A_points'])
        self.tracks['target_valid'][200,point]=False
        result=band_maps(self.tracks,self.grid,self.maps)
        crossing=(self.t[:-20]<=20)&(self.t[20:]>=20)
        self.assertTrue(np.isnan(result['B_rate'][crossing,1,col-2]).all())
        self.assertTrue(np.isnan(result['B_band_rate'][crossing,1,col-2]).all())
        self.assertTrue(result['valid_mask'][:,0].all())
        self.assertTrue(result['valid_mask'][:,2].all())
        self.assertTrue(self.tracks['target_valid'][199,point])


if __name__=='__main__':unittest.main()
