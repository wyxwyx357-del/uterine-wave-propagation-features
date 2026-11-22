import unittest
import numpy as np
from analysis.mse_pregnancy_association.code.measurement.deformation import interval_support
from analysis.mse_pregnancy_association.code.measurement.modes import analyze_modes


class NoCrossSegment(unittest.TestCase):
    def test_reanchor_and_epoch_do_not_get_differentiated_across(self):
        t=np.arange(9,dtype=float)
        track=dict(time_s=t,target_valid=np.ones((9,1),bool),step_valid=np.ones((9,1),bool),
            segment_id=np.zeros((9,1),int),pose_epoch=np.zeros(9,int))
        track['segment_id'][4:]=1
        supported=interval_support(track,2)[:,0]
        self.assertFalse(supported[2]);self.assertFalse(supported[3]);self.assertTrue(supported[4])
        track['segment_id'][:]=0;track['pose_epoch'][4:]=1
        np.testing.assert_array_equal(interval_support(track,2)[:,0],supported)

    def test_FFT_window_never_crosses_missing_time_support(self):
        t=np.arange(0,70,.25);s=np.linspace(.2,.8,7)
        field=np.sin(2*np.pi*t[:,None]/20-5*s[None,:])
        a=np.stack([field,field],axis=1);valid=np.ones_like(a,dtype=bool)
        valid[(t>=25)&(t<30)]=False
        r=analyze_modes(dict(A=a,valid_mask=valid,interval_center_s=t,s=s),include_velocity=False)
        self.assertGreater(len(r['paired_ER_windows']),0)
        for row in r['paired_ER_windows']:
            self.assertFalse(row['window_start_s']<30 and row['window_end_s']>=25)
