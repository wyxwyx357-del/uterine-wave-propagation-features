import unittest
import numpy as np
from analysis.mse_pregnancy_association.code.measurement.controls import synthetic_geometry,reference_tracks
from analysis.mse_pregnancy_association.code.measurement.geometry import layout
from analysis.mse_pregnancy_association.code.replay_measurement import replay


class SavedTracksChain(unittest.TestCase):
    def test_analytic_radial_tracks_reach_ER_and_record_MSE(self):
        grid=layout(synthetic_geometry());t=np.arange(0,70,.25)
        seed=np.vstack((grid['A_points'],grid['B_points']))
        points=np.broadcast_to(seed,(len(t),)+seed.shape).copy()
        for wall in range(2):
            for section,(inner,outer) in enumerate(grid['A_pair_ids'][wall]):
                phase=2*np.pi*t/20+(1 if wall==0 else -1)*6*grid['u'][section]
                scale=1+.02*np.sin(phase)
                points[:,outer]=seed[inner]+(seed[outer]-seed[inner])*scale[:,None]
        result=replay(reference_tracks(points,t),grid)
        self.assertEqual(result['nominal_interval_s'],2.)
        self.assertGreater(result['record']['paired_window_count'],0)
        self.assertTrue(np.isfinite(result['record']['record_MSE_time_equal']))
        self.assertGreater(result['record']['record_MSE_time_equal'],.1)
