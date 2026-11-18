import unittest
from functools import partial
from unittest.mock import patch
import numpy as np
from experiments.event_input_smoothing import measurement_span_mean
from experiments.partial_phase_events import detect_partial
from experiments.test_double_event_signals import double_fixture,raw_truth
from uterine_wave import radial_events
from uterine_wave.amplitude import band_limited_signal


class EventInputSmoothingTests(unittest.TestCase):
    def test_symmetric_local_average_does_not_delay_interior_activity(self):
        t=np.arange(0.,60.,.12)
        y=np.exp(-.5*((t-30.)/4.)**2)
        z=measurement_span_mean(y,t,span_s=2.04)
        self.assertEqual(z.shape,y.shape)
        self.assertEqual(np.argmax(z),np.argmax(y))

    def test_mean_removal_and_trailing_spatial_dimensions(self):
        t=np.arange(0.,20.,.12)
        y=np.sin(t[:,None,None])*np.array([[[1.,2.],[3.,4.]]])
        f=partial(measurement_span_mean,span_s=2.04)
        np.testing.assert_allclose(f(y+7.,t),f(y,t),atol=1e-13,rtol=0)
        self.assertEqual(f(y,t).shape,y.shape)

    def test_missing_run_does_not_connect_filter_context(self):
        t=np.arange(0.,60.,.12);s=np.array([.1,.5,.9])
        y=np.sin(t[:,None]/4)*np.ones((1,3));valid=np.ones_like(y,dtype=bool)
        gap=(t>=20)&(t<23);valid[gap]=False
        changed=y.copy();changed[t<20]*=10.
        function=partial(measurement_span_mean,span_s=2.04)
        with patch.object(radial_events,'band_limited_signal',function):
            before=detect_partial(y,valid,t,s,noise_floor=0.)['filtered_A']
            after=detect_partial(changed,valid,t,s,noise_floor=0.)['filtered_A']
        self.assertTrue(np.isnan(before[gap]).all())
        np.testing.assert_array_equal(before[t>=23],after[t>=23])

    def test_known_distinguishing_phases_are_not_erased(self):
        t=np.arange(1.02,59.,.12);s=np.linspace(.1,.9,9)
        y,_=double_fixture(t,s,2.04,'C2F',10.,.04)
        truth=raw_truth(y,t);col=4
        periodic=band_limited_signal(y,t)
        local=measurement_span_mean(y,t,span_s=2.04)
        for identity,prefix,sign in [(0,'negative',-1),(1,'positive',1)]:
            a,b=truth['columns'][col][identity][prefix+'_phase_indices']
            self.assertGreater(np.max(sign*local[a:b,col]),np.max(sign*periodic[a:b,col]))


if __name__=='__main__':
    unittest.main()
