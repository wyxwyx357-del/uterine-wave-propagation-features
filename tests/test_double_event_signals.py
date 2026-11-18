import unittest
import numpy as np
from experiments.test_double_event_signals import double_fixture, raw_truth, one_to_one, phase_identity
from experiments.audit_event_filtering import known_match


class DoubleEventTruthTests(unittest.TestCase):
    def setUp(self):
        self.t=np.arange(0.,60.,.12)
        self.s=np.linspace(.1,.9,9)

    def test_latent_changes_and_observable_maxima_differ(self):
        for gap,amp,expected in [(5.,.04,1),(10.,.01,1),(10.,.04,2),(15.,.01,2)]:
            y, latent=double_fixture(self.t,self.s,2.04,'C2F',gap,amp)
            self.assertEqual(latent.shape,(2,len(self.s)))
            self.assertEqual(raw_truth(y,self.t)['uniform_observable_count'],expected)

    def test_native_zero_at_maximum_keeps_both_signed_phases(self):
        t=np.arange(7.,dtype=float)
        truth=raw_truth(np.array([0.,1.,2.,0.,-2.,-1.,0.])[:,None],t)
        event=truth['columns'][0][0]
        self.assertEqual(event['arrival_s'],3.)
        self.assertEqual(event['positive_peak_s'],2.)
        self.assertEqual(event['negative_peak_s'],4.)

    def test_one_candidate_cannot_recover_two_truth_events(self):
        groups=[dict(group_id='one',classification='C2F',assignments={
            0:[dict(section=0,error_s=0.)],1:[dict(section=1,error_s=0.)]})]
        result=one_to_one(groups,2,2)
        self.assertEqual(sum(r['matched'] for r in result),1)
        self.assertEqual(len(result),2)

    def test_opposite_raw_phase_is_not_matched_by_nearest_peak(self):
        t=np.arange(7.,dtype=float)
        truth=raw_truth(np.array([0.,1.,2.,0.,-2.,-1.,0.])[:,None],t)
        self.assertEqual(phase_identity(0,1,2.,truth),0)
        self.assertIsNone(phase_identity(0,1,4.,truth))

    def test_single_phase_reference_does_not_follow_measured_peak(self):
        raw=np.array([0.,1.,0.,0.,0.])[:,None]*np.ones((1,3))
        event=dict(event_id='shifted',classification='C2F',sections=[0,1,2],
                   arrival_time_s=[3.,3.,3.],evidence_kind='single_phase',leading_polarity=1)
        match=known_match(dict(events=[event],time_s=np.arange(5.)),
                          'C2F',np.ones(3),raw,np.array([0.,.5,1.]))
        self.assertEqual(match['arrival_RMSE_to_injection_s'],2.)


if __name__=='__main__':
    unittest.main()
