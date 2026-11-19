import unittest
import numpy as np
from experiments.activity_strength import RigidReference, score_activity


class ActivityStrength(unittest.TestCase):
    def test_reference_matches_offsets_duration_and_missing_support(self):
        y = np.arange(30., dtype=float).reshape(5, 1, 6)
        ref = RigidReference([('a', y, np.arange(5.))])
        measured = ref.maximum([0, 2], [0., 1.], [1., 2.])
        expected = []
        for start in range(3):
            for col in range(4):
                expected.append(np.median([np.sqrt(np.mean(y[start:start+2,0,col]**2)),
                    np.sqrt(np.mean(y[start+1:start+3,0,col+2]**2))]))
        self.assertEqual(measured['matched_windows'], len(expected))
        self.assertAlmostEqual(measured['maximum_rms'], max(expected))
        y[:] = np.nan
        self.assertIsNone(RigidReference([('a', y, np.arange(5.))]).maximum([0,2],[0.,1.],[1.,2.])['maximum_rms'])

    def test_one_section_spike_does_not_promote_whole_activity(self):
        t = np.arange(8.)
        source = {'base:a':dict(entity_id='base:a',profile={str(c):[
            dict(start_s=0.,end_s=7.)] for c in range(3)})}
        event = dict(event_proposal_id='e',canonical_source_entity_id='base:a',direction='C2F',ownership_status='retained_existing')
        ref = RigidReference([('noise', np.ones((8,1,3)), t)])
        y = np.full((8,3), .5); y[3,0] = 100.
        score = score_activity(event,source,y,np.ones_like(y,bool),t,ref)
        self.assertEqual(score['decision'], 'not_above_rigid_reference')
        self.assertEqual(score['activity_rms'], .5)
        y[:] = 2.
        self.assertEqual(score_activity(event,source,y,np.ones_like(y,bool),t,ref)['decision'],'above_rigid_reference')
        self.assertTrue(np.isfinite(y).all())

    def test_unmatched_reference_is_pending_not_low_strength(self):
        from experiments.test_activity_strength_trial import selection
        t = np.arange(8.); y = np.full((8,3),2.)
        source = {'base:a':dict(entity_id='base:a',profile={str(c):[
            dict(start_s=0.,end_s=7.)] for c in range(3)})}
        event = dict(event_proposal_id='e',canonical_source_entity_id='base:a',direction='C2F',ownership_status='retained_existing')
        reference = RigidReference([('too_narrow',np.ones((8,1,2)),t)])
        scores, keep, metrics = selection(dict(source_evidence=source,event_proposals=[event]),
                                         y,np.ones_like(y,bool),t,reference)
        self.assertEqual(scores[0]['decision'],'unscored_no_matched_reference')
        self.assertFalse(keep)
        self.assertEqual(metrics['not_below_reference_rows'],1)
        self.assertEqual(metrics['unscored_proposal_rows'],1)


if __name__ == '__main__': unittest.main()
