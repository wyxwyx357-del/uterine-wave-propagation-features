import unittest
import numpy as np
from experiments.normalized_activity import normalize_magnitude, score_salience


class NormalizedActivity(unittest.TestCase):
    def test_whole_video_two_walls_scope_and_strict_cutoff(self):
        y=np.array([[[0.,6.,10.],[0.,3.,5.]]]);valid=np.ones_like(y,bool)
        n,meta=normalize_magnitude(y,valid)
        np.testing.assert_allclose(n[0,0],[0.,.6,1.])
        np.testing.assert_allclose(n[0,1],[0.,.3,.5])
        self.assertEqual(meta['salient_samples'],1)
        y[0,1,2]=1e9;valid[0,1,2]=False
        n,meta=normalize_magnitude(y,valid)
        self.assertEqual(meta['maximum'],10.)
        self.assertTrue(np.isnan(n[0,1,2]))

    def test_constant_map_does_not_create_salience(self):
        y=np.ones((4,2,5));n,meta=normalize_magnitude(y,np.ones_like(y,bool))
        self.assertEqual(meta['status'],'constant_magnitude')
        self.assertFalse((n>.6).any())

    def test_native_adjacent_support_no_spatial_bridging_or_identity_mutation(self):
        t=np.arange(4.);v=np.ones((4,6),bool);y=np.zeros_like(v,float)
        source={'base:a':dict(profile={str(c):[dict(start_s=0.,end_s=3.)] for c in range(6)})}
        event=dict(event_proposal_id='e',canonical_source_entity_id='base:a',direction='C2F',ownership_status='retained_existing')
        y[1,[0,2,4]]=1
        score=score_salience(event,source,y,v,t)
        self.assertFalse(score['retained_sections'])
        y[1,[1,2,3]]=1
        score=score_salience(event,source,y,v,t)
        self.assertEqual(score['retained_sections'],[0,1,2,3,4])
        self.assertEqual(event['direction'],'C2F')
        v[2,2]=False
        self.assertFalse(score_salience(event,source,y,v,t)['retained_sections'])


if __name__=='__main__':unittest.main()
