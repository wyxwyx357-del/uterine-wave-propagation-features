import unittest
import numpy as np
from tests.test_direction_identity_trial import node
from experiments.activity_counting import count_activities,retained_phase_fits


class ActivityCountingTests(unittest.TestCase):
    def test_same_structure_opposite_labels_counts_once(self):
        r=count_activities([node('a',0),node('b',1,'F2C')],np.arange(3),2.,.1,60.)
        self.assertEqual(r['significant_propagation_activity_count'],1)
        self.assertTrue(r['activities'][0]['direction_conflict_QC'])

    def test_two_separate_activities_count_twice(self):
        r=count_activities([node('a',0),node('b',1,shift=10.,arrival=[20,21,22])],np.arange(3),2.,.1,60.)
        self.assertEqual(r['significant_propagation_activity_count'],2)

    def test_synchronous_is_not_propagation(self):
        r=count_activities([node('a',0,arrival=[10,10,10])],np.arange(3),2.,.1,60.)
        self.assertEqual(r['significant_propagation_activity_count'],0)

    def test_phase_conflict_evidence_is_passed_to_classifier(self):
        n=node('a',0,arrival=[10.,12.,11.])
        for c in n['sections']:
            for p in n['profile'][c]:p['peak_s']=20.-c+3*p['sign']
        result=count_activities([n],np.arange(3),2.,.1,60.)
        m=result['activities'][0]['block_evidence'][0]['measurement']
        self.assertEqual(len(m['phase_arrival_fits']),2)
        self.assertIn('phase_direction_conflict',m['quality_flags'])
        self.assertEqual(result['significant_propagation_activity_count'],0)


if __name__=='__main__':unittest.main()
