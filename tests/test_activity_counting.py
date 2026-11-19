import unittest
import numpy as np
from tests.test_direction_identity_trial import node
from experiments.activity_counting import count_activities


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


if __name__=='__main__':unittest.main()
