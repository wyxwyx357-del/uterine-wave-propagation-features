import unittest
import numpy as np
from experiments.patient_activity_aggregation import aggregate_nodes,observation_union,aggregate_by_direction


def node(name,wall,shift=0.,direction='C2F'):
    return dict(node_id=name,wall=wall,sections=[0,1,2],direction=direction,arrival_marker='zero',
        arrival={c:10.+c+shift for c in range(3)},profile={c:[
            dict(sign=1,peak_s=8.+c+shift,start_s=5.+shift,end_s=10.+c+shift,epoch=[wall,40]),
            dict(sign=-1,peak_s=12.+c+shift,start_s=10.+c+shift,end_s=18.+shift,epoch=[wall,40])]
            for c in range(3)})


class PatientActivityAggregation(unittest.TestCase):
    def test_two_walls_same_activity_count_once_despite_different_run_ids(self):
        result=aggregate_nodes([node('a',0),node('b',1,.2)],2.,.1)
        self.assertEqual(result['patient_candidate_rows'],1)
        self.assertEqual(result['cross_wall_pairs_merged'],1)
        self.assertEqual(result['activities'][0]['walls'],[0,1])

    def test_separate_activities_and_competing_owners_are_not_forced_together(self):
        self.assertEqual(aggregate_nodes([node('a',0),node('b',1,10.)],2.,.1)['patient_candidate_rows'],2)
        result=aggregate_nodes([node('a',0),node('b',1,.2),node('c',1,.3)],2.,.1)
        self.assertEqual(result['patient_candidate_rows'],3)
        self.assertEqual(result['cross_wall_pairs_merged'],0)
        self.assertEqual(result['unresolved_cross_wall_rows'],3)
        self.assertEqual(result['conditional_identity_count_range'],[2,3])

    def test_direction_conflict_preserved_and_exposure_is_union_including_quiet(self):
        result=aggregate_nodes([node('a',0),node('b',1,0.,'F2C')],2.,.1)
        self.assertEqual(result['patient_candidate_rows'],2)
        self.assertTrue(result['unresolved_cross_wall_rows'])
        maps=dict(valid_mask=np.ones((3,2,3),bool),A=np.zeros((3,2,3)),
                  interval_start_s=np.array([0.,1.,5.]),interval_end_s=np.array([2.,3.,7.]))
        exposure=observation_union(maps)
        self.assertEqual(exposure['seconds'],5.)
        self.assertEqual(exposure['intervals_s'],[[0.,3.],[5.,7.]])

    def test_primary_direction_buckets_exclude_local_and_do_not_merge_opposing_waves(self):
        nodes=[node('a',0),node('b',1,.2),node('reverse',0,0.,'F2C'),node('local',1,0.,None)]
        result=aggregate_by_direction(nodes,2.,.1,60.)
        self.assertEqual(result['C2F']['conditional_identity_count_range'],[1,1])
        self.assertEqual(result['F2C']['conditional_identity_count_range'],[1,1])
        self.assertEqual(result['other']['conditional_identity_count_range'],[1,1])
        self.assertEqual(result['C2F']['candidate_rate_range_cpm'],[1.,1.])
        same_direction_separate=aggregate_by_direction([node('a',0),node('later',1,10.)],2.,.1,60.)
        self.assertEqual(same_direction_separate['C2F']['conditional_identity_count_range'],[2,2])


if __name__=='__main__':unittest.main()
