"""Scope/interface contracts; these are not additional benchmark conditions."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
from experiments.candidate_split_verifier import prepare_parent,decide_split
from experiments.test_double_event_signals import double_fixture
from uterine_wave import radial_events as core


class CandidateSplitScopeTests(unittest.TestCase):
    def setUp(self):
        self.t=np.arange(0.,60.,.12);self.s=np.linspace(.2,.8,9)
        self.y,_=double_fixture(self.t,self.s,2.04,'C2F',10.,.04)
        self.valid=np.ones_like(self.y,dtype=bool)
        # A declared scope exercises the verifier contract only. Benchmark
        # parents always come verbatim from the cached periodic detector.
        self.parent=dict(event_id='scope_contract',start_s=10.,end_s=50.,
                         sections=list(range(9)),classification='complex',
                         phase_ridge_ids=[],arrival_time_s=[],evidence_kind='unit_scope')

    def prepare(self,y=None,valid=None,parent=None):
        return prepare_parent(self.y if y is None else y,self.valid if valid is None else valid,
            self.t,self.s,self.parent if parent is None else parent,
            noise_floor=.00075,measurement_span_s=2.04)

    def test_outside_context_cannot_change_the_result(self):
        changed=self.y.copy()
        outside=(self.t<self.parent['start_s']-2.04)|(self.t>self.parent['end_s']+2.04)
        changed[outside]=1000.
        a=self.prepare();b=self.prepare(y=changed)
        np.testing.assert_array_equal(a['local_observations']['smoothed_A'],b['local_observations']['smoothed_A'])
        self.assertEqual(decide_split(a),decide_split(b))

    def test_parent_is_immutable_children_have_local_provenance(self):
        original=deepcopy(self.parent)
        result=decide_split(self.prepare())
        self.assertEqual(self.parent,original)
        self.assertEqual(result['split_status'],'split_two')
        for child in (result['child_1'],result['child_2']):
            self.assertEqual(child['parent_candidate_id'],self.parent['event_id'])
            self.assertTrue(set(child['sections'])<=set(self.parent['sections']))
            self.assertTrue(all(self.parent['start_s']<=x<=self.parent['end_s'] for x in child['arrival_time_s']))
        self.assertIsNone(result['event_count'])
        self.assertIsNone(result['event_frequency'])

    def test_missing_runs_do_not_support_a_cross_boundary_split(self):
        valid=self.valid.copy();valid[(self.t>=29.5)&(self.t<30.5)]=False
        self.assertNotEqual(decide_split(self.prepare(valid=valid))['split_status'],'split_two')

    def test_nonadjacent_native_sections_are_not_compressed(self):
        parent=dict(self.parent,sections=[0,2,4,6,8])
        self.assertNotEqual(decide_split(self.prepare(parent=parent))['split_status'],'split_two')

    def test_activity_identity_does_not_force_propagation(self):
        with patch.object(core,'_pair_phases',return_value=([],[])):
            result=decide_split(self.prepare())
        self.assertEqual(result['split_status'],'split_two')
        for child in (result['child_1'],result['child_2']):
            self.assertTrue(child['activity_split_supported'])
            self.assertFalse(child['propagation_supported'])
            self.assertIsNone(child['direction'])
            self.assertIsNone(child['velocity_paths_per_s'])


if __name__=='__main__':
    unittest.main()
