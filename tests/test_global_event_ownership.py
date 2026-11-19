"""Ownership/interface unit contracts, not new scientific validation fixtures."""
from copy import deepcopy
import unittest
import numpy as np
from experiments.global_event_ownership import reconcile,build_entities,compare_identity
from experiments.apply_global_ownership_real import export_missing_observations


def source_event(eid,center,sections=range(5)):
    ids=list(sections);arrival=[center+.2*c for c in ids]
    return dict(event_id=eid,sections=ids,arrival_time_s=arrival,classification='C2F',
        evidence_kind='paired_phase',phase_ridge_ids=[2*int(eid[-1]),2*int(eid[-1])+1],
        start_s=center-5.,end_s=center+6.,leading_polarity=1)


def source_report(events):
    fragments=[]
    for e in events:
        phases=[]
        for c,arrival in zip(e['sections'],e['arrival_time_s']):
            for sign in (1,-1):
                phases.append(dict(section=c,sign=sign,peak_s=arrival-2*sign,
                    observed_start_s=-25. if sign==1 else arrival,
                    observed_end_s=arrival if sign==1 else 45.,run_id=0))
        fragments.append(dict(fragment_id=e['event_id'],sections=e['sections'],
            arrival_time_s=e['arrival_time_s'],local_phase_support=phases))
    return dict(events=events,fragments=fragments,
                proposed_events=[dict(proposal_id=e['event_id'],fragment_ids=[e['event_id']]) for e in events])


def child(eid,parent,center,sections=range(3)):
    ids=list(sections);arrivals=[center+.2*c for c in ids]
    return dict(event_id=eid,parent_candidate_id=parent,sections=ids,arrival_time_s=arrivals,
        direction='C2F',classification='C2F',evidence_kind='paired_phase',phase_ridge_ids=[0,1],
        supporting_phase_evidence=[dict(section=c,phases=[dict(sign=sign,peak_s=arrival-2*sign,
            observed_start_s=-25. if sign==1 else arrival,
            observed_end_s=arrival if sign==1 else 45.,run_id=0) for sign in (1,-1)])
            for c,arrival in zip(ids,arrivals)])


class GlobalOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.t=np.arange(-30.,50.,.12);self.valid=np.ones((len(self.t),5),bool)

    def run_ownership(self,report,proposals):
        return reconcile(report,proposals,self.t,self.valid,measurement_span_s=2.04)

    def test_existing_wider_event_preserved_and_child_aliases_reused(self):
        report=source_report([source_event('p0',5.),source_event('p1',15.)]);original=deepcopy(report)
        split=dict(parent_candidate_id='p0',split_status='split_two',
            child_1=child('p0_child_1','p0',5.),child_2=child('p0_child_2','p0',15.))
        result=self.run_ownership(report,[split])
        self.assertEqual(report,original)
        self.assertEqual(len(result['event_proposals']),2)
        self.assertEqual(result['split_actions'][0]['status'],'reuse_existing')
        self.assertEqual(len(set(result['split_actions'][0]['final_event_ids'])),2)
        weak=next(e for e in result['event_proposals'] if 'p1' in e['source_parent_ids'])
        self.assertEqual(weak['supporting_sections'],list(range(5)))
        self.assertEqual(weak['canonical_measurement'],report['events'][1])
        self.assertIsNone(result['event_count']);self.assertIsNone(result['event_frequency'])

    def test_siblings_are_not_merged_into_one_existing_identity(self):
        report=source_report([source_event('p0',5.)])
        split=dict(parent_candidate_id='p0',split_status='split_two',
            child_1=child('p0_child_1','p0',5.),child_2=child('p0_child_2','p0',5.5))
        result=self.run_ownership(report,[split])
        self.assertEqual(result['split_actions'][0]['status'],'ambiguous')
        self.assertEqual(len(result['event_proposals']),1)
        self.assertNotIn('child:p0_child_2',result['source_to_final'])
        self.assertIn('child:p0_child_2',result['source_evidence'])

    def test_transitive_timing_bridge_cannot_join_separate_original_events(self):
        report=source_report([source_event('p0',0.),source_event('p1',1.9),source_event('p2',3.8)])
        result=self.run_ownership(report,[])
        self.assertNotEqual(result['source_to_final']['base:p0'],result['source_to_final']['base:p2'])
        self.assertTrue(any(a['source_entity_id']=='base:p1' for a in result['ambiguous_ownership']))

    def test_opposite_direction_with_same_structure_counts_once(self):
        report=source_report([source_event('p0',5.),source_event('p1',5.)])
        report['events'][1]['classification']='F2C'
        result=self.run_ownership(report,[])
        self.assertEqual(len(result['event_proposals']),1)
        entities,_=build_entities(report,[],self.t,self.valid)
        rel=compare_identity(entities['base:p0'],entities['base:p1'],sample_dt_s=.12,measurement_span_s=2.04)
        self.assertEqual(rel['relation'],'duplicate')
        self.assertTrue(rel['direction_conflict'])

    def test_physical_continuity_is_required_for_duplicate_merge(self):
        report=source_report([source_event('p0',5.),source_event('p1',5.)])
        entities,_=build_entities(report,[],self.t,self.valid)
        other=deepcopy(entities['base:p1'])
        for nodes in other['profile'].values():
            for node in nodes:node['epoch']=(999,1200)
        rel=compare_identity(entities['base:p0'],other,sample_dt_s=.12,measurement_span_s=2.04)
        self.assertNotIn(rel['relation'],('duplicate','partial_duplicate'))

    def test_phase_order_is_checked_even_with_partial_other_sections(self):
        report=source_report([source_event('p0',5.),source_event('p1',5.)])
        entities,_=build_entities(report,[],self.t,self.valid)
        a=entities['base:p0'];b=deepcopy(entities['base:p1']);b['complete']=False
        for c in b['profile']:b['profile'][c]=list(reversed(b['profile'][c]))
        rel=compare_identity(a,b,sample_dt_s=.12,measurement_span_s=2.04)
        self.assertEqual(rel['relation'],'distinct')

    def test_missing_observations_export_as_null_without_filling(self):
        values=np.array([[1.,np.nan],[2.,3.]])
        record=[dict(local_observations=dict(raw_A=values,smoothed_A=values.copy(),valid_mask=np.isfinite(values)))]
        export_missing_observations(record)
        self.assertIsNone(record[0]['local_observations']['raw_A'][0][1])
        self.assertEqual(record[0]['local_observations']['smoothed_A'][1],[2.,3.])
        self.assertTrue(np.isnan(values[0,1]))


if __name__=='__main__':unittest.main()
