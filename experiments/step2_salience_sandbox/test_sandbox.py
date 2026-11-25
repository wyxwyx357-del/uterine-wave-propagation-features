"""Infrastructure tests A-J and guards; no scorer threshold tuning."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from . import upstream
from .core import (BaselineMinMaxScorer,continuous_branch,evaluate_threshold,baseline_equivalence_gate,
                   BaselineEquivalenceError,s_crit)
from .fixtures import case,KINDS,FIXED_NOISE_FLOOR
from .scorers import (DevRigidBank,PerSectionRigidReferenceScorer,OutOfBandMADScorer,
                      InBandMADScorer,PercentileScorer,mad_scale)
from .matching import match_events,upstream_recall,miss_attribution,activity_predictions


def tiny_bank():
    maps,_=case('zero')
    # Fixed nonzero reference texture is a software fixture, not a calibrated bank.
    maps['A'][:]=.0002*np.sin(2*np.pi*maps['interval_center_s']/20)[:,None,None]
    return DevRigidBank([dict(control_id='dev_zero_fixture',split='Dev',condition='rigid',maps=maps)])


def scorers(shared):
    return [BaselineMinMaxScorer(shared),PerSectionRigidReferenceScorer(tiny_bank()),
            OutOfBandMADScorer(shared,(10,25)),InBandMADScorer(shared),
            PercentileScorer(shared,95),PercentileScorer(shared,99)]


def truth(name='T',start=10,end=20,center=15,direction='C2F',sections=(0,1,2)):
    return dict(truth_event_id=name,truth_start_s=start,truth_end_s=end,truth_center_s=center,
                truth_direction=direction,truth_sections=list(sections),truth_amplitude_class='clear',
                truth_amplitude=.04,truth_speed=.04)


def prediction(name='P',center=15,direction='C2F',sections=(0,1,2),propagates=True):
    return dict(prediction_id=name,evidence_start_s=center-.5,evidence_end_s=center+.5,
                sections=list(sections),direction=direction,propagation_supported=propagates)


def matcher_examples():
    cases={
        'E_one_to_one':([truth()],[prediction()]),
        'F_two_predictions_one_truth':([truth()],[prediction('P1',14),prediction('P2',16)]),
        'G_two_truths_one_prediction':([truth('T1'),truth('T2',center=16)],[prediction()]),
        'H_direction_error':([truth()],[prediction(direction='F2C')]),
        'I_insufficient_spatial_overlap':([truth()],[prediction(sections=(1,2,3))]),
    }
    results={name:match_events(t,p,2.04,60) for name,(t,p) in cases.items()}
    for side,edge in [('lower',10-2.04),('upper',20+2.04)]:
        for label,mid in [('at',edge),('inside',np.nextafter(edge,15.)),
                          ('outside',np.nextafter(edge,-np.inf if side=='lower' else np.inf))]:
            # Zero-width evidence avoids re-rounding the exact boundary midpoint.
            p=prediction(center=float(mid));p['evidence_start_s']=p['evidence_end_s']=float(mid)
            results[f'J_{side}_{label}']=match_events([truth()],[p],2.04,60)
    return results


class SandboxTests(unittest.TestCase):
    def test_A_baseline_equivalence_all_tiny_inputs(self):
        for name in KINDS:
            with self.subTest(case=name):
                maps,_=case(name)
                shared=upstream.extract_shared_upstream(maps,FIXED_NOISE_FLOOR)
                branch=continuous_branch(shared,BaselineMinMaxScorer(shared))
                evaluation=evaluate_threshold(shared,branch,.6)
                self.assertTrue(baseline_equivalence_gate(maps,FIXED_NOISE_FLOOR,shared,branch,evaluation)['passed'])

    def test_A_gap_and_no_observable_support(self):
        for observable in (True,False):
            maps,_=case('clear_C2F')
            if observable:
                maps['valid_mask'][200:225,0,0]=False
                maps['A'][205:210,1,1]=np.nan
            else:
                maps['valid_mask'][:]=False
                maps['valid_mask'][:10]=True
            shared=upstream.extract_shared_upstream(maps,FIXED_NOISE_FLOOR)
            branch=continuous_branch(shared,BaselineMinMaxScorer(shared))
            evaluation=evaluate_threshold(shared,branch,.6)
            self.assertTrue(baseline_equivalence_gate(maps,FIXED_NOISE_FLOOR,shared,branch,evaluation)['passed'])

    def test_A_gate_reports_layer_and_blocks_a_difference(self):
        maps,_=case('clear_C2F');shared=upstream.extract_shared_upstream(maps,FIXED_NOISE_FLOOR)
        branch=continuous_branch(shared,BaselineMinMaxScorer(shared));evaluation=evaluate_threshold(shared,branch,.6)
        branch['normalized_values']=branch['normalized_values'].copy()
        branch['normalized_values'][0,0,0]+=.001
        with self.assertRaisesRegex(BaselineEquivalenceError,'normalized_values'):
            baseline_equivalence_gate(maps,FIXED_NOISE_FLOOR,shared,branch,evaluation)

    def test_A_audit_stops_before_reference_bank_on_gate_failure(self):
        from . import audit
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(audit,'baseline_equivalence_gate',side_effect=BaselineEquivalenceError('injected normalized_values mismatch')), \
             patch.object(audit,'DevRigidBank') as bank:
            with self.assertRaises(BaselineEquivalenceError):
                audit.run(SimpleNamespace(output=Path(directory),controls=Path(directory),step4a=Path(directory)))
            bank.assert_not_called()

    def test_B_scrit_random_scores_strict_equivalence(self):
        rng=np.random.default_rng(20261125)
        for _ in range(500):
            scores={c:(float(rng.normal()) if rng.random()>.2 else None) for c in range(25) if rng.random()>.15}
            values=[v for v in scores.values() if v is not None]
            for tau in [-2,-.5,0,.6,1,2,values[0] if values else 0]:
                longest=run=0
                for c in range(25):
                    value=scores.get(c)
                    run=run+1 if value is not None and np.isfinite(value) and value>tau else 0
                    longest=max(longest,run)
                critical=s_crit(scores)
                self.assertEqual(critical is not None and critical>tau,longest>=3)
        self.assertIsNone(s_crit({0:1,2:1,3:1,4:None}))
        self.assertFalse(s_crit({0:.9,1:.8,2:.7})>.7)

    def test_C_shared_candidate_pool_and_no_detector_rerun(self):
        maps,_=case('clear_C2F')
        with patch.object(upstream,'detect_partial',wraps=upstream.detect_partial) as detector, \
             patch.object(upstream,'prepare_parent',wraps=upstream.prepare_parent) as prepare, \
             patch.object(upstream,'decide_split',wraps=upstream.decide_split) as split, \
             patch.object(upstream,'reconcile',wraps=upstream.reconcile) as reconcile:
            shared=upstream.extract_shared_upstream(maps,FIXED_NOISE_FLOOR)
            calls=[x.call_count for x in (detector,prepare,split,reconcile)]
            self.assertEqual(detector.call_count,2)
            self.assertEqual(reconcile.call_count,2)
            signatures=[]
            for scorer in scorers(shared):
                branch=continuous_branch(shared,scorer)
                self.assertEqual(branch['pool_hash'],shared['pool_hash'])
                signatures.append([(w['wall'],p['event']['event_proposal_id'],p['event']['canonical_source_entity_id'])
                                   for w in branch['walls'] for p in w['proposals']])
            self.assertTrue(all(s==signatures[0] for s in signatures))
            self.assertEqual(calls,[x.call_count for x in (detector,prepare,split,reconcile)])

    def test_D_upstream_miss_not_charged_to_scorers(self):
        maps,known=case('upstream_miss');shared=upstream.extract_shared_upstream(maps,FIXED_NOISE_FLOOR)
        recall=upstream_recall(shared,known)
        self.assertEqual(recall['upstream_recall'],0)
        self.assertEqual(recall['upstream_missed_truth_ids'],['below_upstream'])
        for scorer in scorers(shared):
            branch=continuous_branch(shared,scorer)
            self.assertEqual(sum(len(w['proposals']) for w in branch['walls']),0)
            attribution=miss_attribution(recall,match_events(known,[],shared['span'],shared['exposure']['seconds']))
            self.assertEqual(attribution['salience_and_downstream_missed_represented_truth_ids'],[])

    def test_E_one_truth_one_prediction(self):
        r=matcher_examples()['E_one_to_one'];self.assertEqual((r['TP'],r['FP'],r['FN']),(1,0,0))

    def test_F_two_predictions_one_truth(self):
        r=matcher_examples()['F_two_predictions_one_truth'];self.assertEqual((r['TP'],r['FP'],r['FN']),(1,1,0))

    def test_G_two_truths_one_prediction(self):
        r=matcher_examples()['G_two_truths_one_prediction'];self.assertEqual((r['TP'],r['FP'],r['FN']),(1,0,1))

    def test_H_direction_error_is_still_event_tp(self):
        r=matcher_examples()['H_direction_error'];self.assertEqual((r['TP'],r['FP'],r['FN']),(1,0,0))
        self.assertEqual(r['direction_accuracy'],0)

    def test_I_insufficient_overlap(self):
        r=matcher_examples()['I_insufficient_spatial_overlap'];self.assertEqual((r['TP'],r['FP'],r['FN']),(0,1,1))
        self.assertEqual(r['pair_evidence'][0]['intersection_count'],2)
        self.assertFalse(r['pair_evidence'][0]['spatial_gate_passed'])

    def test_J_actual_measurement_span_inclusive_boundaries(self):
        results=matcher_examples()
        for side in ('lower','upper'):
            for position in ('at','inside','outside'):
                self.assertEqual(results[f'J_{side}_{position}']['TP'],0 if position=='outside' else 1)

    def test_matcher_maximum_cardinality_beats_greedy(self):
        r=match_events([truth('A',10,30,20),truth('B',10,12,11)],
                       [prediction('P1',11),prediction('P2',25)],2.04,60)
        self.assertEqual(r['TP'],2)
        self.assertEqual({(e['truth_id'],e['prediction_id']) for e in r['matches']},{('A','P2'),('B','P1')})

    def test_matcher_minimum_total_center_deviation(self):
        r=match_events([truth('A',0,100,10),truth('B',0,100,20)],
                       [prediction('P1',11),prediction('P2',19)],2.04,60)
        self.assertEqual(sum(e['center_deviation_s'] for e in r['matches']),2)

    def test_propagation_error_does_not_erase_event_tp(self):
        r=match_events([truth()],[prediction(direction=None,propagates=False)],2.04,60)
        self.assertEqual(r['TP'],1);self.assertEqual(r['propagation_supported_accuracy'],0)
        self.assertEqual(r['local_propagation_confusion']['truth_1_prediction_0'],1)

    def test_val_or_nonzero_motion_cannot_enter_rigid_bank(self):
        maps,_=case('zero')
        for split,condition in [('Val','rigid'),('Dev','brightness')]:
            with self.assertRaises(ValueError):
                DevRigidBank([dict(control_id='forbidden',split=split,condition=condition,maps=maps)])

    def test_numerators_are_sectionwise_and_mad_definition_explicit(self):
        self.assertAlmostEqual(mad_scale([1,2,3]),1.4826)
        maps,_=case('clear_C2F');shared=upstream.extract_shared_upstream(maps,FIXED_NOISE_FLOOR)
        proposal=shared['walls'][0]['reconciliation']['event_proposals'][0]
        for scorer in scorers(shared):
            rows=scorer.score(shared,0,proposal)
            self.assertGreaterEqual(len(rows),3)
            self.assertEqual(len({r['section'] for r in rows}),len(rows))
            self.assertTrue(all('score' in r for r in rows))

    def test_threshold_evaluation_calls_production_downstream_each_time(self):
        from . import core
        maps,_=case('clear_C2F');shared=upstream.extract_shared_upstream(maps,FIXED_NOISE_FLOOR)
        branch=continuous_branch(shared,BaselineMinMaxScorer(shared))
        with patch.object(core,'count_activities',wraps=core.count_activities) as counter:
            evaluate_threshold(shared,branch,.6);evaluate_threshold(shared,branch,.9)
            self.assertEqual(counter.call_count,2)

    def test_pool_mutation_is_detected(self):
        maps,_=case('clear_C2F');shared=upstream.extract_shared_upstream(maps,FIXED_NOISE_FLOOR)
        scorer=BaselineMinMaxScorer(shared)
        shared['walls'][0]['reconciliation']['event_proposals'][0]['direction']='changed'
        with self.assertRaisesRegex(RuntimeError,'mutated'):
            continuous_branch(shared,scorer)


if __name__=='__main__':
    unittest.main()
