"""Gate boundaries and complete reporting tested BEFORE real-panel prelock."""
from copy import deepcopy
import tempfile,unittest
from pathlib import Path
from .gates import decide
from .scope import audit_scope

def fake():
    strata={label:dict(TP=80,truth=100) for label in ('clear','medium','structured_weak')}
    aggregate={'M0':dict(A2=80,TP=80,FP=80,FN=20,strata=deepcopy(strata)),
               'M1':dict(A2=16,TP=80,FP=10,FN=20,strata=deepcopy(strata))}
    textures={str(i):dict(M0=dict(A2=10),M1=dict(A2=2)) for i in range(8)}
    co={m:dict(TP=80,truth_count=100,cross_truth_pairing_error=1,merge_two_truth_error=1) for m in ('M0','M1')}
    neg={m:dict(FP_min=1,FP_max=5,minutes=56.) for m in ('M0','M1')}
    return aggregate,textures,co,neg

class ValidationTests(unittest.TestCase):
    def test_01_scope_matches_frozen_M1_and_exercises_actual_split(self):
        result=audit_scope();self.assertTrue(result['passed']);self.assertGreater(result['local_split_pair_call_count'],0)
    def test_02_eighty_percent_boundary_passes(self):
        args=fake();r=decide(*args,True);self.assertEqual(r['M1_UNSEEN_VALIDATION'],'PASS');self.assertEqual(r['evidence']['A2Reduction'],.8)
        args[0]['M1']['A2']=17;self.assertFalse(decide(*args,True)['Gate1_subgates']['G1.1'])
    def test_03_signed_negative_texture_contribution_is_preserved(self):
        a,t,c,n=fake()
        for x in t.values():x['M1']['A2']=0
        t['0']['M1']['A2']=11;a['M1']['A2']=11
        r=decide(a,t,c,n,True);self.assertLess(r['evidence']['texture_contributions']['0'],0)
        self.assertEqual(r['evidence']['contribution_denominator'],69)
    def test_04_concentration_and_leave_one_out_are_real_gates(self):
        a,t,c,n=fake();t['0']['M0']['A2']=100
        a['M0']['A2']=170;a['M1']['A2']=16
        r=decide(a,t,c,n,True);self.assertFalse(r['Gate1_subgates']['G1.4'])
    def test_05_recall_exact_minus_two_points(self):
        a,t,c,n=fake();a['M1']['strata']['medium']['TP']=78
        self.assertTrue(decide(a,t,c,n,True)['Gate3_subgates']['medium'])
        a['M1']['strata']['medium']['TP']=77
        self.assertFalse(decide(a,t,c,n,True)['Gate3_subgates']['medium'])
    def test_06_coexist_pooled_boundary_and_error_counts(self):
        a,t,c,n=fake();c['M1']['TP']=78
        self.assertTrue(decide(a,t,c,n,True)['gates']['Gate4'])
        c['M1']['merge_two_truth_error']=2
        self.assertFalse(decide(a,t,c,n,True)['Gate4_subgates']['G4.2'])
    def test_07_negative_gate_uses_upper_bound(self):
        a,t,c,n=fake();n['M1'].update(FP_min=0,FP_max=6)
        self.assertFalse(decide(a,t,c,n,True)['gates']['Gate5'])
    def test_08_less_than_eight_cannot_receive_formal_pass(self):
        self.assertEqual(decide(*fake(),False)['M1_UNSEEN_VALIDATION'],'PILOT_ONLY_NO_FORMAL_DECISION')
    def test_09_zero_M0_A2_and_missing_LOO_denominator_fail(self):
        a,t,c,n=fake();a['M0']['A2']=0;a['M1']['A2']=0
        for x in t.values():x['M0']['A2']=x['M1']['A2']=0
        r=decide(a,t,c,n,True);self.assertFalse(r['Gate1_subgates']['G1.1']);self.assertFalse(r['Gate1_subgates']['G1.5'])
    def test_10_complete_report_on_analytical_inputs(self):
        from ..test_benchmark import fixture_bases,PARAMS,SOURCE
        from ..benchmark_data import make_case,scenarios
        from ..upstream import extract_shared_upstream,stable_hash
        from ..core import BaselineMinMaxScorer,continuous_branch,evaluate_threshold
        from ..mechanism_2x2.mechanisms import intervention
        from .evaluate import evaluate_saved
        from .report import write
        results={'M0':[],'M1':[]}
        for scenario in [s for s in scenarios() if s['scenario'] in ('coexist_medium_2','coexist_medium_3','coexist_medium_5')]:
            maps,truth=make_case(SOURCE,scenario,fixture_bases(),PARAMS)
            for model in ('M0','M1'):
                logs=[]
                with intervention(model,logs):shared=extract_shared_upstream(maps,.0007509593468916695)
                ev=evaluate_threshold(shared,continuous_branch(shared,BaselineMinMaxScorer(shared)),.6)
                record=dict(case_id='analytic/'+scenario['scenario'],source=dict(texture_id='analytic'),scenario=scenario,truth=truth,shared=shared,
                    interval_start=maps['interval_start_s'],interval_end=maps['interval_end_s'],input_hash=stable_hash(maps))
                results[model].append(evaluate_saved(dict(record=record,evaluation=ev,pairing_logs=logs),model))
        with tempfile.TemporaryDirectory() as folder:
            report=write(Path(folder),results,False)
            self.assertEqual(report['gate']['M1_UNSEEN_VALIDATION'],'PILOT_ONLY_NO_FORMAL_DECISION')
            self.assertTrue((Path(folder)/'M1_UNSEEN_TEXTURE_GATE_DECISION.json').is_file())

if __name__=='__main__':unittest.main()
