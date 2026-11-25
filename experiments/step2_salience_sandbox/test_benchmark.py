"""Pre-run checks of generation pairing, interval metrics and cache equivalence."""
import unittest
import numpy as np
from .benchmark_data import scenarios,make_case
from .benchmark_metrics import interval_error,evaluate,detailed_summary,unique_controls
from .core import BaselineMinMaxScorer,continuous_branch
from .upstream import extract_shared_upstream,stable_hash
from .matching import upstream_recall
from .fixtures import case,FIXED_NOISE_FLOOR


def fixture_bases():
    frame=np.arange(751)*.12;begin,end=frame[:-17],frame[17:];t=(begin+end)/2
    s=np.linspace(.1,.9,8);a=np.zeros((len(t),2,8))
    maps=dict(A=a,valid_mask=np.ones_like(a,bool),s=s,interval_start_s=begin,
              interval_end_s=end,interval_center_s=t,interval_dt_s=end-begin)
    return {k:maps for k in ('static','rigid','brightness')}


PARAMS=dict(timing=dict(Dev_center=16.,Val_center=17.,artifact_Dev_center=42.,artifact_Val_center=45.,
                       artifact_Dev_position_fraction=.45,artifact_Val_position_fraction=.65))
SOURCE=dict(texture_id='test',split='Dev',event_jitter=.1,artifact_jitter=0)


class BenchmarkTests(unittest.TestCase):
    def test_fixed_matrix_and_duration_core_identity(self):
        matrix=scenarios();self.assertEqual(len(matrix),31)
        selected=[r for r in matrix if r['scenario'] in ('medium_C2F','duration_30','duration_90')]
        generated=[make_case(SOURCE,s,fixture_bases(),PARAMS) for s in selected]
        self.assertTrue(all(truth==generated[0][1] for _,truth in generated))
        common=min(len(m['A']) for m,_ in generated)
        for maps,_ in generated:
            np.testing.assert_array_equal(maps['A'][:common],generated[0][0]['A'][:common])

    def test_structured_weak_exceeds_predefined_step4a_multiple(self):
        from analysis.mse_pregnancy_association.code.activity_strength import filtered_runs
        row=next(s for s in scenarios() if s['scenario']=='structured_weak_C2F')
        maps,truth=make_case(SOURCE,row,fixture_bases(),PARAMS)
        y=filtered_runs(maps['A'],maps['valid_mask'],maps['interval_center_s'])
        inside=(maps['interval_center_s']>=truth[0]['truth_start_s'])&(maps['interval_center_s']<=truth[0]['truth_end_s'])
        per_section=np.sqrt(np.mean(y[inside,0]**2,axis=0))
        self.assertGreaterEqual(float(np.median(per_section)),3*.000222)
        self.assertGreaterEqual(len(truth[0]['truth_sections']),3)

    def test_coexistence_retains_identical_weaker_truth(self):
        matrix=scenarios();base=fixture_bases()
        solo=next(s for s in matrix if s['scenario']=='structured_weak_C2F')
        _,known=make_case(SOURCE,solo,base,PARAMS)
        for ratio in (2,3,5):
            pair=next(s for s in matrix if s['scenario']==f'coexist_structured_weak_{ratio}')
            _,truth=make_case(SOURCE,pair,base,PARAMS)
            self.assertEqual(truth[0],known[0])
            self.assertAlmostEqual(truth[1]['truth_amplitude']/truth[0]['truth_amplitude'],ratio)

    def test_interval_error_is_not_midpoint_mae(self):
        self.assertEqual(interval_error(1,3,2)['distance'],0)
        self.assertEqual(interval_error(1,3,1)['distance'],0)
        self.assertEqual(interval_error(1,3,4)['distance'],1)
        self.assertIsNone(interval_error(None,None,1)['coverage'])

    def test_retained_section_cache_equals_full_rerun(self):
        maps,truth=case('clear_C2F');shared=extract_shared_upstream(maps,FIXED_NOISE_FLOOR)
        branch=continuous_branch(shared,BaselineMinMaxScorer(shared))
        record=dict(shared=shared,branches={'baseline_minmax':branch},truth=truth,
            source=dict(texture_id='test'),scenario=dict(scenario='clear_C2F',family='single'),
            case_id='test/clear',upstream=upstream_recall(shared,truth),downstream_cache={})
        for threshold in (.1,.6,.9,1.):
            cached=evaluate(record,'baseline_minmax',threshold)
            forced=evaluate(record,'baseline_minmax',threshold,force=True)
            self.assertEqual(stable_hash(cached),stable_hash(forced))


if __name__=='__main__':unittest.main()
