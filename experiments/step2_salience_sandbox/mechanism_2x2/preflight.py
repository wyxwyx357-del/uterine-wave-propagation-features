"""Eight fixed implementation checks, run only after prelock exists."""
import unittest
import numpy as np
from ..fixtures import case,FIXED_NOISE_FLOOR
from ..upstream import extract_shared_upstream,stable_hash
from ..scorers import OutOfBandMADScorer
from analysis.mse_pregnancy_association.code import candidate_split_verifier as split
from analysis.mse_pregnancy_association.code import activity_counting as counting
from .mechanisms import intervention,filter_only,weighted_pair,butter_run,strength,CORE

def four_lobes(amplitudes):
    filtered=np.zeros((len(amplitudes),3));ridges=[]
    for i,amp in enumerate(amplitudes):
        sign=(-1,1,-1,1)[i] if len(amplitudes)==4 else (1,-1)[i]
        filtered[i,:]=sign*amp
        ridges.append(dict(id=i,sign=sign,nodes=[dict(column=c,first=i,last=i,amplitude=amp) for c in range(3)]))
    weights=(11.,13.,12.) if len(amplitudes)==4 else (5.,)
    edges={(i,i+1):dict(first=i,second=i+1,columns=[0,1,2],association_weight=w) for i,w in enumerate(weights)}
    return ridges,edges,filtered

class ImplementationTests(unittest.TestCase):
    def test_1_M0_exact_fixture(self):
        maps,_=case('clear_C2F');baseline=extract_shared_upstream(maps,FIXED_NOISE_FLOOR)
        with intervention('M0'):replay=extract_shared_upstream(maps,FIXED_NOISE_FLOOR)
        self.assertEqual(stable_hash(baseline),stable_hash(replay))
    def test_2_force_G_one_exact(self):
        for kind in ('clear_C2F','two','zero'):
            maps,_=case(kind)
            with intervention('M0'):a=extract_shared_upstream(maps,FIXED_NOISE_FLOOR)
            with intervention('M1',force_one=True):b=extract_shared_upstream(maps,FIXED_NOISE_FLOOR)
            self.assertEqual(stable_hash(a),stable_hash(b))
    def test_3_four_lobe_manual_weights(self):
        r,e,f=four_lobes([.2,1.,1.,.2]);selected,_,audit=weighted_pair(r,e,f,True)
        np.testing.assert_allclose([x['G'] for x in audit],[.2,1.,.2],rtol=0,atol=0)
        np.testing.assert_allclose([x['W_new'] for x in audit],[2.2,13.,2.4],rtol=1e-15)
        self.assertEqual(selected,[(1,2)])
    def test_4_sign_symmetric(self):
        r,e,f=four_lobes([1.,1.]);_,_,a=weighted_pair(r,e,f,True);self.assertEqual(a[0]['G'],1.)
    def test_5_absolute_asymmetric(self):
        r,e,f=four_lobes([1.,.4]);_,_,a=weighted_pair(r,e,f,True);self.assertEqual(a[0]['G'],.4)
    def test_6_constant_zero_finite(self):
        t=np.arange(500)*.12
        for value in (0.,1.):self.assertTrue(np.isfinite(butter_run(np.full(len(t),value),t)).all())
    def test_7_gap_independent(self):
        t=np.arange(600)*.12;y=np.sin(t)[:,None,None]*np.ones((1,2,3));valid=np.ones_like(y,bool);valid[240:260]=False
        with filter_only('M2'):a=strength.filtered_runs(y,valid,t)
        y[260:]*=100
        with filter_only('M2'):b=strength.filtered_runs(y,valid,t)
        np.testing.assert_array_equal(a[:240],b[:240]);self.assertTrue(np.isnan(a[240:260]).all())
    def test_8_other_interfaces_identical_and_MAD_denominator_unchanged(self):
        maps,_=case('clear_C2F');measure=CORE._measure_pair;local=split.measurement_span_mean;decide=split.decide_split
        prop=counting._node_evidence;identity=counting.compare_identity_without_direction
        with intervention('M0'):a=extract_shared_upstream(maps,FIXED_NOISE_FLOOR)
        with intervention('M3'):
            self.assertIs(CORE._measure_pair,measure);self.assertIs(split.measurement_span_mean,local);self.assertIs(split.decide_split,decide)
            self.assertIs(counting._node_evidence,prop);self.assertIs(counting.compare_identity_without_direction,identity)
            # Local split uses the same function OUTSIDE the global detector;
            # it must receive original weights and produce original selection.
            r,e,f=four_lobes([.2,1.,1.,.2])
            self.assertEqual(set(CORE._pair_phases(r,e)[0]),{(0,1),(2,3)})
            b=extract_shared_upstream(maps,FIXED_NOISE_FLOOR)
        self.assertEqual(stable_hash(OutOfBandMADScorer(a,(10,20)).scales),stable_hash(OutOfBandMADScorer(b,(10,20)).scales))
