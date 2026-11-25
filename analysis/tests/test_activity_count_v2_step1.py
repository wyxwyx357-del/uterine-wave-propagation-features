"""Deterministic Step-1 tests for activity_count_v2.

These tests deliberately isolate the two logic fixes. They do not retune the
0.6 salience threshold, filters, event splitter, propagation classifier, or
core measurement chain.

Run from repository root after applying the Step-1 patch:
    pytest -q analysis/tests/test_activity_count_v2_step1.py
"""
import numpy as np
import pytest

import analysis.mse_pregnancy_association.code.activity_counting as counting
from analysis.mse_pregnancy_association.code.normalized_activity import normalize_magnitude
from analysis.mse_pregnancy_association.code.patient_activity_aggregation import observation_union


def _maps(duration_s=60.0, dt=0.12, sections=12):
    n = int(round(duration_s/dt))
    begin = np.arange(n, dtype=float)*dt
    end = begin + dt
    center = 0.5*(begin+end)
    A = np.zeros((n, 2, sections), float)
    valid = np.ones_like(A, bool)
    return dict(A=A, valid_mask=valid, interval_start_s=begin,
                interval_end_s=end, interval_center_s=center,
                interval_dt_s=np.full(n, dt), s=np.linspace(0, 1, sections))


def _node(node_id, wall, *, propagates=True, start_s=10.0):
    sections=[0,1,2]
    profile={c:[dict(sign=1, peak_s=start_s+0.2*c,
                     start_s=start_s-0.5, end_s=start_s+1.0)] for c in sections}
    return dict(node_id=node_id, wall=wall, sections=sections,
                arrival={c:start_s+0.2*c for c in sections}, profile=profile,
                direction='C2F' if propagates else None, _propagates=propagates)


def _patch_logic(monkeypatch, relations):
    """Control only identity relation + propagation truth for Step-1 logic tests."""
    def fake_compare(a, b, span, dt):
        relation = relations.get(frozenset((a['node_id'], b['node_id'])),
                                 'no_shared_identity_evidence')
        return dict(node_a=a['node_id'], node_b=b['node_id'], relation=relation)

    def fake_node_evidence(node, s, dt):
        return [dict(node_id=node['node_id'], sections=[0,1,2], measurement={},
                     phase_fit_availability=[],
                     propagation_supported=bool(node['_propagates']))]

    monkeypatch.setattr(counting, 'compare_identity_without_direction', fake_compare)
    monkeypatch.setattr(counting, '_node_evidence', fake_node_evidence)


def _count(nodes, monkeypatch, relations, obs_s=60.0):
    _patch_logic(monkeypatch, relations)
    return counting.count_activities(nodes, np.linspace(0,1,12), 2.0, 0.12, obs_s)


# T1 -------------------------------------------------------------------------
def test_T1_zero_activity_negative_control_has_zero_count_and_60s_exposure(monkeypatch):
    maps=_maps(60.0)
    exposure=observation_union(maps, min_run_s=12.0)
    result=_count([], monkeypatch, {}, exposure['seconds'])
    assert exposure['seconds'] == pytest.approx(60.0, abs=1e-9)
    assert result['significant_propagation_activity_count_range'] == [0,0]
    assert result['significant_propagation_activity_frequency_cpm_range'] == [0.0,0.0]


# T2 -------------------------------------------------------------------------
@pytest.mark.parametrize('truth_count',[1,2,3])
def test_T2_strict_bilateral_one_to_one_events_are_exact(monkeypatch, truth_count):
    nodes=[];relations={}
    for i in range(truth_count):
        a=_node(f'a{i}',0,start_s=10+15*i)
        b=_node(f'b{i}',1,start_s=10+15*i)
        nodes += [a,b]
        relations[frozenset((a['node_id'],b['node_id']))]='shared_identity_supported'
    result=_count(nodes,monkeypatch,relations)
    assert result['significant_propagation_activity_count_range'] == [truth_count,truth_count]
    assert result['significant_propagation_activity_count_v2_exact'] == truth_count
    assert result['ambiguous_cross_wall_propagation_pairs'] == 0


# T3 -------------------------------------------------------------------------
def test_T3_one_wall_partial_support_does_not_duplicate_when_identity_is_strict(monkeypatch):
    # Step-1 identity accounting contract: once upstream still yields one node
    # per wall and strict shared identity, the final count remains one.
    a=_node('a',0);b=_node('b',1)
    result=_count([a,b],monkeypatch,{frozenset(('a','b')):'shared_identity_supported'})
    assert result['significant_propagation_activity_count_range'] == [1,1]


# T4 -------------------------------------------------------------------------
def test_T4_unresolved_one_to_one_is_range_not_forced_double_count(monkeypatch):
    a=_node('a',0);b=_node('b',1)
    result=_count([a,b],monkeypatch,{frozenset(('a','b')):'unresolved'})
    assert result['significant_propagation_activity_count_v1_legacy'] == 2
    assert result['significant_propagation_activity_count_range'] == [1,2]
    assert result['significant_propagation_activity_count_v2_exact'] is None
    assert result['ambiguous_cross_wall_propagation_pairs'] == 1
    assert result['resolved_propagation_activity_count'] == 0


# T5 -------------------------------------------------------------------------
def test_T5_sub12s_fragments_are_excluded_from_frequency_denominator():
    maps=_maps(60.0)
    t=maps['interval_center_s']
    keep=((t<30.0) |
          ((t>=32.0)&(t<40.0)) |
          ((t>=42.0)&(t<50.0)) |
          ((t>=52.0)&(t<60.0)))
    maps['valid_mask'][:] = keep[:,None,None]
    exposure=observation_union(maps,min_run_s=12.0)
    assert exposure['raw_seconds'] == pytest.approx(54.0,abs=0.13)
    assert exposure['seconds'] == pytest.approx(30.0,abs=0.13)
    assert exposure['minimum_run_samples'] == 100  # ceil(12 / 0.12)


# T6 -------------------------------------------------------------------------
def test_T6_one_to_two_cross_wall_relation_returns_2_to_3_not_three_exact(monkeypatch):
    anterior=_node('a',0)
    posterior1=_node('b1',1)
    posterior2=_node('b2',1,start_s=10.5)
    relations={frozenset(('a','b1')):'unresolved',
               frozenset(('a','b2')):'unresolved'}
    result=_count([anterior,posterior1,posterior2],monkeypatch,relations)
    assert result['significant_propagation_activity_count_v1_legacy'] == 3
    assert result['significant_propagation_activity_count_range'] == [2,3]
    assert result['ambiguous_cross_wall_propagation_pairs'] == 1


# T7 -------------------------------------------------------------------------
def test_T7_two_independent_events_remain_two_not_transitively_merged(monkeypatch):
    a1=_node('a1',0,start_s=10);b1=_node('b1',1,start_s=10)
    a2=_node('a2',0,start_s=18);b2=_node('b2',1,start_s=18)
    relations={frozenset(('a1','b1')):'shared_identity_supported',
               frozenset(('a2','b2')):'shared_identity_supported'}
    result=_count([a1,b1,a2,b2],monkeypatch,relations)
    assert result['significant_propagation_activity_count_range'] == [2,2]
    assert result['significant_propagation_activity_count_v2_exact'] == 2


# Extra guard for the subtle matching bug ------------------------------------
def test_nonpropagation_neighbor_cannot_consume_propagation_matching_slot(monkeypatch):
    a=_node('a',0,propagates=True)
    b_noise=_node('b_noise',1,propagates=False)
    b_prop=_node('b_prop',1,propagates=True)
    relations={frozenset(('a','b_noise')):'unresolved',
               frozenset(('a','b_prop')):'unresolved'}
    result=_count([a,b_noise,b_prop],monkeypatch,relations)
    # Only a<->b_prop belongs in the propagation ambiguity graph.
    assert result['significant_propagation_activity_count_range'] == [1,2]
    assert result['ambiguous_cross_wall_propagation_pairs'] == 1


# T8 -------------------------------------------------------------------------
def test_T8_global_minmax_is_fragile_to_large_filtered_artifact():
    # This is intentionally a Step-2 baseline test, not a Step-1 fix.
    # It injects the artifact AFTER filtering to represent a low-frequency
    # artifact that survived the 0.5--5 cpm band-pass.
    filtered=np.zeros((100,2,12),float)
    valid=np.ones_like(filtered,bool)
    filtered[20:30,0,4:8]=1.0  # moderate true activity
    normalized_clean,_=normalize_magnitude(filtered,valid)
    clean_peak=float(np.nanmax(normalized_clean[20:30,0,4:8]))

    contaminated=filtered.copy()
    contaminated[70:80,1,6]=10.0  # surviving low-frequency artifact cluster/peak
    normalized_artifact,_=normalize_magnitude(contaminated,valid)
    contaminated_peak=float(np.nanmax(normalized_artifact[20:30,0,4:8]))

    assert clean_peak > 0.6
    assert contaminated_peak < 0.6
    # The test documents the current Min-Max weakness; Step 1 must not change it.
