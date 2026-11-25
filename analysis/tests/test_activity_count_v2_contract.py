"""Request-specific coverage beyond the supplied patch tests; synthetic inputs only."""
from copy import deepcopy
from types import SimpleNamespace
import json

import numpy as np
import pytest

from analysis.tests.test_activity_count_v2_step1 import _maps, _node, _count
from analysis.mse_pregnancy_association.code import batch_wave_counts as batch
from analysis.mse_pregnancy_association.code.patient_activity_aggregation import observation_union
from analysis.mse_pregnancy_association.code.activity_strength import filtered_runs
from analysis.mse_pregnancy_association.code.measurement.io import jsonable
from analysis.mse_pregnancy_association.code.activity_counting import count_activities


def test_T7_two_by_two_unresolved_has_maximum_matching_two(monkeypatch):
    nodes = [_node('a1', 0), _node('a2', 0), _node('b1', 1), _node('b2', 1)]
    relations = {frozenset((a, b)): 'unresolved'
                 for a in ('a1', 'a2') for b in ('b1', 'b2')}
    result = _count(nodes, monkeypatch, relations)
    assert result[batch.COUNT + '_range'] == [2, 4]
    assert result['ambiguous_cross_wall_propagation_pairs'] == 2
    assert result['ambiguous_propagation_node_count'] == 4
    assert result[batch.COUNT + '_exact'] is None


def test_matching_reassigns_and_does_not_use_greedy_pairing(monkeypatch):
    nodes = [_node('a1', 0), _node('a2', 0), _node('b1', 1), _node('b2', 1)]
    relations = {frozenset(pair): 'unresolved'
                 for pair in [('a1', 'b1'), ('a1', 'b2'), ('a2', 'b1')]}
    assert _count(nodes, monkeypatch, relations)[batch.COUNT + '_range'] == [2, 4]


def test_T8_mixed_nodes_never_consume_propagation_matching(monkeypatch):
    nodes = [_node('a', 0), _node('b0_noise', 1, propagates=False), _node('b1', 1)]
    relations = {frozenset(('a', b)): 'unresolved' for b in ('b0_noise', 'b1')}
    result = _count(nodes, monkeypatch, relations)
    assert result[batch.COUNT + '_range'] == [1, 2]
    assert result['ambiguous_propagation_node_count'] == 2
    assert result['ambiguous_propagation_fraction'] == 1


def test_ambiguity_fraction_uses_all_propagation_nodes(monkeypatch):
    nodes = [_node('a', 0), _node('b', 1), _node('c', 0), _node('d', 1), _node('e', 0)]
    result = _count(nodes, monkeypatch, {frozenset(('a', 'b')): 'unresolved',
                                       frozenset(('c', 'd')): 'shared_identity_supported'})
    assert result['ambiguous_propagation_fraction'] == pytest.approx(2/5)
    assert result['resolved_propagation_activity_count'] == 2
    assert result['legacy_significant_propagation_activity_count'] == result[batch.COUNT] == 4
    assert _count([], monkeypatch, {})['ambiguous_propagation_fraction'] is None


def test_T3_real_identity_with_partial_section_overlap():
    from tests.test_direction_identity_trial import node
    a, b = node('a', 0), node('b', 1)
    a['sections'] += [3]
    a['arrival'][3] = 13
    a['profile'][3] = deepcopy(a['profile'][2])
    for phase in a['profile'][3]:
        phase['peak_s'] += 1
    result = count_activities([a, b], np.arange(4), 2, .1, 60)
    assert result[batch.COUNT + '_range'] == [1, 1]
    assert result['identity_comparisons'][0]['matched_sections'] == [0, 1, 2]


def test_T5_frequency_uses_filtered_30_seconds(monkeypatch):
    maps = _maps()
    t = maps['interval_center_s']
    keep = (t < 30) | ((t >= 32) & (t < 40)) | ((t >= 42) & (t < 50)) | (t >= 52)
    maps['valid_mask'][:] = keep[:, None, None]
    exposure = observation_union(maps)
    result = _count([_node('a', 0)], monkeypatch, {}, exposure['seconds'])
    assert result[batch.RATE + '_range'] == pytest.approx([2, 2])
    assert exposure['filtered_observation_fraction'] == pytest.approx(30/54, abs=.003)


def test_support_is_per_section_before_spatial_overlap_and_wall_union():
    maps = _maps(duration_s=40, dt=1, sections=4)
    maps['valid_mask'][:] = False
    # Each native section has >=12 consecutive samples, with only 6 in common.
    for section, first in enumerate((0, 3, 6)):
        maps['valid_mask'][first:first+12, 0, section] = True
    maps['valid_mask'][20:35, 1, :3] = True
    filtered = filtered_runs(maps['A'], maps['valid_mask'], maps['interval_center_s'])
    finite = np.isfinite(filtered)
    supported = (finite[:, :, :-2] & finite[:, :, 1:-1] & finite[:, :, 2:]).any((1, 2))
    result = observation_union(maps)
    assert result['seconds'] == supported.sum() == 21
    assert result['intervals_s'] == [[6, 12], [20, 35]]
    assert result['raw_observation_seconds'] == 21


def test_overlapping_measurement_intervals_union_and_gap():
    maps = _maps(duration_s=30, dt=1, sections=3)
    maps['interval_start_s'] = np.r_[np.arange(15), np.arange(15)+25].astype(float)
    maps['interval_end_s'] = maps['interval_start_s']+2
    maps['interval_center_s'] = maps['interval_start_s']+1
    result = observation_union(maps)
    assert result['seconds'] == 32
    assert result['intervals_s'] == [[0, 16], [25, 41]]


@pytest.mark.parametrize('support', ['missing', 'short', 'nonadjacent', 'full'])
def test_batch_observability_and_schema(support):
    maps = _maps(duration_s=30, dt=.25, sections=5)
    if support != 'full':
        maps['valid_mask'][:] = False
    if support == 'short':
        maps['valid_mask'][:20] = True
    if support == 'nonadjacent':
        maps['valid_mask'][:, :, ::2] = True
    result = batch.count_map(maps, .00075)
    assert set(batch.V2_FIELDS) <= result.keys()
    assert result['filtered_observation_seconds'] == (30 if support == 'full' else 0)
    assert result[batch.COUNT] == (0 if support == 'full' else None)
    assert result['reviewed'] is result['frozen_count_algorithm'] is False
    if support == 'missing':
        assert result['filtered_observation_fraction'] is None


@pytest.mark.parametrize('supported', [True, False])
def test_synthetic_batch_exports_v2_and_reads_id_column_only(tmp_path, monkeypatch, supported):
    features, output = tmp_path/'features', tmp_path/'output'
    features.mkdir()
    (features/'patient_results.csv').write_text('case_id\nsynthetic\n', encoding='utf-8')
    (features/'synthetic').mkdir()
    maps = _maps(duration_s=30, dt=.25)
    maps['valid_mask'][:] = supported
    np.savez(features/'synthetic'/'spacetime_2s.npz', **maps)
    protocol = tmp_path/'noise.json'
    protocol.write_text('{"noise_floor_rate": 0.00075}', encoding='utf-8')
    read_csv = batch.pd.read_csv
    def id_only(*args, **kwargs):
        assert kwargs['usecols'] == ['case_id']
        return read_csv(*args, **kwargs)
    monkeypatch.setattr(batch.pd, 'read_csv', id_only)
    batch.run(SimpleNamespace(output=output, features=features, noise_protocol=protocol,
                             case_id=None, workers=1))
    table = read_csv(output/'wave_counts.csv')
    assert set(batch.V2_FIELDS) <= set(table.columns)
    assert (output/'wave_counts.xlsx').exists()
    summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    assert summary['total_significant_propagation_activity_count_range'] == ([0, 0] if supported else None)
    result = json.loads((output/'synthetic.json').read_text(encoding='utf-8'))
    assert result[batch.COUNT + '_range'] == ([0, 0] if supported else None)
    assert result['count_algorithm_version'] == batch.VERSION


def test_upstream_and_v1_ledger_equal_legacy_implementation():
    from experiments.batch_wave_counts import count_map as legacy_count_map
    from experiments.activity_strength import filtered_runs as legacy_filtered
    from experiments.normalized_activity import normalize_magnitude as legacy_normalize
    from analysis.mse_pregnancy_association.code.normalized_activity import normalize_magnitude
    maps = _maps(duration_s=90, dt=.25, sections=6)
    t = maps['interval_center_s']
    for wall in (0, 1):
        for section in range(6):
            maps['A'][:, wall, section] = .01*np.sin(2*np.pi*(t-section)/20)
    original = deepcopy(maps)
    valid = maps['valid_mask']
    filtered = filtered_runs(maps['A'], valid, t)
    expected = legacy_filtered(maps['A'], valid, t)
    np.testing.assert_array_equal(filtered, expected)
    np.testing.assert_array_equal(normalize_magnitude(filtered, valid)[0], legacy_normalize(expected, valid)[0])
    before = legacy_count_map(maps, .00075)
    after = batch.count_map(maps, .00075)
    assert len(after['activities']) > 0  # Ensure the candidate/split/classifier comparison is exercised.
    for key in ('walls', 'activities', 'identity_comparisons', 'normalization', batch.COUNT):
        assert json.dumps(jsonable(after[key]), sort_keys=True) == json.dumps(jsonable(before[key]), sort_keys=True)
    for key in original:
        np.testing.assert_array_equal(maps[key], original[key])
