from copy import deepcopy
import unittest
import numpy as np

from experiments.provenance_count_trial import provenance_nodes, trial_count
from tests.test_direction_identity_trial import node


def candidate(name, wall=0, kind='paired_phase', **kwargs):
    value = node(name, wall, **kwargs)
    value['evidence_kind'] = kind
    if kind == 'single_phase':
        for c in value['sections']:
            value['profile'][c] = value['profile'][c][:1]
    return value


class ProvenanceCountTests(unittest.TestCase):
    def test_canonical_evidence_kind_is_carried_instead_of_inferred(self):
        n = node('a', 0)
        entity = dict(canonical_measurement=dict(evidence_kind='single_phase'),
                      source_kind='existing', profile=n['profile'])
        event = dict(event_proposal_id='p', canonical_source_entity_id='base:p',
            supporting_sections=n['sections'], arrival_time_s=list(n['arrival'].values()),
            direction='C2F', canonical_measurement=dict(classification='C2F', arrival_marker='phase_peak'),
            ownership_status='retained_existing', source_entity_ids=['base:p'],
            source_parent_ids=['p'], phase_ridge_provenance=[dict(parent_id='p', phase_ridge_ids=[1])],
            preserved_original_measurements=[dict(evidence_kind='single_phase')])
        ownership = dict(walls=[dict(reconciliation=dict(event_proposals=[event],
                                                        source_evidence={'base:p': entity}))])
        salience = dict(walls=[dict(wall=0, scores=[dict(event_proposal_id='p',
            decision='salient_support', retained_sections=n['sections'])])])
        nodes = provenance_nodes(salience, ownership)
        self.assertEqual(nodes[0]['evidence_kind'], 'single_phase')
        self.assertEqual(nodes[0]['provenance']['source_evidence_kinds'], ['single_phase'])
        self.assertEqual(trial_count(nodes, np.arange(3), 2., .1, 60.)['primary_candidate_row_count'], 0)

    def test_single_evidence_retained_without_becoming_primary(self):
        n = candidate('a', kind='single_phase')
        original = deepcopy(n)
        r = trial_count([n], np.arange(3), 2., .1, 60.)
        self.assertEqual(r['baseline']['significant_propagation_activity_count'], 1)
        self.assertEqual(r['primary_candidate_row_count'], 0)
        self.assertEqual(r['deferred_evidence'][0]['profile'], original['profile'])
        self.assertEqual(n, original)
        self.assertIsNone(r['event_count'])

    def test_unique_paired_opposite_direction_labels_still_one(self):
        r = trial_count([candidate('a'), candidate('b', 1, direction='F2C')], np.arange(3), 2., .1, 60.)
        self.assertEqual(r['primary_candidate_row_count'], 1)
        self.assertEqual(r['identity_range']['conditional_candidate_identity_count_range'], [1, 1])

    def test_one_to_many_range_uses_matching_not_number_of_edges(self):
        r = trial_count([candidate('a'), candidate('b', 1), candidate('c', 1)], np.arange(3), 2., .1, 60.)
        self.assertEqual(r['primary_candidate_row_count'], 3)
        self.assertEqual(r['identity_range']['conditional_candidate_identity_count_range'], [2, 3])
        self.assertTrue(r['identity_range']['identity_remains_unresolved'])

    def test_complementary_sections_not_forced_to_match(self):
        a, b = candidate('a'), candidate('b', 1)
        b['sections'] = [3, 4, 5]
        b['arrival'] = {k+3: v for k, v in b['arrival'].items()}
        b['profile'] = {k+3: v for k, v in b['profile'].items()}
        r = trial_count([a, b], np.arange(6), 2., .1, 60.)
        q = r['identity_range']
        self.assertEqual(q['conditional_candidate_identity_count_range'], [2, 2])
        self.assertTrue(q['identity_remains_unresolved'])
        self.assertFalse(q['range_is_global_identity_bound'])
        self.assertEqual(len(q['no_shared_section_pairs']), 1)


if __name__ == '__main__':
    unittest.main()

