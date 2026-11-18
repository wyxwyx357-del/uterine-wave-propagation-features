import unittest
import numpy as np
from experiments.partial_phase_events import observed_lobes, detect_partial, propose_groups, anchor_to_existing_candidates
from uterine_wave import radial_events


class PartialPhaseEvents(unittest.TestCase):
    def test_unknown_end_has_finite_observed_support_and_no_formal_count(self):
        t = np.arange(501)*.12
        s = np.array([.2, .4, .6])
        y = np.exp(-.5*((t-42)/6)**2)[:, None]*np.ones((1, 3))
        from unittest.mock import patch
        with patch.object(radial_events, 'band_limited_signal', lambda values, times:values):
            result = detect_partial(y, np.ones_like(y, bool), t, s, noise_floor=.001)
        fragment = result['fragments'][0]
        self.assertTrue(np.isfinite(fragment['observed_end_s']))
        self.assertIsNone(fragment['structure_end_s'])
        self.assertTrue(fragment['boundary_status']['end_unknown'])
        self.assertFalse(fragment['reviewed'])
        self.assertIsNone(result['formal_patient_output']['event_count'])

    def test_support_repairs_existing_ownership_without_promoting_a_novel_group(self):
        original = dict(phase_ridges=[
            dict(ridge_id=0, sign=1, sections=list(range(7)), peak_time_s=list(range(10, 17))),
            dict(ridge_id=1, sign=-1, sections=[0, 1, 2], peak_time_s=[20, 21, 22]),
            dict(ridge_id=2, sign=-1, sections=[4, 5, 6], peak_time_s=[24, 25, 26])], events=[
            dict(event_id='part', phase_ridge_ids=[1], sections=[0, 1, 2], classification='C2F'),
            dict(event_id='main', phase_ridge_ids=[0, 2], sections=[4, 5, 6], classification='C2F')])
        observed = dict(phase_ridges=[original['phase_ridges'][0],
            dict(ridge_id=1, sign=-1, sections=list(range(7)), peak_time_s=list(range(20, 27))),
            dict(ridge_id=2, sign=1, sections=[0, 1, 2], peak_time_s=[40, 41, 42])], events=[
            dict(event_id='joined', phase_ridge_ids=[0, 1], sections=list(range(7))),
            dict(event_id='new', phase_ridge_ids=[2], sections=[0, 1, 2])], fragments=[
            dict(fragment_id='joined', phase_ridge_ids=[0, 1], sections=list(range(7))),
            dict(fragment_id='new', phase_ridge_ids=[2], sections=[0, 1, 2])], proposed_events=[
            dict(proposal_id='joined', fragment_ids=['joined']),
            dict(proposal_id='new', fragment_ids=['new'])])
        result = anchor_to_existing_candidates(original, observed)
        self.assertEqual(len(result['existing_candidate_groups']), 1)
        self.assertEqual(result['existing_candidate_groups'][0]['source_candidate_ids'], ['part', 'main'])
        self.assertEqual(result['additional_support_groups'][0]['proposal_id'], 'new')
        self.assertIsNone(result['formal_event_count'])
        original['events'][0]['classification'] = 'F2C'
        conflicted = anchor_to_existing_candidates(original, observed)
        self.assertEqual(len(conflicted['existing_candidate_groups']), 2)

    def test_interior_boundary_peak_is_retained_without_inventing_its_end(self):
        t = np.arange(501)*.12
        y = np.exp(-.5*((t-42)/6)**2)[:, None]*np.ones((1, 3))
        # Use the sampled phase directly to isolate boundary extraction semantics.
        from unittest.mock import patch
        with patch.object(radial_events, 'band_limited_signal', lambda values, times:values):
            _, columns, unresolved = observed_lobes(y, np.ones_like(y, bool), t, np.full(3, .001))
        self.assertEqual(len(columns[0]), 1)
        self.assertTrue(columns[0][0]['right_censored'])
        self.assertTrue(columns[0][0]['peak_observed'])
        self.assertEqual(unresolved, [])

    def test_monotone_boundary_segment_is_unresolved_not_a_fabricated_peak(self):
        t = np.arange(201)*.12
        y = np.linspace(0, 1, len(t))[:, None]
        from unittest.mock import patch
        with patch.object(radial_events, 'band_limited_signal', lambda values, times:values):
            _, columns, unresolved = observed_lobes(y, np.ones_like(y, bool), t, np.array([.001]))
        self.assertEqual(columns[0], [])
        self.assertFalse(unresolved[0]['peak_observed'])

    def test_disjoint_complementary_fragments_get_only_a_proposed_shared_identity(self):
        single = dict(event_id='fragment', evidence_kind='single_phase', phase_ridge_ids=[1],
                      sections=[0, 1, 2], classification='F2C')
        parent = dict(event_id='parent', evidence_kind='paired_phase', phase_ridge_ids=[0, 2],
                      sections=[4, 5, 6], classification='F2C', leading_polarity=1)
        edge = dict(selected=False, phase_ridge_ids=[0, 1], sections=[0, 1, 2],
                    classification='F2C', leading_polarity=1)
        groups, attached, unresolved = propose_groups(dict(events=[single, parent], alternative_phase_pairs=[edge]))
        self.assertEqual(groups, {'parent':['fragment', 'parent']})
        self.assertIn('fragment', attached)
        self.assertEqual(unresolved, [])
        edge['leading_polarity'] = -1
        groups, attached, _ = propose_groups(dict(events=[single, parent], alternative_phase_pairs=[edge]))
        self.assertEqual(len(groups), 2)
        self.assertFalse(attached)

    def test_native_missing_support_is_not_joined_and_default_builder_is_restored(self):
        t = np.arange(751)*.12
        s = np.linspace(.2, .8, 17)
        x = t[:, None]-(25+(s-.5)/.04)[None]
        y = -.03*x/9*np.exp(-.5*(x/3)**2)
        valid = np.ones_like(y, bool)
        valid[(t >= 22)&(t <= 27)] = False
        original = radial_events._lobes
        report = detect_partial(y, valid, t, s, noise_floor=.001)
        self.assertIs(radial_events._lobes, original)
        self.assertTrue(np.isnan(report['filtered_A'][(t >= 22)&(t <= 27)]).all())
        for fragment in report['fragments']:
            self.assertFalse(fragment['observed_start_s'] < 22 and fragment['observed_end_s'] > 27)
        self.assertIsNone(report['formal_patient_output']['event_count'])


if __name__ == '__main__':
    unittest.main()
