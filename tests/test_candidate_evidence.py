"""Review diagnostics preserve weak/periodic evidence, native gaps and identity."""
from copy import deepcopy
import json
import unittest
import numpy as np
from uterine_wave.candidate_evidence import analyze_candidate_evidence
from uterine_wave.radial_events import detect_radial_events
from uterine_wave.io import jsonable


class CandidateEvidence(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(751)*.12
        self.s = np.linspace(.2, .8, 17)

    def pulse(self):
        x = self.t[:, None]-(25+(self.s-.5)/.04)[None]
        return -.03*x/9*np.exp(-.5*(x/3)**2)

    def source(self, y, mask=None):
        valid = np.ones_like(y, bool) if mask is None else mask
        walls, filtered = [], []
        for wall in (0, 1):
            row = detect_radial_events(y, valid, self.t, self.s, noise_floor=.001)
            filtered.append(row.pop('filtered_A'))
            for event in row['events']: event['event_id'] = f'wall{wall}_'+event['event_id']
            walls.append(dict(wall=wall, **row))
        values = np.stack(filtered, axis=1)
        arrays = dict(filtered_A=values, candidate_valid_mask=np.stack([valid, valid], axis=1)&np.isfinite(values),
                      interval_center_s=self.t.copy(), s=self.s.copy())
        return dict(walls=walls), arrays

    def test_known_pulse_has_continuous_shapes_without_confirming_or_merging(self):
        report, arrays = self.source(self.pulse())
        before = json.dumps(jsonable(report), sort_keys=True)
        original = deepcopy(arrays)
        result = analyze_candidate_evidence(report, arrays)
        for event in result['events']:
            self.assertGreater(event['activity']['activity_to_context_rms_ratio'], 2.)
            self.assertGreater(event['continuity']['aligned_correlation']['mean'], .98)
            self.assertEqual(event['continuity']['reversed_adjacent_steps'], 0)
            self.assertFalse(event['confirmed_event'])
        self.assertIsNone(result['physiological_event_count'])
        self.assertEqual(json.dumps(jsonable(report), sort_keys=True), before)
        for key in arrays: np.testing.assert_array_equal(arrays[key], original[key])

    def test_regular_real_wave_does_not_require_an_outlier_context_ratio(self):
        y = .006*np.cos(2*np.pi*(self.t[:, None]/30-self.s[None]))
        report, arrays = self.source(y)
        result = analyze_candidate_evidence(report, arrays)
        ratios = [e['activity']['activity_to_context_rms_ratio'] for e in result['events']]
        self.assertAlmostEqual(float(np.median(ratios)), 1., delta=.05)
        self.assertEqual(len(result['events']), sum(len(w['events']) for w in report['walls']))
        self.assertTrue(all(not e['activity']['context_is_noise_reference'] for e in result['events']))

    def test_context_does_not_cross_a_real_missing_interval(self):
        y = .006*np.cos(2*np.pi*(self.t[:, None]/30-self.s[None]))
        valid = np.ones_like(y, bool)
        valid[(self.t >= 42)&(self.t <= 47)] = False
        report, arrays = self.source(y, valid)
        result = analyze_candidate_evidence(report, arrays)
        for event in result['events']:
            for section in event['activity']['per_section']:
                self.assertTrue(section['same_run_end_s'] < 42 or section['same_run_start_s'] > 47)
        self.assertTrue(np.isnan(arrays['filtered_A'][(self.t >= 42)&(self.t <= 47)]).all())

    def test_same_wall_phase_reuse_is_reported_but_cross_wall_ids_are_independent(self):
        report, arrays = self.source(self.pulse())
        result = analyze_candidate_evidence(report, arrays)
        self.assertEqual(result['exact_same_wall_phase_reuses'], 0)
        cross = [r for r in result['relations'] if not r['same_wall']]
        self.assertTrue(cross)
        self.assertEqual(cross[0]['shared_phase_ridge_ids'], [])
        self.assertEqual(cross[0]['mean_arrival_lag_s'], 0.)
        self.assertFalse(cross[0]['same_event_confirmed'])
        duplicate = deepcopy(report['walls'][0]['events'][0])
        duplicate['event_id'] = 'duplicate_demo'
        report['walls'][0]['events'].append(duplicate)
        changed = analyze_candidate_evidence(report, arrays)
        self.assertEqual(changed['exact_same_wall_phase_reuses'], 1)
        self.assertIsNone(changed['physiological_event_count'])

    def test_different_arrival_markers_cannot_be_compared_as_same_time(self):
        report, arrays = self.source(self.pulse())
        report['walls'][1]['events'][0]['arrival_marker'] = 'phase_peak'
        relation = analyze_candidate_evidence(report, arrays)['relations'][0]
        self.assertFalse(relation['arrival_markers_comparable'])
        self.assertIsNone(relation['mean_arrival_lag_s'])

    def test_empty_static_and_misaligned_sources_are_explicit(self):
        report, arrays = self.source(np.zeros((len(self.t), len(self.s))))
        self.assertEqual(analyze_candidate_evidence(report, arrays)['events'], [])
        report, arrays = self.source(self.pulse())
        report['walls'][0]['phase_ridges'][0]['peak_time_s'][0] += .03
        with self.assertRaises(ValueError): analyze_candidate_evidence(report, arrays)


if __name__ == '__main__': unittest.main()
