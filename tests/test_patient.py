import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from uterine_wave.events import measure_event
from uterine_wave.patient import _duration, summarize_patient


def maps():
    t = np.arange(601)*.1
    s = np.linspace(0, 1, 11)
    a = .01*np.cos(2*np.pi*(t[:, None]/30-s[None]))
    a = np.stack((a, a), axis=1)
    return dict(A=a, epsilon=2*a, valid_mask=np.ones_like(a, bool),
                interval_center_s=t, interval_dt_s=np.full(len(t), 2.), s=s,
                nominal_interval_s=2.)


def event(wall=0, direction=1):
    return dict(event_id='one', wall=wall, reviewed=True, start_s=10., end_s=30.,
                sections=[3, 4, 5, 6, 7],
                arrival_time_s=(20+direction*20*(np.arange(3, 8)/10-.5)).tolist())


class PatientFeatures(unittest.TestCase):
    def test_amplitude_measured_per_pair_without_phase_cancellation(self):
        result = summarize_patient(maps(), s_range=(0, 1))
        self.assertAlmostEqual(result['walls'][0]['radial_stretch_activity_amplitude'],
                               .02/np.sqrt(2), delta=.0001)
        self.assertEqual(result['walls'][0]['requested_sections'], 11)
        self.assertIsNone(result['event_summary']['event_count'])

    def test_two_walls_count_one_event_and_arrival_velocity(self):
        data = maps()
        front, back = event(), event(1)
        back['arrival_time_s'] = [x+1 for x in back['arrival_time_s']]
        review = dict(complete=True, intervals=[[0, 60]], events=[front, back])
        result = summarize_patient(data, s_range=(.2, .8), review=review)
        self.assertEqual(result['event_summary']['event_count'], 1)
        self.assertAlmostEqual(result['event_summary']['effective_duration_s'], 60.)
        self.assertAlmostEqual(result['event_summary']['event_frequency_cpm'], 1.)
        self.assertAlmostEqual(result['events'][0]['velocity_paths_per_s'], .05)
        self.assertEqual(result['events'][0]['direction'], 'C2F')
        self.assertEqual(result['paired_wall_events'][0]['mean_arrival_difference_s'], 1.)
        self.assertTrue(all(row['first_section'] >= 3 and row['last_section'] <= 9
                            for row in result['modes']['paired_ER_windows']))

    def test_reverse_complex_and_synchronous_arrival_rules(self):
        data = maps()
        reverse = measure_event(data, event(direction=-1))
        self.assertEqual(reverse['direction'], 'F2C')
        self.assertAlmostEqual(reverse['velocity_paths_per_s'], -.05)
        item = event();item['arrival_time_s'] = [20, 18, 16, 18, 20]
        self.assertEqual(measure_event(data, item)['direction'], 'complex')
        item['arrival_time_s'] = [20]*5
        self.assertIsNone(measure_event(data, item)['velocity_paths_per_s'])
        reference = event()
        reference['arrival_time_s'] = [20, 18, 16, 18, 20]
        paired = dict(reference, wall=1)
        result = summarize_patient(data, s_range=(.2, .8),
            review=dict(complete=True, intervals=[[0, 60]], events=[reference, paired]))
        self.assertEqual(result['event_summary']['event_count'], 1)
        self.assertEqual(result['event_summary']['direction_proportions']['complex'], 1.)
        self.assertEqual(len(result['paired_wall_events']), 1)
        data['A'][:] = 0;data['epsilon'][:] = 0
        self.assertEqual(measure_event(data, event())['status'], 'no_resolved_propagation')

    def test_segment_gap_does_not_become_zero_count_or_joined_amplitude(self):
        data = maps()
        data['valid_mask'][200:250] = False
        data['epsilon'][250:] += 5  # New segment baseline is irrelevant to within-run SD.
        result = summarize_patient(data, s_range=(.2, .8),
            review=dict(complete=True, intervals=[[0, 60]], events=[event()]))
        self.assertIsNone(result['event_summary']['event_count'])
        self.assertIsNone(result['event_summary']['event_frequency_cpm'])
        self.assertEqual(result['events'][0]['status'], 'outside_fixed_observation_support')
        self.assertLess(result['walls'][0]['radial_stretch_activity_amplitude'], .03)
        self.assertEqual(len(result['walls'][0]['per_pair'][0]['blocks']), 2)
        self.assertLess(result['event_summary']['effective_duration_s'], 60)

    def test_explicit_zero_review_and_partial_review(self):
        data = maps()
        for complete, expected in [(True, 0), (False, None)]:
            result = summarize_patient(data, s_range=(.2, .8),
                review=dict(complete=complete, intervals=[[0, 60]], events=[]))
            self.assertEqual(result['event_summary']['event_count'], expected)
        with self.assertRaises(ValueError):
            summarize_patient(data, s_range=(.2, .8), review=dict(complete=True,
                intervals=[[0, 60]], events=[event(), event()]))

    def test_cli_roundtrip(self):
        from uterine_wave.cli import main
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez(root/'maps.npz', **maps())
            main(['summarize', '--maps', str(root/'maps.npz'), '--s-range', '.2', '.8',
                  '--output', str(root/'summary.json')])
            result = json.loads((root/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(result['nominal_interval_s'], 2.)
            self.assertIsNone(result['event_summary']['event_frequency_cpm'])

    def test_fixed_time_and_region_exclude_unrequested_activity(self):
        data = maps()
        baseline = summarize_patient(data, s_range=(.3, .7), time_range=(0, 30))
        data['epsilon'][301:] *= 50
        data['epsilon'][:, :, :3] *= 100
        result = summarize_patient(data, s_range=(.3, .7), time_range=(0, 30))
        self.assertEqual(result['walls'], baseline['walls'])
        self.assertEqual(result['analyzed_time_range_s'], [1., 29.])
        self.assertEqual(result['analyzed_acquisition_time_range_s'], [0., 30.])

    def test_measurement_intervals_cannot_cross_acquisition_time_bounds(self):
        data = maps()
        clock = data['interval_center_s']
        data['interval_start_s'] = clock-1
        data['interval_end_s'] = clock+1
        # Large changes at centers just inside 10..50 would leak motion
        # from outside that acquisition window under center-only selection.
        border = ((clock >= 10) & (clock < 11)) | ((clock > 49) & (clock <= 50))
        baseline = summarize_patient(data, s_range=(.2, .8), time_range=(10, 50))
        data['epsilon'][border] += 50
        result = summarize_patient(data, s_range=(.2, .8), time_range=(10, 50))
        self.assertEqual(result['walls'], baseline['walls'])
        self.assertEqual(result['analyzed_interval_center_range_s'], [11., 49.])
        self.assertEqual(result['analyzed_acquisition_time_range_s'], [10., 50.])
        review = dict(complete=True, intervals=[[10, 50]], events=[])
        reviewed = summarize_patient(data, s_range=(.2, .8), time_range=(10, 50), review=review)
        self.assertEqual(reviewed['event_summary']['event_count'], 0)
        self.assertAlmostEqual(reviewed['event_summary']['effective_duration_s'], 40.)
        # Older maps have identical behavior from center +/- actual dt/2.
        del data['interval_start_s'];del data['interval_end_s']
        older = summarize_patient(data, s_range=(.2, .8), time_range=(10, 50))
        self.assertEqual(result['walls'], older['walls'])

    def test_missing_sections_cannot_become_a_full_region_primary_amplitude(self):
        data = maps()
        data['valid_mask'][:, 0, 5] = False
        result = summarize_patient(data, s_range=(.2, .8))
        front, back = result['walls']
        self.assertEqual(front['analyzed_sections'], 6)
        self.assertEqual(front['requested_sections'], 7)
        self.assertFalse(front['amplitude_covers_all_requested_sections'])
        self.assertIsNone(front['radial_stretch_activity_amplitude'])
        self.assertIsNotNone(front['partial_region_activity_amplitude_QC'])

        self.assertEqual(front['amplitude_status'], 'insufficient_section_support')
        self.assertIsNotNone(back['radial_stretch_activity_amplitude'])
        self.assertIsNone(back['partial_region_activity_amplitude_QC'])
        # A section with only sub-minimum fragments is also absent from the
        # primary summary, even though many of its individual samples exist.
        data = maps();data['valid_mask'][::100, 0, 5] = False
        front = summarize_patient(data, s_range=(.2, .8))['walls'][0]
        self.assertIsNone(front['radial_stretch_activity_amplitude'])
        self.assertIsNotNone(front['partial_region_activity_amplitude_QC'])

    def test_review_exposure_also_requires_full_interval_containment(self):
        data = maps()
        result = summarize_patient(data, s_range=(.2, .8),
            review=dict(complete=True, intervals=[[10, 50]], events=[]))
        self.assertEqual(result['event_summary']['event_count'], 0)
        self.assertAlmostEqual(result['event_summary']['effective_duration_s'], 40.)

    def test_complete_overlapping_measurements_cover_the_acquisition(self):
        result = summarize_patient(maps(), s_range=(.2, .8), time_range=(0, 60),
            review=dict(complete=True, intervals=[[0, 40], [20, 60]], events=[event()]))
        self.assertEqual([w['common_region_duration_s'] for w in result['walls']], [60., 60.])
        self.assertEqual(result['event_summary']['effective_duration_s'], 60.)
        self.assertEqual(result['event_summary']['event_frequency_cpm'], 1.)

    def test_supported_interval_union_preserves_a_real_gap(self):
        data = maps()
        clock = data['interval_center_s']
        data['valid_mask'][(clock > 19) & (clock < 31)] = False
        result = summarize_patient(data, s_range=(.2, .8), time_range=(0, 60),
            review=dict(complete=True, intervals=[[0, 60]], events=[]))
        # Valid intervals cover [0,20] and [30,60]. The ten-second gap
        # contributes nothing, and two walls do not double the denominator.
        self.assertEqual([w['common_region_duration_s'] for w in result['walls']], [50., 50.])
        self.assertEqual(result['event_summary']['effective_duration_s'], 50.)

    def test_interval_union_handles_order_duplicates_and_empty_selection(self):
        start = np.array([10., 0., 1., 0., 30.])
        end = np.array([12., 2., 3., 2., 31.])
        self.assertEqual(_duration(np.array([True]*4+[False]), start, end), 5.)
        self.assertEqual(_duration(np.zeros(5, bool), start, end), 0.)


if __name__ == '__main__':
    unittest.main()

