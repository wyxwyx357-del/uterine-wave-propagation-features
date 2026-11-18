"""Review inputs hide machine judgements and retain native missing points."""
from copy import deepcopy
import json
import unittest
import numpy as np
from experiments.prepare_blind_wave_review import review_items, display_tracks


class BlindWaveReview(unittest.TestCase):
    def test_blind_whitelist_identity_and_fixed_random_order(self):
        cases = []
        for i in range(3):
            events = [dict(event_id=f'e{j}', classification=d, sections=[0, 1],
                           start_s=10+j, end_s=14+j, RMS=123, quality_flags=['bad'])
                      for j, d in enumerate(('C2F', 'F2C', 'complex'))]
            cases.append((f'private_case_{i}', dict(walls=[dict(wall=0, events=events)])))
        before = deepcopy(cases)
        public, key = review_items(cases)
        self.assertEqual(review_items(cases), (public, key))
        self.assertEqual(cases, before)
        self.assertEqual(len(public), 6)
        self.assertEqual(len({x['token'] for x in public}), 6)
        self.assertEqual({(x['review_id'], x['event_id']) for x in key},
                         {(f'private_case_{i}', f'e{j}') for i in range(3) for j in range(2)})
        self.assertNotIn('private_case', json.dumps(public))
        for x in public:
            self.assertEqual(set(x), {'token', 'case_alias', 'wall', 'sections', 'start_s', 'end_s'})
        for x, private in zip(public, key):
            self.assertEqual(x['token'], private['token'])
            original = cases[int(private['review_id'][-1])][1]['walls'][0]['events'][int(private['event_id'][-1])]
            self.assertEqual(x['start_s'], original['start_s'])

    def test_original_frame_time_and_missing_points_are_not_interpolated(self):
        points = np.arange(24, dtype=float).reshape(3, 4, 2)
        valid = np.ones((3, 4), bool)
        valid[1, 2] = False
        tracks = dict(positions=points, corrected=points+100, target_valid=valid,
                      time_s=np.array([0., .12, .24]))
        grid = dict(A_points=np.zeros((4, 2)), A_pair_ids=np.array([[[0, 1]], [[2, 3]]]))
        result = display_tracks(tracks, grid, .5)
        self.assertEqual(result['time_s'], [0., .12, .24])
        self.assertEqual(len(result['frames']), 3)
        self.assertIsNone(result['frames'][1][2])
        self.assertEqual(result['frames'][2][2], (points[2, 2]*2).tolist())
        np.testing.assert_array_equal(tracks['positions'], points)

    def test_nonmonotone_native_timeline_is_reported(self):
        tracks = dict(positions=np.zeros((2, 2, 2)), target_valid=np.ones((2, 2), bool),
                      time_s=np.array([.1, .1]))
        with self.assertRaisesRegex(ValueError, 'frame times'):
            display_tracks(tracks, dict(A_points=np.zeros((2, 2))), .5)


if __name__ == '__main__':
    unittest.main()
