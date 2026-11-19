import unittest

import numpy as np

from experiments.direction_identity_trial import (
    compare_identity_without_direction,
    reconcile_identity_then_direction,
    revalidate_retained_direction,
)


def node(name, wall, direction='C2F', shift=0., arrival=None, extra_phase=False):
    if arrival is None:
        arrival = [10., 11., 12.]
    phases = lambda section: [
        dict(sign=1, peak_s=8.+section+shift, start_s=7.+section,
             end_s=9.+section, epoch=[wall, 1]),
        dict(sign=-1, peak_s=12.+section+shift, start_s=11.+section,
             end_s=13.+section, epoch=[wall, 1])]
    profile = {section: phases(section) for section in range(3)}
    if extra_phase:
        for section in range(3):
            profile[section].append(dict(sign=1, peak_s=20.+section,
                start_s=19.+section, end_s=21.+section, epoch=[wall, 1]))
    return dict(node_id=name, wall=wall, sections=[0, 1, 2], direction=direction,
        arrival_marker='zero', arrival=dict(enumerate(arrival)), profile=profile)


class DirectionIdentityTrial(unittest.TestCase):
    def test_retained_positions_can_expose_direction_conflict(self):
        candidate = node('a', 0, 'F2C', arrival=[10., 11., 12.])
        result = revalidate_retained_direction(candidate, np.arange(3), .1)
        self.assertEqual(result['retained_refit_direction'], 'C2F')
        self.assertEqual(result['direction_status'], 'retained_support_conflicts_with_original')
        self.assertIsNone(result['supported_direction'])

    def test_identity_precedes_opposing_wall_direction(self):
        result = reconcile_identity_then_direction(
            [node('a', 0, 'C2F'), node('b', 1, 'F2C', .1)],
            np.arange(3), 2., .1)
        self.assertEqual(result['activity_rows'], 1)
        self.assertEqual(result['direction_unresolved_rows'], 1)
        self.assertEqual(result['direction_counts'], {'C2F': 0, 'F2C': 0})

    def test_extra_local_phase_does_not_prevent_identity_match(self):
        left = node('a', 0)
        right = node('b', 1, shift=.1, extra_phase=True)
        edge = compare_identity_without_direction(left, right, 2., .1)
        self.assertEqual(edge['relation'], 'shared_identity_supported')

    def test_separate_times_remain_separate(self):
        result = reconcile_identity_then_direction(
            [node('a', 0), node('b', 1, shift=8., arrival=[18., 19., 20.])],
            np.arange(3), 2., .1)
        self.assertEqual(result['activity_rows'], 2)

    def test_one_to_many_identity_competition_is_not_forced(self):
        result = reconcile_identity_then_direction(
            [node('a', 0), node('b', 1, shift=.1), node('c', 1, shift=.2)],
            np.arange(3), 2., .1)
        self.assertEqual(result['activity_rows'], 3)
        self.assertEqual(result['cross_wall_identity_pairs'], 0)


if __name__ == '__main__':
    unittest.main()
