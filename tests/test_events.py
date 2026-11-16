import unittest
import numpy as np

from uterine_wave.event_detection import EventDetectionConfig, detect_events
from uterine_wave.event_features import effective_observation_seconds, event_rate_per_minute
from uterine_wave.coordination import match_bilateral_events
from uterine_wave.patient_features import aggregate_patient_features


def traveling_map(direction="C2F", *, nt=240, ns=21, dt=.12):
    t = np.arange(nt) * dt
    s = np.linspace(0, 1, ns)
    y = 0.05 * np.sin(2*np.pi*t[:, None]/11 + 0.3*s[None, :])
    if direction == "C2F":
        centers = 5.0 + 4.0*s
    elif direction == "F2C":
        centers = 9.0 - 4.0*s
    elif direction == "sync":
        centers = np.full_like(s, 7.0)
    elif direction == "local":
        centers = 7.0 + 2.0*s
    else:
        raise ValueError(direction)
    for j, c in enumerate(centers):
        if direction == "local" and not (0.42 <= s[j] <= 0.58):
            continue
        y[:, j] += 2.5*np.exp(-0.5*((t-c)/0.16)**2)
    return y, np.ones_like(y, bool), t, s


class EventDetectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = EventDetectionConfig(
            z_threshold=2.5,
            min_component_cells=6,
            min_space_points=4,
            min_path_fraction=.15,
            min_fit_r2=.7,
            synchronous_tolerance_s=.5,
        )

    def test_c2f_and_f2c(self):
        for direction in ("C2F", "F2C"):
            y, v, t, s = traveling_map(direction)
            out = detect_events(y, v, t, s, self.cfg)
            found = [e for e in out["events"] if e["classification"] == direction]
            self.assertTrue(found, (direction, out["events"]))
            self.assertGreater(found[0]["path_fraction"], .8)
            self.assertGreater(found[0]["fit_r2"], .9)

    def test_synchronous(self):
        y, v, t, s = traveling_map("sync")
        out = detect_events(y, v, t, s, self.cfg)
        self.assertTrue(any(e["classification"] == "synchronous" for e in out["events"]))

    def test_local(self):
        y, v, t, s = traveling_map("local")
        out = detect_events(y, v, t, s, self.cfg)
        self.assertTrue(any(e["classification"] == "local" for e in out["events"]))

    def test_invalid_gap_does_not_get_filled(self):
        y, v, t, s = traveling_map("C2F")
        gap = (t >= 6.8) & (t <= 7.2)
        v[gap] = False
        y[gap] = np.nan
        out = detect_events(y, v, t, s, self.cfg)
        self.assertFalse(out["active_mask"][gap].any())

    def test_coordination_and_patient_features(self):
        y, v, t, s = traveling_map("C2F")
        ant = detect_events(y, v, t, s, self.cfg)["events"]
        y2 = 0.05 * np.sin(2*np.pi*t[:, None]/11 + 0.3*s[None, :])
        centers = 5.3 + 4.0*s
        for j, c in enumerate(centers):
            y2[:, j] += 2.5*np.exp(-0.5*((t-c)/0.16)**2)
        post = detect_events(y2, v, t, s, self.cfg)["events"]
        coord = match_bilateral_events(ant, post, max_midpoint_lag_s=1.0)
        self.assertGreaterEqual(coord["matched_pair_count"], 1)
        self.assertEqual(coord["traveling_direction_agreement_fraction"], 1.0)
        features = aggregate_patient_features(ant, post, v, v, t, coord)
        self.assertGreaterEqual(features["unique_traveling_event_count"], 1)
        self.assertIsNotNone(features["unique_traveling_events_per_min"])

    def test_observation_time_and_rate(self):
        t = np.arange(11, dtype=float)
        v = np.ones((11, 4), bool)
        v[5] = False
        seconds = effective_observation_seconds(v, t, .5)
        self.assertEqual(seconds, 8.0)
        events = [dict(classification="C2F"), dict(classification="local")]
        self.assertAlmostEqual(event_rate_per_minute(events, 60), 1.0)


if __name__ == "__main__":
    unittest.main()
