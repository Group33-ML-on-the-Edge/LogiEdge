"""Tests for preprocessing.py. Run with:  python3 -m unittest -v test_preprocessing

The MQTT runtime is tested against a stub of paho-mqtt, so no broker is needed.
"""

import inspect
import json
import os
import sys
import tempfile
import types
import unittest

import numpy as np

import preprocessing as pre


def feed(pipe, seconds, temp_fn=None, vib_fn=None, start=0):
    """Push `seconds` of 1 Hz temperature and 0.5 Hz vibration data; return all windows."""
    temp_fn = temp_fn or (lambda t: 4.0)
    vib_fn = vib_fn or (lambda t: 0.45)
    out = []
    for t in range(start, start + seconds):
        out += pipe.push("temp", float(t), temp_fn(t))
        if t % 2 == 0:
            out += pipe.push("vib", float(t), vib_fn(t))
    return out


class TestFilter(unittest.TestCase):
    def test_five_sample_moving_average(self):
        ma = pre.MovingAverage(5)
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        out = [ma.update(x) for x in data]
        self.assertEqual(out[:4], [1.0, 1.5, 2.0, 2.5])                 # warm-up: mean so far
        self.assertEqual(out[4:], [3.0, 4.0, 5.0, 6.0])                 # steady state: mean of last 5

    def test_matches_numpy_convolution_in_steady_state(self):
        x = np.random.default_rng(1).normal(size=200)
        ma = pre.MovingAverage(5)
        out = np.array([ma.update(v) for v in x])
        ref = np.convolve(x, np.ones(5) / 5, mode="valid")
        np.testing.assert_allclose(out[4:], ref)


class TestFeatures(unittest.TestCase):
    def test_temperature_ramp(self):
        ts = np.arange(30.0)
        x = 4.0 + 0.02 * ts                                            # 1.2 degC/min
        mean, std, rate = pre.temperature_features(ts, x)
        self.assertAlmostEqual(mean, 4.0 + 0.02 * 14.5)
        self.assertAlmostEqual(std, 0.02 * np.std(ts))
        self.assertAlmostEqual(rate, 1.2, places=9)

    def test_temperature_constant(self):
        mean, std, rate = pre.temperature_features(np.arange(30.0), np.full(30, 4.0))
        self.assertEqual((mean, std), (4.0, 0.0))
        self.assertAlmostEqual(rate, 0.0, places=12)

    def test_rate_is_per_minute_not_per_second(self):
        _, _, rate = pre.temperature_features(np.arange(30.0), 0.08 * np.arange(30.0))
        self.assertAlmostEqual(rate, 4.8, places=9)                    # 0.08 degC/s = 4.8 degC/min

    def test_vibration_constant_window(self):
        rms, peak, kurt = pre.vibration_features(np.full(15, 0.45))
        self.assertAlmostEqual(rms, 0.45); self.assertAlmostEqual(peak, 0.45); self.assertEqual(kurt, 0.0)

    def test_vibration_rms_and_peak(self):
        rms, peak, _ = pre.vibration_features(np.array([3.0, -4.0, 0.0, 0.0]))
        self.assertAlmostEqual(rms, np.sqrt(25 / 4)); self.assertEqual(peak, 4.0)

    def test_kurtosis_reference_values(self):
        rng = np.random.default_rng(0)
        self.assertAlmostEqual(pre.vibration_features(rng.normal(size=400000))[2], 0.0, delta=0.03)   # Gaussian
        self.assertAlmostEqual(pre.vibration_features(rng.uniform(size=400000))[2], -1.2, delta=0.03) # uniform
        spiky = np.zeros(1000); spiky[0] = 1.0
        self.assertGreater(pre.vibration_features(spiky)[2], 900)                                   # one spike

    def test_fusion_concatenates_in_documented_order(self):
        v = pre.fuse_features(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        np.testing.assert_array_equal(v, [1, 2, 3, 4, 5, 6])
        self.assertEqual(len(pre.FEATURE_NAMES), 6)
        with self.assertRaises(ValueError):
            pre.fuse_features(np.zeros(2), np.zeros(3))

    def test_sensors_are_extracted_independently(self):
        """Changing only the vibration stream must not change the 3 temperature features, and vice versa."""
        base = feed(pre.PreprocessingPipeline(), 100, lambda t: 4 + 0.01 * t, lambda t: 0.45 + 0.001 * t)
        vib_changed = feed(pre.PreprocessingPipeline(), 100, lambda t: 4 + 0.01 * t, lambda t: 1.2)
        temp_changed = feed(pre.PreprocessingPipeline(), 100, lambda t: 9.0, lambda t: 0.45 + 0.001 * t)
        self.assertEqual(len(base), len(vib_changed))
        for a, b, c in zip(base, vib_changed, temp_changed):
            np.testing.assert_allclose(a.raw[:3], b.raw[:3])
            self.assertFalse(np.allclose(a.raw[3:], b.raw[3:]))
            np.testing.assert_allclose(a.raw[3:], c.raw[3:])
            self.assertFalse(np.allclose(a.raw[:3], c.raw[:3]))


class TestPipeline(unittest.TestCase):
    def test_window_timing_30s_step_10s(self):
        wins = feed(pre.PreprocessingPipeline(), 100)
        self.assertEqual([w.end for w in wins], [30, 40, 50, 60, 70, 80, 90])
        self.assertTrue(all(w.end - w.start == 30 for w in wins))
        self.assertTrue(all(b.start - a.start == 10 for a, b in zip(wins, wins[1:])))

    def test_ten_minutes_gives_57_windows(self):
        self.assertEqual(len(feed(pre.PreprocessingPipeline(), 600)), 57)

    def test_each_window_holds_30_temp_and_15_vib_samples(self):
        for w in feed(pre.PreprocessingPipeline(keep_samples=True), 120):
            self.assertEqual(len(w.temp_samples), 30)
            self.assertEqual(len(w.vib_samples), 15)

    def test_moving_average_is_applied_before_features(self):
        data = np.random.default_rng(3).normal(4, 0.3, 120)
        wins = feed(pre.PreprocessingPipeline(keep_samples=True), 120, temp_fn=lambda t: data[t])
        ma = pre.MovingAverage(5)
        filtered = np.array([ma.update(x) for x in data])
        np.testing.assert_allclose(wins[0].temp_samples, filtered[0:30])
        np.testing.assert_allclose(wins[2].temp_samples, filtered[20:50])
        self.assertAlmostEqual(wins[1].raw[0], filtered[10:40].mean())

    def test_incomplete_window_is_skipped_not_fabricated(self):
        pipe = pre.PreprocessingPipeline()
        wins = []
        for t in range(0, 31):                               # temperature missing for seconds 5-14
            if not 5 <= t <= 14:
                wins += pipe.push("temp", float(t), 4.0)
            if t % 2 == 0:
                wins += pipe.push("vib", float(t), 0.45)
        self.assertEqual(wins, [])
        self.assertEqual(pipe.skipped_windows, 1)

    def test_duplicate_and_late_messages_are_dropped(self):
        pipe = pre.PreprocessingPipeline()
        pipe.push("temp", 10.0, 4.0)
        pipe.push("temp", 10.0, 9.0)
        pipe.push("temp", 9.0, 9.0)
        self.assertEqual(pipe.dropped_out_of_order, 2)

    def test_unknown_stream_rejected(self):
        with self.assertRaises(ValueError):
            pre.PreprocessingPipeline().push("door", 0.0, 1.0)

    def test_drift_shows_in_rate_of_change(self):
        wins = feed(pre.PreprocessingPipeline(), 100, temp_fn=lambda t: 4.0 + 0.08 * t)
        self.assertAlmostEqual(wins[-1].raw[2], 4.8, delta=0.05)       # MA lag does not change the slope


class TestNormaliser(unittest.TestCase):
    def setUp(self):
        self.mean = np.array([4.0, 0.12, 0.0, 0.445, 0.47, -0.8])
        self.std = np.array([0.055, 0.026, 0.36, 0.0105, 0.0117, 0.53])
        self.n = pre.Normaliser(self.mean, self.std)

    def test_transform(self):
        x = self.mean + 2 * self.std
        np.testing.assert_allclose(self.n.transform(x), np.full(6, 2.0))

    def test_save_load_round_trip_and_shape(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "training_stats.npy")
            np.save(path, np.vstack([self.mean, self.std]))
            loaded = pre.Normaliser.load(path)
            np.testing.assert_array_equal(loaded.mean, self.mean)
            np.testing.assert_array_equal(loaded.std, self.std)
            np.save(path, np.zeros((3, 6)))
            with self.assertRaises(ValueError):
                pre.Normaliser.load(path)

    def test_rejects_bad_statistics(self):
        with self.assertRaises(ValueError):
            pre.Normaliser(self.mean, np.zeros(6))
        with self.assertRaises(ValueError):
            pre.Normaliser(self.mean[:5], self.std[:5])

    def test_statistics_are_read_only(self):
        with self.assertRaises(ValueError):
            self.n.mean[0] = 99.0

    def test_shift_by_three_sigma(self):
        shifted = self.n.shifted(+3.0)
        x = self.mean + np.arange(6)
        np.testing.assert_allclose(shifted.transform(x), self.n.transform(x) - 3.0)
        np.testing.assert_allclose(self.n.mean, self.mean)              # original untouched
        one = self.n.shifted(+3.0, [3])
        np.testing.assert_allclose(one.transform(x) - self.n.transform(x), [0, 0, 0, -3, 0, 0])


class TestStatsAreNeverRecomputedAtRuntime(unittest.TestCase):
    def test_runtime_classes_have_no_fit_or_update(self):
        for cls in (pre.Normaliser, pre.PreprocessingPipeline, pre.MqttPreprocessor):
            names = [n for n in dir(cls) if not n.startswith("__")]
            self.assertFalse([n for n in names if n in ("fit", "update_stats", "refit", "recompute")], cls)

    def test_runtime_source_never_computes_or_saves_statistics(self):
        for cls in (pre.Normaliser, pre.PreprocessingPipeline, pre.MqttPreprocessor):
            src = inspect.getsource(cls)
            self.assertNotIn("compute_training_stats", src)
            self.assertNotIn("np.save", src)

    def test_statistics_unchanged_after_processing_live_data(self):
        n = pre.Normaliser(np.zeros(6), np.ones(6))
        before = (n.mean.copy(), n.std.copy())
        feed(pre.PreprocessingPipeline(n), 300, temp_fn=lambda t: 4 + 0.08 * t, vib_fn=lambda t: 1.2)
        np.testing.assert_array_equal(n.mean, before[0]); np.testing.assert_array_equal(n.std, before[1])

    def test_make_stats_produces_expected_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "training_stats.npy")
            stats = pre.make_training_stats(path, seed=5)
            self.assertEqual(np.load(path).shape, (2, 6))
            np.testing.assert_array_equal(np.load(path), stats)
            self.assertAlmostEqual(stats[0][0], 4.0, delta=0.05)        # temperature mean
            self.assertAlmostEqual(stats[0][3], 0.45, delta=0.01)       # vibration RMS
            self.assertTrue(np.all(stats[1] > 0))
            np.testing.assert_array_equal(pre.make_training_stats(path, seed=5), stats)   # reproducible

    def test_compute_training_stats_needs_enough_clean_windows(self):
        with self.assertRaises(ValueError):
            pre.compute_training_stats(np.zeros((5, 6)))


class FakeClient:
    instances = []

    def __init__(self, *a, **k):
        self.subscribed, self.published = [], []
        FakeClient.instances.append(self)

    def reconnect_delay_set(self, **k): pass
    def connect_async(self, *a): pass
    def loop_start(self): self.on_connect(self, None, {}, 0, None)
    def loop_stop(self): pass
    def disconnect(self): pass
    def subscribe(self, topic, qos=0): self.subscribed.append((topic, qos))
    def publish(self, topic, payload, qos=0, retain=False): self.published.append((topic, json.loads(payload), qos))


class TestMqttRuntime(unittest.TestCase):
    def setUp(self):
        FakeClient.instances.clear()
        paho, pkg, client = types.ModuleType("paho"), types.ModuleType("paho.mqtt"), types.ModuleType("paho.mqtt.client")
        client.Client, client.CallbackAPIVersion = FakeClient, types.SimpleNamespace(VERSION2=2)
        paho.mqtt, pkg.client = pkg, client
        self._saved = {k: sys.modules.get(k) for k in ("paho", "paho.mqtt", "paho.mqtt.client")}
        sys.modules.update({"paho": paho, "paho.mqtt": pkg, "paho.mqtt.client": client})

    def tearDown(self):
        for k, v in self._saved.items():
            sys.modules.pop(k, None) if v is None else sys.modules.__setitem__(k, v)

    def test_subscribes_preprocesses_and_publishes_features(self):
        norm = pre.Normaliser(np.array([4.0, 0.12, 0.0, 0.445, 0.47, -0.8]), np.array([0.055, 0.026, 0.36, 0.0105, 0.0117, 0.53]))
        seen = []
        mp = pre.MqttPreprocessor(pre.PreprocessingPipeline(norm), "localhost", 1883, "T001", on_window=seen.append)
        mp.client.loop_start()
        c = FakeClient.instances[0]
        self.assertIn(("truck/T001/temp", 0), c.subscribed)
        self.assertIn(("truck/T001/vib", 0), c.subscribed)
        for t in range(0, 45):
            for topic, kind, val in (("truck/T001/temp", "temp", 4.0), ("truck/T001/vib", "vib", 0.45)):
                if kind == "vib" and t % 2:
                    continue
                ts = f"2026-10-24T10:00:{t:02d}.000Z"
                mp._on_message(c, None, types.SimpleNamespace(topic=topic, payload=json.dumps({"ts": ts, "value": val}).encode()))
        self.assertEqual(len(seen), 2)                          # windows ending at 30 s and 40 s
        self.assertEqual(len(c.published), 2)
        topic, body, qos = c.published[0]
        self.assertEqual(topic, "truck/T001/features")
        self.assertEqual(len(body["raw"]), 6); self.assertEqual(len(body["z"]), 6)

    def test_malformed_messages_are_ignored(self):
        mp = pre.MqttPreprocessor(pre.PreprocessingPipeline(), "localhost", 1883, "T001")
        c = FakeClient.instances[0]
        for payload in (b"not json", b'{"value": 1}', b'{"ts": "bad", "value": 1}'):
            mp._on_message(c, None, types.SimpleNamespace(topic="truck/T001/temp", payload=payload))
        self.assertEqual(c.published, [])


class TestCli(unittest.TestCase):
    def test_make_stats_command(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.npy")
            self.assertEqual(pre.main(["make-stats", "--out", path]), 0)
            self.assertEqual(np.load(path).shape, (2, 6))

    def test_run_without_stats_file_fails_cleanly(self):
        self.assertEqual(pre.main(["run", "--stats", "/nonexistent/training_stats.npy"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
