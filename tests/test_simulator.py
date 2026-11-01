"""Tests for simulator.py. Run with:  python3 -m unittest -v test_simulator

The MQTT adapter is tested against a stub of paho-mqtt, so no broker is needed.
A live check against Mosquitto is described in the Task C1 report.
"""

import json
import random
import statistics
import sys
import threading
import types
import unittest
from datetime import datetime

import simulator as sim

START = 1_800_000_000.0


class ListPublisher:
    """Collects (topic, payload dict, qos) instead of publishing."""

    def __init__(self):
        self.messages = []

    def connect(self): pass
    def close(self): pass

    def publish(self, topic, payload, qos):
        self.messages.append((topic, json.loads(payload), qos))
        return True


def collect(anomaly="none", duration=600, seed=7, **kw):
    """Run the simulator on a fake clock (instant and deterministic)."""
    clock = {"t": 0.0}
    cfg = sim.SimConfig(anomaly=anomaly, duration_s=duration, seed=seed, **kw)
    pub = ListPublisher()
    counts = sim.run(cfg, pub, clock=lambda: clock["t"],
                     wait=lambda s: clock.__setitem__("t", clock["t"] + s), start_epoch=START)
    return pub.messages, counts


def values(messages, kind):
    return [m[1]["value"] for m in messages if m[0].endswith("/" + kind)]


def epoch(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


class TestSensorModels(unittest.TestCase):
    def test_normal_temperature_statistics(self):
        s = sim.TemperatureSensor(random.Random(1), drift_enabled=False)
        v = [s.read(False) for _ in range(20000)]
        self.assertAlmostEqual(statistics.mean(v), 4.0, delta=0.02)
        self.assertAlmostEqual(statistics.stdev(v), 0.3, delta=0.01)

    def test_temperature_drift_is_linear_008_per_reading(self):
        s = sim.TemperatureSensor(random.Random(2), drift_enabled=True)
        v = [s.read(True) for _ in range(600)]
        n = len(v)
        xs = range(1, n + 1)
        xbar, ybar = statistics.mean(xs), statistics.mean(v)
        slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, v)) / sum((x - xbar) ** 2 for x in xs)
        self.assertAlmostEqual(slope, 0.08, delta=0.002)
        self.assertAlmostEqual(statistics.mean(v[95:105]), 4.0 + 0.08 * 100, delta=0.3)

    def test_temperature_cap(self):
        s = sim.TemperatureSensor(random.Random(3), drift_enabled=True, cap_c=35.0)
        self.assertLessEqual(max(s.read(True) for _ in range(1000)), 35.0)

    def test_vibration_normal_and_fault(self):
        n = sim.VibrationSensor(random.Random(4), True)
        nv = [n.read(False) for _ in range(20000)]
        self.assertAlmostEqual(statistics.mean(nv), 0.45, delta=0.005)
        self.assertAlmostEqual(statistics.stdev(nv), 0.05, delta=0.003)
        f = sim.VibrationSensor(random.Random(5), True)
        fv = [f.read(True) for _ in range(20000)]
        self.assertAlmostEqual(statistics.mean(fv), 1.2, delta=0.01)
        self.assertAlmostEqual(statistics.stdev(fv), 0.15, delta=0.005)

    def test_vibration_never_negative(self):
        s = sim.VibrationSensor(random.Random(6), True)
        self.assertGreaterEqual(min(s.read(False) for _ in range(5000)), 0.0)

    def test_door_alternates_and_starts_with_open(self):
        d = sim.DoorSimulator(random.Random(8), 300)
        events = [d.pop_event() for _ in range(200)]
        states = [e[1] for e in events]
        self.assertEqual(states[0], "OPEN")
        self.assertTrue(all(a != b for a, b in zip(states, states[1:])))
        times = [e[0] for e in events]
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])))
        open_durations = [events[i + 1][0] - events[i][0] for i in range(0, 198, 2)]
        self.assertTrue(all(sim.DOOR_OPEN_MIN_S <= x <= sim.DOOR_OPEN_MAX_S for x in open_durations))


class TestStreams(unittest.TestCase):
    def test_rates_and_timestamp_spacing(self):
        msgs, counts = collect("none", duration=100)
        self.assertEqual(counts["temp"], 100)          # 1 Hz
        self.assertEqual(counts["vib"], 50)            # 0.5 Hz
        for kind, step in (("temp", 1.0), ("vib", 2.0)):
            ts = [epoch(m[1]["ts"]) for m in msgs if m[0].endswith("/" + kind)]
            gaps = [b - a for a, b in zip(ts, ts[1:])]
            self.assertTrue(all(abs(g - step) < 1e-3 for g in gaps))

    def test_topics_qos_and_payload_fields(self):
        msgs, _ = collect("none", duration=1200, door_interval_s=60)
        seen = {}
        for topic, body, qos in msgs:
            seen[topic] = qos
            self.assertEqual(body["truck_id"], "T001")
            self.assertIn("ts", body)
        self.assertEqual(seen["truck/T001/temp"], 0)
        self.assertEqual(seen["truck/T001/vib"], 0)
        self.assertEqual(seen["truck/T001/door"], 1)
        door = [m[1] for m in msgs if m[0].endswith("/door")]
        self.assertTrue(door and all(d["event"] in ("OPEN", "CLOSE") for d in door))

    def test_sequence_numbers_increase_by_one(self):
        msgs, _ = collect("none", duration=60)
        for kind in ("temp", "vib"):
            seqs = [m[1]["seq"] for m in msgs if m[0].endswith("/" + kind)]
            self.assertEqual(seqs, list(range(1, len(seqs) + 1)))

    def test_none_mode_is_all_normal(self):
        msgs, _ = collect("none", duration=1200)
        self.assertAlmostEqual(statistics.mean(values(msgs, "temp")), 4.0, delta=0.05)
        self.assertAlmostEqual(statistics.mean(values(msgs, "vib")), 0.45, delta=0.01)

    def test_temp_drift_mode(self):
        msgs, _ = collect("temp_drift", duration=600)
        t, v = values(msgs, "temp"), values(msgs, "vib")
        self.assertGreater(statistics.mean(t[-30:]), 4.0 + 0.08 * 550)
        self.assertAlmostEqual(statistics.mean(v), 0.45, delta=0.02)

    def test_vibration_mode(self):
        msgs, _ = collect("vibration", duration=1200)
        self.assertAlmostEqual(statistics.mean(values(msgs, "temp")), 4.0, delta=0.05)
        self.assertAlmostEqual(statistics.mean(values(msgs, "vib")), 1.2, delta=0.03)

    def test_combined_mode_has_both_anomalies(self):
        msgs, _ = collect("combined", duration=600, door_interval_s=60)
        self.assertGreater(statistics.mean(values(msgs, "temp")[-30:]), 40)
        self.assertAlmostEqual(statistics.mean(values(msgs, "vib")), 1.2, delta=0.04)
        door = [m[1]["event"] for m in msgs if m[0].endswith("/door")]
        self.assertGreater(len(door), 2)                    # door events continue in every mode

    def test_anomaly_start_delays_onset(self):
        msgs, _ = collect("combined", duration=300, anomaly_start_s=100)
        t, v = values(msgs, "temp"), values(msgs, "vib")
        self.assertAlmostEqual(statistics.mean(t[:95]), 4.0, delta=0.1)       # before onset
        self.assertAlmostEqual(statistics.mean(v[:45]), 0.45, delta=0.03)
        self.assertGreater(statistics.mean(t[-20:]), 4.0 + 0.08 * 180)       # after onset
        self.assertGreater(statistics.mean(v[-20:]), 1.0)

    def test_same_seed_reproducible_different_seed_differs(self):
        a, _ = collect("combined", duration=120, seed=11)
        b, _ = collect("combined", duration=120, seed=11)
        c, _ = collect("combined", duration=120, seed=12)
        self.assertEqual(a, b)
        self.assertNotEqual(values(a, "temp"), values(c, "temp"))

    def test_speed_factor_shortens_real_time(self):
        clock = {"t": 0.0, "waited": 0.0}

        def wait(s):
            clock["t"] += s
            clock["waited"] += s

        cfg = sim.SimConfig(duration_s=100, speed=10.0, seed=1)
        sim.run(cfg, ListPublisher(), clock=lambda: clock["t"], wait=wait, start_epoch=START)
        self.assertAlmostEqual(clock["waited"], 10.0, delta=0.5)   # 100 simulated s in about 10 real s

    def test_stop_event_ends_run(self):
        stop = threading.Event()
        stop.set()
        counts = sim.run(sim.SimConfig(duration_s=0), ListPublisher(), stop_event=stop)
        self.assertEqual(counts["temp"], 0)


class TestAnomalyEnd(unittest.TestCase):
    def test_clean_data_returns_after_anomaly_end(self):
        msgs, _ = collect("combined", duration=600, anomaly_start_s=100, anomaly_end_s=300)
        t, v = values(msgs, "temp"), values(msgs, "vib")
        self.assertGreater(statistics.mean(t[250:290]), 4.0 + 0.08 * 150)             # drifting inside the window
        self.assertAlmostEqual(statistics.mean(t[310:600]), 4.0, delta=0.1)           # clean again afterwards
        self.assertAlmostEqual(statistics.mean(v[160:300]), 0.45, delta=0.02)
        self.assertGreater(statistics.mean(v[60:140]), 1.0)

    def test_drift_restarts_from_zero_in_a_second_episode(self):
        s = sim.TemperatureSensor(random.Random(1), True)
        for _ in range(50):
            s.read(True)
        s.read(False)
        self.assertEqual(s.anomalous_readings, 0)
        self.assertLess(statistics.mean(s.read(True) for _ in range(5)), 4.0 + 0.08 * 6 + 0.6)

    def test_no_end_means_the_anomaly_lasts(self):
        msgs, _ = collect("temp_drift", duration=300, anomaly_start_s=100)
        self.assertGreater(statistics.mean(values(msgs, "temp")[-20:]), 4.0 + 0.08 * 180)

    def test_cli_accepts_and_validates_anomaly_end(self):
        a = sim.build_parser().parse_args(["--anomaly", "combined", "--anomaly-start", "10", "--anomaly-end", "20"])
        self.assertEqual((a.anomaly_start, a.anomaly_end), (10, 20))
        with self.assertRaises(SystemExit):
            sim.main(["--dry-run", "--anomaly-start", "50", "--anomaly-end", "40"])


class TestCli(unittest.TestCase):
    def test_default_is_none_and_localhost(self):
        a = sim.build_parser().parse_args([])
        self.assertEqual((a.anomaly, a.host, a.port), ("none", "localhost", 1883))

    def test_all_anomaly_choices_accepted(self):
        for mode in ("none", "temp_drift", "vibration", "combined"):
            self.assertEqual(sim.build_parser().parse_args(["--anomaly", mode]).anomaly, mode)

    def test_invalid_anomaly_rejected(self):
        with self.assertRaises(SystemExit) as cm:
            sim.build_parser().parse_args(["--anomaly", "bogus"])
        self.assertEqual(cm.exception.code, 2)

    def test_bad_speed_rejected(self):
        with self.assertRaises(SystemExit):
            sim.main(["--dry-run", "--speed", "0"])


class FakeClient:
    """Stub of paho.mqtt.client.Client that records calls."""
    instances = []

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.published, self.calls = [], []
        self.will = None
        FakeClient.instances.append(self)

    def username_pw_set(self, u, p=None): self.calls.append(("auth", u, p))
    def will_set(self, topic, payload, qos=0, retain=False): self.will = (topic, payload, qos, retain)
    def reconnect_delay_set(self, min_delay, max_delay): self.calls.append(("reconnect", min_delay, max_delay))
    def connect_async(self, host, port, keepalive): self.calls.append(("connect_async", host, port, keepalive))
    def loop_start(self): self.on_connect(self, None, {}, 0, None)
    def loop_stop(self): self.calls.append(("loop_stop",))
    def disconnect(self): self.calls.append(("disconnect",))

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, payload, qos, retain))
        return types.SimpleNamespace(rc=0, wait_for_publish=lambda timeout=None: None)


class TestMqttAdapter(unittest.TestCase):
    def setUp(self):
        FakeClient.instances.clear()
        paho = types.ModuleType("paho")
        mqtt_pkg = types.ModuleType("paho.mqtt")
        client = types.ModuleType("paho.mqtt.client")
        client.Client = FakeClient
        client.MQTT_ERR_SUCCESS = 0
        client.CallbackAPIVersion = types.SimpleNamespace(VERSION2=2)
        paho.mqtt, mqtt_pkg.client = mqtt_pkg, client
        self._saved = {k: sys.modules.get(k) for k in ("paho", "paho.mqtt", "paho.mqtt.client")}
        sys.modules.update({"paho": paho, "paho.mqtt": mqtt_pkg, "paho.mqtt.client": client})

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_end_to_end_publish_to_localhost(self):
        pub = sim.MqttPublisher("localhost", 1883, "T001")
        pub.connect()
        clock = {"t": 0.0}
        cfg = sim.SimConfig(anomaly="combined", duration_s=20, seed=3, door_interval_s=5)
        counts = sim.run(cfg, pub, clock=lambda: clock["t"],
                         wait=lambda s: clock.__setitem__("t", clock["t"] + s), start_epoch=START)
        pub.close()
        c = FakeClient.instances[0]
        self.assertIn(("connect_async", "localhost", 1883, 30), c.calls)
        self.assertEqual(c.will, ("truck/T001/status", "offline", 1, True))
        data = [p for p in c.published if not p[0].endswith("/status")]
        self.assertEqual(sum(p[0].endswith("/temp") for p in data), 20)
        self.assertEqual(sum(p[0].endswith("/vib") for p in data), 10)
        self.assertEqual(counts["dropped"], 0)
        for topic, payload, qos, retain in data:
            self.assertFalse(retain)
            self.assertEqual(qos, sim.QOS[topic.rsplit("/", 1)[1]])
            json.loads(payload)
        statuses = [p[1] for p in c.published if p[0].endswith("/status")]
        self.assertEqual(statuses, ["online", "offline"])
        self.assertIn(("disconnect",), c.calls)

    def test_connection_failure_raises(self):
        class NeverConnects(FakeClient):
            def loop_start(self): pass
        sys.modules["paho.mqtt.client"].Client = NeverConnects
        pub = sim.MqttPublisher("localhost", 1883, "T001", connect_timeout=0.05)
        with self.assertRaises(ConnectionError):
            pub.connect()
        self.assertEqual(sim.main(["--connect-timeout", "0.05"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
