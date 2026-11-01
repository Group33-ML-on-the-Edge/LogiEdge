"""Tests for the Task D2 deployment files. Run with:  python3 -m unittest -v test_deployment

What these tests cover (no Docker, TensorFlow or Mosquitto needed):
  * the Dockerfile obeys the task's rules (base image, pip layers before the model COPY, MODEL_PATH)
  * Docker's cache rule applied to the Dockerfile: a model-only change rebuilds only the model layer
  * the inference service: MODEL_PATH switching, topic, payload, MQTT calls (paho stub), TFLite path (fake interpreter)
  * the bandwidth arithmetic

What they cannot cover: a real `docker build`, the real TFLite runtime and a live broker.
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import types
import unittest

import numpy as np

import generate_dataset as gd
import inference_service as svc
import layer_cache_check as lcc
import ota_bandwidth as ota
from preprocessing import Normaliser

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
with open(os.path.join(HERE, "Dockerfile")) as _f:
    DOCKERFILE = _f.read()
INSTR = lcc.parse_dockerfile(DOCKERFILE)


def idx(pred):
    return [i for i, (n, a) in enumerate(INSTR) if pred(n, a)]


class TestDockerfileRules(unittest.TestCase):
    def test_base_image(self):
        self.assertEqual(INSTR[0], ("FROM", "python:3.11-slim"))

    def test_every_pip_install_comes_before_the_model_copy(self):
        model = idx(lambda n, a: n == "COPY" and a == "model.tflite .")
        self.assertEqual(len(model), 1, "exactly one 'COPY model.tflite .' instruction")
        pips = idx(lambda n, a: n == "RUN" and "pip install" in a)
        self.assertTrue(pips, "there is at least one pip install layer")
        self.assertTrue(all(p < model[0] for p in pips))

    def test_requirements_are_copied_before_they_are_installed(self):
        req = idx(lambda n, a: n == "COPY" and a.startswith("requirements.txt"))[0]
        pip = idx(lambda n, a: n == "RUN" and "pip install" in a)[0]
        self.assertLess(req, pip)

    def test_model_is_the_last_layer_that_carries_data(self):
        model = idx(lambda n, a: n == "COPY" and a == "model.tflite .")[0]
        after = [n for n, _ in INSTR[model + 1:]]
        self.assertTrue(set(after) <= {"USER", "CMD", "ENTRYPOINT"}, after)

    def test_model_path_variable_is_set_and_the_service_reads_it(self):
        env = " ".join(a for n, a in INSTR if n == "ENV")
        self.assertIn("MODEL_PATH=/app/model.tflite", env)
        self.assertEqual(svc.Config.from_env({"MODEL_PATH": "/models/x.tflite"}).model_path, "/models/x.tflite")

    def test_command_runs_the_service_as_a_non_root_user(self):
        cmd = [a for n, a in INSTR if n == "CMD"][0]
        self.assertIn("inference_service.py", cmd)
        self.assertIn(("USER", "logiedge"), INSTR)

    def test_copy_sources_exist_except_the_model_that_must_be_converted(self):
        missing = []
        for n, a in INSTR:
            if n == "COPY":
                for src in lcc.copy_sources(a):
                    if not os.path.exists(os.path.join(HERE, src)):
                        missing.append(src)
        self.assertIn(missing, ([], ["model.tflite"]))        # produced by convert_to_tflite.py

    def test_dockerignore_sends_only_what_is_copied(self):
        with open(os.path.join(HERE, ".dockerignore")) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        self.assertEqual(lines[0], "*")
        allowed = {l[1:] for l in lines if l.startswith("!")}
        sources = {s for n, a in INSTR if n == "COPY" for s in lcc.copy_sources(a)}
        self.assertTrue(sources - {"./"} <= allowed)

    def test_requirements_are_pinned(self):
        with open(os.path.join(HERE, "requirements.txt")) as f:
            reqs = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        names = {r.split("==")[0] for r in reqs}
        self.assertEqual(names, {"numpy", "paho-mqtt", "tflite-runtime"})
        self.assertTrue(all("==" in r for r in reqs))


class TestLayerCacheRule(unittest.TestCase):
    """Applies Docker's cache rule to the real Dockerfile with two build contexts."""

    def make_context(self, root, model_bytes, code="code v1", reqs="numpy==1.26.4\n"):
        os.makedirs(root, exist_ok=True)
        files = {"requirements.txt": reqs, "preprocessing.py": "pre", "drift_monitor.py": "drift",
                 "inference_service.py": code, "training_stats.npy": "stats", "model.tflite": model_bytes}
        for name, data in files.items():
            with open(os.path.join(root, name), "w") as f:
                f.write(data)
        return root

    def status(self, a, b):
        return {r["instruction"]: r["status"] for r in lcc.compare(DOCKERFILE, a, b)}

    def test_only_the_model_layer_is_rebuilt_when_only_the_model_changes(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.make_context(os.path.join(d, "a"), "model v1")
            b = self.make_context(os.path.join(d, "b"), "model v2")
            rows = lcc.compare(DOCKERFILE, a, b)
            rebuilt_data = [r["instruction"] for r in rows if r["status"] == "REBUILT" and r["adds_data"]]
            self.assertEqual(rebuilt_data, ["COPY model.tflite ."])
            cached = [r["instruction"] for r in rows if r["status"] == "CACHED"]
            self.assertTrue(any(i.startswith("RUN pip install") for i in cached))
            self.assertTrue(any(i.startswith("COPY preprocessing.py drift_monitor.py") for i in cached))
            self.assertTrue(any(i.startswith("FROM") for i in cached))

    def test_identical_contexts_are_fully_cached(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.make_context(os.path.join(d, "a"), "m")
            b = self.make_context(os.path.join(d, "b"), "m")
            self.assertTrue(all(r["status"] == "CACHED" for r in lcc.compare(DOCKERFILE, a, b)))

    def test_changing_code_also_rebuilds_the_model_layer_which_is_why_order_matters(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.make_context(os.path.join(d, "a"), "m")
            b = self.make_context(os.path.join(d, "b"), "m", code="code v2")
            st = self.status(a, b)
            self.assertEqual(st["COPY model.tflite ."], "REBUILT")
            pip = [v for k, v in st.items() if k.startswith("RUN pip install")][0]
            self.assertEqual(pip, "CACHED")                 # dependencies stay cached

    def test_changing_requirements_rebuilds_pip_and_everything_after(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.make_context(os.path.join(d, "a"), "m")
            b = self.make_context(os.path.join(d, "b"), "m", reqs="numpy==1.26.4\npaho-mqtt==2.1.0\n")
            st = self.status(a, b)
            self.assertEqual([v for k, v in st.items() if k.startswith("RUN pip install")][0], "REBUILT")
            self.assertEqual(st["COPY model.tflite ."], "REBUILT")

    def test_parser_handles_comments_and_continuations(self):
        text = "# c\nFROM a\nENV X=1 \\\n    Y=2\n# mid\nRUN echo hi\n"
        self.assertEqual(lcc.parse_dockerfile(text), [("FROM", "a"), ("ENV", "X=1 Y=2"), ("RUN", "echo hi")])

    def test_missing_copy_source_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                lcc.cache_keys("FROM a\nCOPY nothing.txt .\n", d)


# ----------------------------------------------------------------------------
# paho stub
# ----------------------------------------------------------------------------
class FakeClient:
    instances = []

    def __init__(self, *a, **k):
        self.published, self.subscribed, self.calls = [], [], []
        self.will = None
        FakeClient.instances.append(self)

    def username_pw_set(self, u, p=None): self.calls.append(("auth", u, p))
    def will_set(self, topic, payload, qos=0, retain=False): self.will = (topic, payload, qos, retain)
    def reconnect_delay_set(self, **k): pass
    def connect_async(self, host, port, keepalive): self.calls.append(("connect", host, port))
    def loop_start(self): self.on_connect(self, None, {}, 0, None)
    def loop_stop(self): self.calls.append(("loop_stop",))
    def disconnect(self): self.calls.append(("disconnect",))
    def subscribe(self, topic, qos=0): self.subscribed.append((topic, qos))

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, payload, qos, retain))
        return types.SimpleNamespace(rc=0, wait_for_publish=lambda timeout=None: None)


class PahoStubMixin:
    def setUp(self):
        FakeClient.instances.clear()
        paho, pkg, client = types.ModuleType("paho"), types.ModuleType("paho.mqtt"), types.ModuleType("paho.mqtt.client")
        client.Client, client.CallbackAPIVersion, client.MQTT_ERR_SUCCESS = FakeClient, types.SimpleNamespace(VERSION2=2), 0
        paho.mqtt, pkg.client = pkg, client
        self._saved = {k: sys.modules.get(k) for k in ("paho", "paho.mqtt", "paho.mqtt.client")}
        sys.modules.update({"paho": paho, "paho.mqtt": pkg, "paho.mqtt.client": client})

    def tearDown(self):
        for k, v in self._saved.items():
            sys.modules.pop(k, None) if v is None else sys.modules.__setitem__(k, v)


NORM = Normaliser.load(os.path.join(HERE, "training_stats.npy"))
W_MAIN = os.path.join(HERE, "model_weights.npz")
W_ONSET = os.path.join(HERE, "model_weights_onset.npz")


def feed_simulator(service, mode, seconds=300, seed=777, prefix="truck/T001"):
    client = FakeClient.instances[0]
    for topic, body in gd.run_simulator(mode, seconds, seed=seed):
        kind = topic.rsplit("/", 1)[1]
        if kind in ("temp", "vib"):
            msg = types.SimpleNamespace(topic=f"{prefix}/{kind}", payload=json.dumps(body).encode())
            service._on_message(client, None, msg)
    return client


class TestServiceConfigAndModels(unittest.TestCase):
    def test_inference_topic_matches_the_specification(self):
        self.assertEqual(svc.Config(truck_id="T042").inference_topic, "logibridge/trucks/T042/inference")
        self.assertEqual(svc.Config.from_env({"TRUCK_ID": "T7"}).inference_topic, "logibridge/trucks/T7/inference")

    def test_defaults_and_overrides_from_environment(self):
        d = svc.Config.from_env({})
        self.assertEqual((d.model_path, d.mqtt_host, d.mqtt_port, d.truck_id), ("/app/model.tflite", "localhost", 1883, "T001"))
        c = svc.Config.from_env({"MQTT_HOST": "mosquitto", "MQTT_PORT": "8883", "SENSOR_TOPIC_PREFIX": "x/{truck_id}", "TRUCK_ID": "A"})
        self.assertEqual((c.mqtt_host, c.mqtt_port), ("mosquitto", 8883))
        self.assertEqual(set(c.sensor_topics), {"x/A/temp", "x/A/vib"})

    def test_numpy_model_matches_the_training_code(self):
        import train_model as tm
        X, y, _ = tm.load_dataset(os.path.join(HERE, "training_dataset.npz"), NORM)
        ref = tm.MLP.load(W_MAIN)
        m = svc.load_model(W_MAIN)
        pred = np.array([int(np.argmax(m.predict_proba(x))) for x in X])
        np.testing.assert_array_equal(pred, ref.predict(X))
        np.testing.assert_allclose(m.predict_proba(X[5]), ref.predict_proba(X[5:6])[0], atol=1e-9)

    def test_model_path_switch_loads_a_different_variant(self):
        a, b = svc.load_model(W_MAIN), svc.load_model(W_ONSET)
        self.assertNotEqual(a.sha256, b.sha256)
        self.assertEqual((a.file, b.file), ("model_weights.npz", "model_weights_onset.npz"))

    def test_probabilities_or_logits_are_both_accepted(self):
        np.testing.assert_allclose(svc.to_probabilities(np.array([0.2, 0.3, 0.5])), [0.2, 0.3, 0.5])
        np.testing.assert_allclose(svc.to_probabilities(np.array([1.0, 2.0, 3.0])).sum(), 1.0)

    def test_startup_errors_return_code_2(self):
        env = {"STATS_PATH": os.path.join(HERE, "training_stats.npy")}
        self.assertEqual(svc.main({**env, "MODEL_PATH": "/nonexistent/model.tflite"}), 2)
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "model.onnx")
            with open(bad, "wb") as f:
                f.write(b"x")
            self.assertEqual(svc.main({**env, "MODEL_PATH": bad}), 2)
        self.assertEqual(svc.main({"MODEL_PATH": W_MAIN, "STATS_PATH": "/nonexistent.npy"}), 2)


class TestServicePublishing(PahoStubMixin, unittest.TestCase):
    def make(self, model_path=W_MAIN, **cfg):
        c = svc.Config(model_path=model_path, **cfg)
        return svc.InferenceService(c, svc.load_model(model_path), NORM)

    def test_connect_subscribes_sets_will_and_reports_online(self):
        s = self.make()
        s.client.loop_start()
        c = FakeClient.instances[0]
        self.assertEqual(c.will, ("logibridge/trucks/T001/status", "offline", 1, True))
        self.assertIn(("truck/T001/temp", 0), c.subscribed)
        self.assertIn(("truck/T001/vib", 0), c.subscribed)
        self.assertEqual(c.published[0], ("logibridge/trucks/T001/status", "online", 1, True))

    def test_each_window_is_published_once_to_the_inference_topic(self):
        s = self.make()
        s.client.loop_start()
        c = feed_simulator(s, "none", 100)
        results = [p for p in c.published if p[0].endswith("/inference")]
        self.assertEqual(len(results), 7)                                   # windows ending 30..90 s
        for topic, payload, qos, retain in results:
            self.assertEqual(topic, "logibridge/trucks/T001/inference")
            self.assertEqual((qos, retain), (1, False))
            body = json.loads(payload)
            self.assertEqual(set(body), {"truck_id", "window_end", "class", "label", "confidence",
                                         "probabilities", "features", "model"})
            self.assertEqual(body["label"], svc.CLASS_NAMES[body["class"]])
            self.assertAlmostEqual(sum(body["probabilities"]), 1.0, delta=2e-3)
            self.assertEqual(len(body["features"]), 6)

    def test_topic_follows_truck_id(self):
        s = self.make(truck_id="T042", sensor_topic_prefix="truck/{truck_id}")
        s.client.loop_start()
        c = feed_simulator(s, "none", 60, prefix="truck/T042")
        self.assertTrue(any(p[0] == "logibridge/trucks/T042/inference" for p in c.published))

    def test_results_follow_the_simulator_mode(self):
        for mode, expected in (("none", 0), ("temp_drift", 1), ("combined", 2)):
            FakeClient.instances.clear()
            s = self.make()
            s.client.loop_start()
            c = feed_simulator(s, mode, 300)
            classes = [json.loads(p[1])["class"] for p in c.published if p[0].endswith("/inference")]
            self.assertGreater(np.mean(np.array(classes) == expected), 0.9, mode)

    def test_model_path_switch_changes_the_model_reported_in_results(self):
        seen = []
        for path in (W_MAIN, W_ONSET):
            FakeClient.instances.clear()
            s = self.make(model_path=path)
            s.client.loop_start()
            c = feed_simulator(s, "none", 60)
            body = json.loads([p for p in c.published if p[0].endswith("/inference")][0][1])
            seen.append((body["model"]["file"], body["model"]["sha256"]))
        self.assertNotEqual(seen[0], seen[1])

    def test_malformed_messages_are_counted_not_fatal(self):
        s = self.make()
        c = FakeClient.instances[0]
        for payload in (b"junk", b'{"value": 1}', b'{"ts": "bad", "value": 1}'):
            s._on_message(c, None, types.SimpleNamespace(topic="truck/T001/temp", payload=payload))
        self.assertEqual(s.errors, 3)

    def test_a_failing_model_does_not_stop_the_service(self):
        s = self.make()
        s.model = types.SimpleNamespace(predict_proba=lambda z: 1 / 0, file="x", sha256="y")
        s.client.loop_start()
        c = feed_simulator(s, "none", 60)
        self.assertGreater(s.errors, 0)
        self.assertEqual(s.published, 0)

    def test_stop_publishes_offline_and_disconnects(self):
        s = self.make()
        stop = threading.Event()
        stop.set()
        s.run(stop)
        c = FakeClient.instances[0]
        self.assertEqual(c.published[-1], ("logibridge/trucks/T001/status", "offline", 1, True))
        self.assertIn(("disconnect",), c.calls)


class TestServiceDriftMonitoring(PahoStubMixin, unittest.TestCase):
    REF = os.path.join(HERE, "reference_dist.json")

    def make(self, **cfg):
        c = svc.Config(model_path=W_MAIN, reference_dist_path=self.REF, **cfg)
        return svc.InferenceService(c, svc.load_model(W_MAIN), NORM)

    def test_config_reads_drift_settings_from_the_environment(self):
        c = svc.Config.from_env({"REFERENCE_DIST_PATH": "/opt/logibridge/reference_dist.json", "DRIFT_SCORE": "max_confidence"})
        self.assertEqual((c.reference_dist_path, c.drift_score), ("/opt/logibridge/reference_dist.json", "max_confidence"))
        self.assertIsNone(svc.Config.from_env({}).reference_dist_path)
        self.assertEqual(svc.Config(truck_id="T9").drift_topic, "logibridge/trucks/T9/drift")

    def test_monitoring_is_off_without_a_reference(self):
        s = svc.InferenceService(svc.Config(model_path=W_MAIN), svc.load_model(W_MAIN), NORM)
        self.assertIsNone(s.drift)

    def test_bad_reference_disables_monitoring_but_not_inference(self):
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "ref.json")
            with open(bad, "w") as f:
                f.write("{}")
            s = svc.InferenceService(svc.Config(model_path=W_MAIN, reference_dist_path=bad), svc.load_model(W_MAIN), NORM)
        self.assertIsNone(s.drift)

    def test_psi_lines_are_printed_and_published_every_60_seconds(self):
        import contextlib, io
        s = self.make()
        s.client.loop_start()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            c = feed_simulator(s, "none", 400)
        out = buf.getvalue()
        self.assertIn("[LOGIBRIDGE] t=", out)
        self.assertIn("PSI n/a (collecting", out)                          # fewer than 100 inferences so far
        drift = [json.loads(p[1]) for p in c.published if p[0] == "logibridge/trucks/T001/drift"]
        self.assertGreaterEqual(len(drift), 5)
        self.assertTrue(all(d["status"] == "warming_up" for d in drift))

    def test_alert_line_appears_when_the_data_drifts(self):
        import contextlib, io
        s = self.make()
        s.client.loop_start()
        s.drift.window = 20                                                  # short window so the test is quick
        s.drift._buf = type(s.drift._buf)(maxlen=20)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            feed_simulator(s, "none", 300)
            feed_simulator(s, "combined", 400, seed=3, prefix="truck/T001")
        self.assertRegex(buf.getvalue(), r"\[LOGIBRIDGE DRIFT ALERT\] PSI=\d+\.\d{3}")


class FakeInterpreter:
    """Stands in for the TFLite interpreter."""
    def __init__(self, model_path=None, in_dtype=np.float32, in_q=(0.0, 0), out_dtype=np.float32, out_q=(0.0, 0), out=None):
        self.in_dtype, self.in_q, self.out_dtype, self.out_q = in_dtype, in_q, out_dtype, out_q
        self.out = out if out is not None else np.array([[0.1, 0.2, 3.0]])
        self.set_values, self.invoked = [], 0

    def allocate_tensors(self): pass
    def get_input_details(self): return [{"index": 0, "dtype": self.in_dtype, "quantization": self.in_q}]
    def get_output_details(self): return [{"index": 1, "dtype": self.out_dtype, "quantization": self.out_q}]
    def set_tensor(self, i, x): self.set_values.append(x)
    def invoke(self): self.invoked += 1
    def get_tensor(self, i): return self.out


class TestTFLitePath(unittest.TestCase):
    def setUp(self):
        self._saved = {k: sys.modules.get(k) for k in ("tflite_runtime", "tflite_runtime.interpreter")}
        self.mod = types.ModuleType("tflite_runtime.interpreter")
        sys.modules["tflite_runtime"] = types.ModuleType("tflite_runtime")
        sys.modules["tflite_runtime.interpreter"] = self.mod

    def tearDown(self):
        for k, v in self._saved.items():
            sys.modules.pop(k, None) if v is None else sys.modules.__setitem__(k, v)

    def model_file(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        p = os.path.join(d, "model.tflite")
        with open(p, "wb") as f:
            f.write(b"fake tflite bytes")
        return p

    def test_float_model_gets_a_float32_row_and_returns_softmax(self):
        holder = {}
        self.mod.Interpreter = lambda model_path=None: holder.setdefault("i", FakeInterpreter(model_path))
        m = svc.load_model(self.model_file())
        probs = m.predict_proba(np.arange(6.0))
        i = holder["i"]
        self.assertEqual(i.invoked, 1)
        self.assertEqual(i.set_values[0].dtype, np.float32)
        self.assertEqual(i.set_values[0].shape, (1, 6))
        self.assertAlmostEqual(probs.sum(), 1.0)
        self.assertEqual(int(np.argmax(probs)), 2)
        self.assertEqual(m.file, "model.tflite")

    def test_int8_model_is_quantised_in_and_dequantised_out(self):
        holder = {}
        self.mod.Interpreter = lambda model_path=None: holder.setdefault(
            "i", FakeInterpreter(model_path, np.int8, (0.5, 0), np.int8, (0.1, 0), out=np.array([[10, 20, 100]], dtype=np.int8)))
        m = svc.load_model(self.model_file())
        probs = m.predict_proba(np.array([10.0, -10.0, 1000.0, 0.0, 0.2, 0.0]))
        sent = holder["i"].set_values[0]
        self.assertEqual(sent.dtype, np.int8)
        self.assertEqual(sent.tolist(), [[20, -20, 127, 0, 0, 0]])              # scaled by 1/0.5 and clipped to int8
        self.assertEqual(int(np.argmax(probs)), 2)

    def test_missing_runtime_gives_a_clear_error(self):
        from unittest.mock import patch

        with patch.dict(sys.modules, {
            "tflite_runtime": None,
            "tflite_runtime.interpreter": None,
            "tensorflow": None,
        }):
            with self.assertRaises((RuntimeError, ImportError)):
                svc._make_interpreter(self.model_file())


class TestBandwidth(unittest.TestCase):
    def test_worked_example_for_85_trucks(self):
        r = ota.ota_saving(65_000_000, 6_000, 10_000, 85, 0.10)
        self.assertAlmostEqual(r["full_fleet_mb"], 5525.0)
        self.assertAlmostEqual(r["layer_fleet_mb"], 1.36)
        self.assertAlmostEqual(r["saving_fleet_mb"], 5523.64)
        self.assertAlmostEqual(r["saving_percent"], 99.9754, places=3)
        self.assertAlmostEqual(r["full_cost"], 552.50)
        self.assertAlmostEqual(r["saving_cost"], 552.364)

    def test_larger_models_save_less(self):
        small = ota.ota_saving(65e6, 6e3)["saving_percent"]
        big = ota.ota_saving(65e6, 5e6)["saving_percent"]
        self.assertGreater(small, big)
        self.assertAlmostEqual(big, 92.29, places=1)

    def test_scales_with_the_number_of_trucks(self):
        a, b = ota.ota_saving(65e6, 6e3, trucks=85), ota.ota_saving(65e6, 6e3, trucks=265)
        self.assertAlmostEqual(b["saving_fleet_mb"] / a["saving_fleet_mb"], 265 / 85)


if __name__ == "__main__":
    unittest.main(verbosity=2)
