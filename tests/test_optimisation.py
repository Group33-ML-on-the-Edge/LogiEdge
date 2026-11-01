"""Tests for model_variants.py, benchmark.py and build_variants_tf.py (Tasks F1 and F2).

Run with:  python3 -m unittest -v test_optimisation
TensorFlow is not needed: the TensorFlow script is checked for syntax and required settings only.
"""

import ast
import json
import os
import sys
import types
import unittest

import numpy as np

import benchmark as bm
import model_variants as mv
from train_model import MLP

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
SPLITS = mv.load_splits()
M1 = MLP.load(os.path.join(HERE, "model_weights_onset.npz"))


class TestQuantisation(unittest.TestCase):
    def test_zero_point_makes_zero_exactly_representable(self):
        for lo, hi in ((-3.0, 1270.0), (0.0, 5.0), (-2.0, 2.0), (4.0, 9.0)):
            scale, zero = mv.asym_qparams(lo, hi)
            self.assertEqual(int(mv.quantize(np.array([0.0]), scale, zero)[0]), zero)
            self.assertTrue(-128 <= zero <= 127)

    def test_quantize_clips_to_int8(self):
        q = mv.quantize(np.array([-1e6, 0.0, 1e6]), 0.5, 0)
        self.assertEqual(q.tolist(), [-128, 0, 127])

    def test_needs_at_least_200_calibration_samples(self):
        with self.assertRaises(ValueError):
            mv.QuantisedMLP.from_float(M1, SPLITS["X_calib"][:199])
        mv.QuantisedMLP.from_float(M1, SPLITS["X_calib"][:200])
        self.assertGreaterEqual(len(SPLITS["X_calib"]), 200)

    def test_integer_types_and_ranges(self):
        q = mv.QuantisedMLP.from_float(M1, SPLITS["X_calib"])
        for L in q.layers:
            self.assertEqual(L["qw"].dtype, np.int8)
            self.assertTrue(L["qw"].min() >= -127 and L["qw"].max() <= 127)
            self.assertEqual(L["qb"].dtype, np.int32)
            self.assertEqual(len(L["sw"]), L["qw"].shape[1])           # one scale per output channel

    def test_int8_logits_track_the_float_logits(self):
        q = mv.QuantisedMLP.from_float(M1, SPLITS["X_calib"])
        X = SPLITS["X_val"]
        ref = M1._forward(X)[-1]
        err = np.abs(q.logits(X) - ref)
        self.assertLess(np.median(err) / np.median(np.abs(ref)), 0.15)

    def test_predictions_match_the_float_model_on_validation(self):
        q = mv.QuantisedMLP.from_float(M1, SPLITS["X_calib"])
        np.testing.assert_array_equal(q.predict(SPLITS["X_val"]), M1.predict(SPLITS["X_val"]))

    def test_per_tensor_and_per_channel_agree_here(self):
        a = mv.QuantisedMLP.from_float(M1, SPLITS["X_calib"], per_channel=True)
        b = mv.QuantisedMLP.from_float(M1, SPLITS["X_calib"], per_channel=False)
        np.testing.assert_array_equal(a.predict(SPLITS["X_val"]), b.predict(SPLITS["X_val"]))

    def test_input_quantisation_collapses_the_normal_range(self):
        q = mv.QuantisedMLP.from_float(M1, SPLITS["X_calib"])
        scale, zero = q.in_q
        self.assertGreater(scale, 3.0)                                  # one step is several Normal-class sigmas
        normal = np.random.default_rng(0).normal(0, 1, (200, 6))
        self.assertLessEqual(len(np.unique(mv.quantize(normal, scale, zero))), 3)


class TestPruning(unittest.TestCase):
    def test_polynomial_decay_schedule(self):
        f = lambda t: mv.polynomial_decay(t, 0, 100, 0.0, 0.35)
        self.assertEqual(f(0), 0.0)
        self.assertAlmostEqual(f(100), 0.35)
        self.assertAlmostEqual(f(1000), 0.35)
        self.assertAlmostEqual(f(50), 0.35 * (1 - 0.5 ** 3))
        vals = [f(t) for t in range(0, 101, 10)]
        self.assertEqual(vals, sorted(vals))

    def test_35_percent_of_the_hidden_layers_is_removed(self):
        self.assertEqual(mv.neurons_kept(32, 0.35), 21)
        self.assertEqual(mv.neurons_kept(16, 0.35), 10)

    def test_neuron_scores_are_mean_absolute_incoming_weights(self):
        W = np.array([[1.0, -2.0], [3.0, 0.0]])
        np.testing.assert_allclose(mv.neuron_scores(W), [2.0, 1.0])

    def test_removing_dead_neurons_keeps_the_function(self):
        rng = np.random.default_rng(1)
        W = [rng.normal(size=(6, 8)), rng.normal(size=(8, 5)), rng.normal(size=(5, 3))]
        b = [rng.normal(size=8), rng.normal(size=5), rng.normal(size=3)]
        W[0][:, [1, 4]] = 0.0                                           # two dead neurons, one with a positive bias
        b[0][1], b[0][4] = 2.0, -1.0
        W[1][:, [0]] = 0.0
        X = rng.normal(size=(50, 6))
        big = MLP((6, 8, 5, 3)); big.W, big.b = W, b
        sW, sb = mv.shrink_dead_neurons(W, b)
        self.assertEqual([w.shape for w in sW], [(6, 6), (6, 4), (4, 3)])
        small = MLP((6, 6, 4, 3)); small.W, small.b = sW, sb
        np.testing.assert_allclose(small._forward(X)[-1], big._forward(X)[-1], atol=1e-9)

    def test_structured_pruning_run(self):
        small, info = mv.prune_structured(M1, SPLITS["X_train"], SPLITS["y_train"], 0.35, epochs=60)
        self.assertEqual(list(small.layers), [6, 21, 10, 3])
        self.assertEqual(info["removed"], [11, 6])
        self.assertEqual(small.layers[-1], 3)                           # the output layer is untouched
        self.assertLess(small.n_params, M1.n_params)
        acc = float(np.mean(small.predict(SPLITS["X_val"]) == SPLITS["y_val"]))
        self.assertGreater(acc, 0.95)


class TestAccounting(unittest.TestCase):
    def test_counts_and_payload(self):
        self.assertEqual(mv.n_params([6, 32, 16, 3]), 803)
        self.assertEqual(mv.macs([6, 32, 16, 3]), 752)
        self.assertEqual(mv.payload_bytes("fp32", [6, 32, 16, 3])["total"], 3212)
        i8 = mv.payload_bytes("int8", [6, 32, 16, 3])
        self.assertEqual((i8["weights"], i8["biases"]), (752, 204))
        self.assertEqual(mv.n_params([6, 21, 10, 3]), 400)
        self.assertEqual(mv.arena_bytes("fp32", [6, 32, 16, 3]), 4 * 48)
        self.assertEqual(mv.arena_bytes("int8", [6, 32, 16, 3]), 48)

    def test_int8_file_estimate_is_smaller_than_fp32(self):
        fp = mv.estimated_file_kb("fp32", [6, 32, 16, 3])
        i8 = mv.estimated_file_kb("int8", [6, 32, 16, 3])
        self.assertLess(i8[1], fp[1])
        self.assertLess(mv.estimated_file_kb("int8", [6, 21, 10, 3])[1], i8[1])


class TestSavedResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(HERE, "variants_results.json")) as f:
            cls.r = json.load(f)

    def test_calibration_and_pruning(self):
        self.assertGreaterEqual(self.r["n_calibration"], 200)
        self.assertEqual(self.r["sizes"]["M3"], [6, 21, 10, 3])

    def test_every_variant_exceeds_95_percent_critical_recall_on_both_sets(self):
        for name, m in self.r["models"].items():
            self.assertGreater(m["val"]["critical_recall"], 0.95, name)
            self.assertGreater(m["extended"]["critical_recall"], 0.95, name)

    def test_vibration_only_fault_is_missed_by_every_variant(self):
        v = self.r["vibration_only"]
        for name in ("M1", "M2", "M3", "required_model"):
            self.assertEqual(v[name], [v["windows"], 0, 0], name)


class TestBenchmarkHarness(unittest.TestCase):
    def test_warmup_runs_are_discarded_and_200_are_timed(self):
        calls = []
        lat = bm.time_callable(lambda x: calls.append(1), np.zeros((5, 6)))
        self.assertEqual(len(calls), 210)
        self.assertEqual(len(lat), 200)
        self.assertTrue(np.all(lat >= 0))

    def test_p95_is_the_95th_percentile_of_the_timed_runs(self):
        fake = np.arange(1.0, 201.0)
        self.assertAlmostEqual(float(np.percentile(fake, 95)), 190.05)

    def test_energy_is_power_times_time(self):
        # 15 W x 12.5% = 1.875 W; x 0.1 ms = 0.1875 mJ
        self.assertAlmostEqual(bm.energy_mj(0.1, 12.5, 15.0), 0.1875)
        self.assertAlmostEqual(bm.energy_mj(2.0, 100.0, 28.0), 56.0)

    def test_cpu_percent_comes_from_psutil(self):
        pct = bm.cpu_percent_during(lambda x: sum(range(200)), np.zeros((2, 6)), seconds=0.2)
        self.assertTrue(0.0 <= pct <= 100.0)

    def test_evaluate_callable_reports_critical_recall(self):
        X = np.zeros((4, 6))
        y = np.array([0, 2, 2, 1])
        answers = iter([[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
        r = bm.evaluate_callable(lambda x: next(answers), X, y)
        self.assertEqual((r["accuracy"], r["critical_recall"], r["n_critical"]), (0.75, 0.5, 2))

    def test_benchmark_variant_returns_all_five_metrics(self):
        r = bm.benchmark_variant("x", lambda z: np.array([1.0, 0, 0]), np.zeros((3, 6)), np.zeros(3, int), 1234,
                                 15.0, runs=20, warmup=2, cpu_seconds=0.1)
        for key in ("mean_latency_ms", "p95_latency_ms", "file_kb", "accuracy_percent", "energy_mj"):
            self.assertIn(key, r)
        self.assertEqual(r["file_kb"], 1.234)
        self.assertEqual(r["accuracy_percent"], 100.0)
        self.assertEqual((r["runs"], r["warmup"]), (20, 2))

    def test_numpy_self_check_callables_run(self):
        fns, X, y = bm.numpy_callables()
        for name, fn in fns.items():
            self.assertEqual(bm.evaluate_callable(fn, X, y)["accuracy"], 1.0, name)


class FakeInterpreter:
    def __init__(self, model_path=None):
        self.set_value = None

    def allocate_tensors(self): pass
    def get_input_details(self): return [{"index": 0, "dtype": np.int8, "quantization": (0.5, -10)}]
    def get_output_details(self): return [{"index": 1, "dtype": np.int8, "quantization": (0.25, 0)}]
    def set_tensor(self, i, x): self.set_value = x
    def invoke(self): pass
    def get_tensor(self, i): return np.array([[4, 8, 100]], dtype=np.int8)


class TestTfliteCallable(unittest.TestCase):
    def test_int8_input_is_quantised_and_output_dequantised(self):
        saved = {k: sys.modules.get(k) for k in ("tflite_runtime", "tflite_runtime.interpreter")}
        mod = types.ModuleType("tflite_runtime.interpreter")
        mod.Interpreter = FakeInterpreter
        sys.modules["tflite_runtime"], sys.modules["tflite_runtime.interpreter"] = types.ModuleType("tflite_runtime"), mod
        try:
            fn = bm.tflite_callable("model.tflite")
            out = fn(np.array([10.0, 0.0, -10.0, 1000.0, -1000.0, 1.0]))
            np.testing.assert_allclose(out, [1.0, 2.0, 25.0])
        finally:
            for k, v in saved.items():
                sys.modules.pop(k, None) if v is None else sys.modules.__setitem__(k, v)


class TestTensorFlowScript(unittest.TestCase):
    """The TensorFlow script cannot run here; check it contains the specified settings."""

    @classmethod
    def setUpClass(cls):
        parts = []
        for name in ("convert_ptq.py", "prune_quantise.py", "build_variants_tf.py"):
            with open(os.path.join(HERE, name)) as f:
                text = f.read()
            ast.parse(text)
            parts.append(text)
        cls.src = "\n".join(parts)

    def test_required_settings_are_present(self):
        for token in ("tf.lite.Optimize.DEFAULT", "representative_dataset", "TFLITE_BUILTINS_INT8", "tf.int8",
                      "PolynomialDecay", "final_sparsity=a.sparsity", "prune_structured",
                      "len(X_cal) < 200"):
            self.assertIn(token, self.src, token)

    def test_default_sparsity_is_35_percent(self):
        self.assertIn('"--sparsity", type=float, default=0.35', self.src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
