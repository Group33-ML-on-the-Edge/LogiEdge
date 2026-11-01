"""Tests for generate_dataset.py and train_model.py (Task D1).

Run with:  python3 -m unittest -v test_training
Needs training_stats.npy, training_dataset.npz and model_weights.npz in the current folder
(created by preprocessing.py make-stats, generate_dataset.py and train_model.py).
"""

import json
import os
import tempfile
import unittest

import numpy as np

import generate_dataset as gd
import train_model as tm
from preprocessing import Normaliser

NORM = Normaliser.load("training_stats.npy")


class TestDatasetGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = gd.build_dataset()

    def test_class_sizes_match_specification(self):
        counts = np.bincount(self.ds["y"]).tolist()
        self.assertEqual(counts, [117, 87, 87])           # 20 / 15 / 15 minutes at 30 s windows, 10 s step
        for got, target in zip(counts, (120, 90, 90)):
            self.assertAlmostEqual(got, target, delta=4)  # "approx. 120 / 90 / 90 windows"

    def test_modes_and_durations(self):
        self.assertEqual([(c, m, d) for c, m, d, _ in gd.SPECS],
                         [(0, "none", 1200), (1, "temp_drift", 900), (2, "combined", 900)])

    def test_windows_are_30s_with_10s_step(self):
        for c in range(3):
            ends = self.ds["window_end_s"][self.ds["y"] == c]
            self.assertEqual(ends[0], 30)
            self.assertTrue(np.all(np.diff(ends) == 10))

    def test_each_class_has_the_expected_signature(self):
        raw, y = self.ds["raw"], self.ds["y"]
        n = raw[y == 0]; d = raw[y == 1]; c = raw[y == 2]
        self.assertAlmostEqual(n[:, 0].mean(), 4.0, delta=0.05)         # temperature at setpoint
        self.assertAlmostEqual(n[:, 3].mean(), 0.45, delta=0.01)        # vibration RMS normal
        self.assertAlmostEqual(d[:, 2].mean(), 4.8, delta=0.1)          # drift 0.08 degC/s = 4.8 degC/min
        self.assertAlmostEqual(d[:, 3].mean(), 0.45, delta=0.02)        # temp_drift leaves vibration normal
        self.assertAlmostEqual(c[:, 2].mean(), 4.8, delta=0.1)          # combined drifts too
        self.assertAlmostEqual(c[:, 3].mean(), 1.2, delta=0.03)         # and has the bearing-wear vibration

    def test_reproducible(self):
        again = gd.build_dataset()
        np.testing.assert_array_equal(again["raw"], self.ds["raw"])
        other = gd.build_dataset(seed_offset=1)
        self.assertFalse(np.allclose(other["raw"], self.ds["raw"]))

    def test_state_labels_expose_the_label_mismatch(self):
        # by the Task C thresholds, almost every temp_drift window is already past +3 degC
        ys = self.ds["y_state"][self.ds["y"] == 1]
        self.assertEqual(np.bincount(ys, minlength=3).tolist(), [0, 1, 86])
        self.assertEqual(set(self.ds["y_state"][self.ds["y"] == 0]), {0})
        self.assertEqual(set(self.ds["y_state"][self.ds["y"] == 2]), {2})

    def test_state_label_function(self):
        self.assertEqual(gd.state_label("none", 600), 0)
        self.assertEqual(gd.state_label("temp_drift", 10), 0)           # d = 0.8
        self.assertEqual(gd.state_label("temp_drift", 30), 1)           # d = 2.4
        self.assertEqual(gd.state_label("temp_drift", 60), 2)           # d = 4.8
        self.assertEqual(gd.state_label("combined", 30), 2)             # vibration fault with d > 1

    def test_saved_files_are_consistent(self):
        d = np.load("training_dataset.npz")
        self.assertEqual(d["raw"].shape, (291, 6))
        np.testing.assert_allclose(d["z"], NORM.transform(d["raw"]))
        np.testing.assert_allclose(d["raw"], self.ds["raw"], rtol=1e-10, atol=1e-12)
        with open("training_dataset.csv") as f:
            self.assertEqual(len(f.readlines()), 292)                   # header + 291 windows

    def test_onset_runs_label_transition_windows(self):
        extra = gd.build_onset_dataset(onsets=(200,), duration=400)
        ends, y = extra["window_end_s"], extra["y"]
        self.assertTrue(np.all(y[ends <= 215] == 0))                    # before (and just after) onset: Normal
        self.assertTrue(np.all(y[(ends > 215) & (extra["y"] >= 0)][:1] >= 1))
        self.assertEqual(set(y[:len(y) // 2][ends[:len(y) // 2] > 215]), {1})
        self.assertEqual(set(y[len(y) // 2:][ends[len(y) // 2:] > 215]), {2})


class TestModel(unittest.TestCase):
    def test_architecture(self):
        m = tm.MLP()
        self.assertEqual(m.layers, (6, 32, 16, 3))
        self.assertEqual(m.n_params, 6 * 32 + 32 + 32 * 16 + 16 + 16 * 3 + 3)       # 803
        self.assertEqual(m.n_params, 803)

    def test_relu_hidden_layers_and_softmax_output(self):
        m = tm.MLP(seed=1)
        acts = m._forward(np.random.default_rng(0).normal(size=(5, 6)) * 50)
        self.assertTrue(np.all(acts[1] >= 0) and np.all(acts[2] >= 0))            # ReLU outputs
        p = m.predict_proba(np.ones((4, 6)))
        np.testing.assert_allclose(p.sum(axis=1), 1.0)

    def test_learns_a_simple_problem(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(-3, 1, (60, 6)), rng.normal(0, 1, (60, 6)), rng.normal(3, 1, (60, 6))])
        y = np.repeat([0, 1, 2], 60)
        m = tm.MLP(seed=0); m.fit(X, y, epochs=100)
        self.assertGreater(np.mean(m.predict(X) == y), 0.95)

    def test_save_and_load_round_trip(self):
        m = tm.MLP(seed=2)
        X = np.random.default_rng(1).normal(size=(20, 6))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.npz")
            m.save(path)
            m2 = tm.MLP.load(path)
        np.testing.assert_allclose(m.predict_proba(X), m2.predict_proba(X))


class TestEvaluation(unittest.TestCase):
    def test_stratified_split(self):
        y = np.repeat([0, 1, 2], [117, 87, 87])
        tr, va = tm.stratified_split(y, 0.2, 42)
        self.assertEqual((len(tr), len(va)), (234, 57))
        self.assertEqual(np.bincount(y[va]).tolist(), [23, 17, 17])
        self.assertEqual(len(set(tr) & set(va)), 0)
        np.testing.assert_array_equal(tm.stratified_split(y, 0.2, 42)[1], va)      # deterministic

    def test_wilson_interval(self):
        lo, hi = tm.wilson_interval(56, 57)
        self.assertAlmostEqual(lo, 0.907, delta=0.002)
        self.assertAlmostEqual(hi, 0.997, delta=0.002)

    def test_confusion_precision_recall(self):
        r = tm.evaluate([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0])
        self.assertEqual(r["confusion"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        self.assertAlmostEqual(r["accuracy"], 4 / 6)
        self.assertEqual(r["recall"][0], 0.5)
        self.assertAlmostEqual(r["precision"][1], 2 / 3)


class TestTrainingGate(unittest.TestCase):
    def test_saved_model_exceeds_88_percent_on_the_validation_split(self):
        X, y, _ = tm.load_dataset("training_dataset.npz", NORM)
        _, va = tm.stratified_split(y, 0.2, 42)
        model = tm.MLP.load("model_weights.npz")
        acc = float(np.mean(model.predict(X[va]) == y[va]))
        self.assertGreater(acc, tm.ACCURACY_THRESHOLD)

    def test_training_run_passes_and_saves(self):
        with tempfile.TemporaryDirectory() as d:
            cwd = os.getcwd()
            args = ["--out", os.path.join(d, "w.npz"), "--meta", os.path.join(d, "m.json")]
            self.assertEqual(tm.main(args), 0)
            self.assertTrue(os.path.exists(os.path.join(d, "w.npz")))
            with open(os.path.join(d, "m.json")) as f:
                meta = json.load(f)
            self.assertGreater(meta["validation"]["accuracy"], 0.88)
            self.assertEqual(meta["layers"], [6, 32, 16, 3])
            self.assertEqual(os.getcwd(), cwd)

    def test_gate_blocks_a_model_below_88_percent(self):
        d0 = np.load("training_dataset.npz")
        shuffled = np.random.default_rng(0).permutation(d0["y"])           # labels no longer match features
        with tempfile.TemporaryDirectory() as d:
            ds = os.path.join(d, "bad.npz")
            np.savez(ds, raw=d0["raw"], y=shuffled, y_state=d0["y_state"])
            out = os.path.join(d, "w.npz")
            code = tm.main(["--dataset", ds, "--out", out, "--meta", os.path.join(d, "m.json"), "--epochs", "60"])
            self.assertEqual(code, 1)
            self.assertFalse(os.path.exists(out))                          # no model saved on failure

    def test_augmented_training_also_passes(self):
        with tempfile.TemporaryDirectory() as d:
            code = tm.main(["--augment-onset", "--epochs", "150", "--out", os.path.join(d, "w.npz"),
                            "--meta", os.path.join(d, "m.json")])
            self.assertEqual(code, 0)


class TestEndToEnd(unittest.TestCase):
    def test_simulator_to_prediction(self):
        """Fresh simulator runs -> preprocessing -> saved model give the right class."""
        model = tm.MLP.load("model_weights.npz")
        for mode, expected in (("none", 0), ("temp_drift", 1), ("combined", 2)):
            msgs = gd.run_simulator(mode, 300, seed=777)
            wins = gd.messages_to_windows(msgs, NORM)
            pred = model.predict(np.array([w.z for w in wins]))
            self.assertGreater(np.mean(pred == expected), 0.9, mode)


if __name__ == "__main__":
    unittest.main(verbosity=2)
