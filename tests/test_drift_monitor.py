"""Tests for drift_monitor.py and drift_demo.py (Task E1). Run with:  python3 -m unittest -v test_drift_monitor

Needs training_stats.npy, model_weights.npz and reference_dist.json in the current folder.
"""

import hashlib
import json
import math
import os
import re
import tempfile
import unittest

import numpy as np

import drift_demo
import drift_monitor as dm

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
REF_PATH = os.path.join(HERE, "reference_dist.json")
CLEAN = [1.0, 0.0, 0.0]        # class probabilities of a clean Normal window
FAULT = [0.0, 0.0, 1.0]        # a confident Critical window


def clean_reference():
    return dm.build_reference([CLEAN] * 300, "m.npz", "abc", "test")


class TestPsiMaths(unittest.TestCase):
    def test_identical_distributions_give_zero(self):
        self.assertAlmostEqual(dm.psi([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]), 0.0)

    def test_known_value(self):
        # (0.25)ln(2) + 0 + (-0.125)ln(0.5) + (-0.125)ln(0.5) = 0.34657
        self.assertAlmostEqual(dm.psi([0.25] * 4, [0.5, 0.25, 0.125, 0.125]), 0.346574, places=5)

    def test_is_symmetric_for_equal_floors(self):
        a, b = [0.7, 0.1, 0.1, 0.1], [0.2, 0.3, 0.3, 0.2]
        self.assertAlmostEqual(dm.psi(a, b), dm.psi(b, a))

    def test_zero_proportions_do_not_blow_up(self):
        v = dm.psi([0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0])
        self.assertTrue(math.isfinite(v) and v > 0)
        self.assertAlmostEqual(v, 2 * (1 - 1e-4) * math.log((1 - 1e-4) / 1e-4) , delta=0.01)

    def test_bins_follow_the_specification(self):
        self.assertEqual(dm.BIN_EDGES, (0.0, 0.25, 0.50, 0.75, 1.0))
        self.assertEqual(dm.bin_counts([0.0, 0.2499]), [2, 0, 0, 0])
        self.assertEqual(dm.bin_counts([0.25, 0.4999]), [0, 2, 0, 0])
        self.assertEqual(dm.bin_counts([0.5, 0.7499]), [0, 0, 2, 0])
        self.assertEqual(dm.bin_counts([0.75, 1.0]), [0, 0, 0, 2])          # the last bin includes 1.0

    def test_status_thresholds(self):
        self.assertEqual([dm.status_for(x) for x in (0.0, 0.0999, 0.10, 0.25, 0.2501)],
                         ["stable", "stable", "moderate", "moderate", "drift"])

    def test_scores(self):
        s = dm.scores_from_probs([0.1, 0.2, 0.7])
        self.assertEqual((s["max_confidence"], s["normal_confidence"]), (0.7, 0.1))


class TestReferenceFile(unittest.TestCase):
    def test_round_trip_and_counts(self):
        ref = dm.build_reference([CLEAN] * 300, "m.npz", "abc", "src")
        self.assertEqual(ref["n_windows"], 300)
        self.assertEqual(sum(ref["scores"]["normal_confidence"]["counts"]), 300)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reference_dist.json")
            dm.save_reference(ref, path)
            self.assertEqual(dm.load_reference(path), ref)

    def test_load_rejects_wrong_bins_or_proportions(self):
        ref = clean_reference()
        with tempfile.TemporaryDirectory() as d:
            ref["bin_edges"] = [0, 0.5, 1.0]
            path = os.path.join(d, "bad.json")
            with open(path, "w") as f:
                json.dump(ref, f)
            with self.assertRaises(ValueError):
                dm.load_reference(path)
            ref = clean_reference()
            ref["scores"]["normal_confidence"]["proportions"] = [0.5, 0, 0, 0]
            with open(path, "w") as f:
                json.dump(ref, f)
            with self.assertRaises(ValueError):
                dm.load_reference(path)

    def test_shipped_reference_matches_the_specification(self):
        ref = dm.load_reference(REF_PATH)
        self.assertEqual(ref["n_windows"], 300)
        self.assertEqual(ref["bin_edges"], [0.0, 0.25, 0.5, 0.75, 1.0])
        with open(os.path.join(HERE, "model_weights.npz"), "rb") as f:
            self.assertEqual(ref["model"]["sha256"], hashlib.sha256(f.read()).hexdigest()[:12])
        for name in dm.SCORE_NAMES:
            self.assertEqual(sum(ref["scores"][name]["counts"]), 300)


class TestMonitorBehaviour(unittest.TestCase):
    def make(self, **kw):
        lines = []
        mon = dm.PsiMonitor(clean_reference(), emit=lines.append, time_label=lambda t: f"{t:.0f}", **kw)
        return mon, lines

    def feed(self, mon, start, probs_list):
        out = []
        for i, p in enumerate(probs_list):
            out += mon.add(start + 10.0 * i, p)
        return out

    def test_evaluates_every_60_seconds_of_data_time(self):
        mon, lines = self.make()
        ev = self.feed(mon, 1000.0, [CLEAN] * 31)             # 300 s of data
        self.assertEqual([e["t"] for e in ev], [1060, 1120, 1180, 1240, 1300])
        self.assertEqual(len(lines), 5)

    def test_warms_up_until_100_inferences_are_available(self):
        mon, lines = self.make()
        ev = self.feed(mon, 0.0, [CLEAN] * 120)
        self.assertTrue(all(e["psi"] is None for e in ev if e["n"] < 100))
        self.assertTrue(any("collecting" in l for l in lines))
        self.assertTrue(all(e["psi"] is not None for e in ev if e["n"] == 100))

    def test_window_is_the_last_100_inferences(self):
        mon, _ = self.make()
        self.feed(mon, 0.0, [FAULT] * 60 + [CLEAN] * 100)       # the faults have rolled out of the window
        self.assertAlmostEqual(mon.current_psi(), 0.0)
        mon2, _ = self.make()
        self.feed(mon2, 0.0, [CLEAN] * 70 + [FAULT] * 30)
        self.assertGreater(mon2.current_psi(), 0.25)

    def test_alert_line_has_the_exact_format_and_only_appears_above_threshold(self):
        mon, lines = self.make()
        self.feed(mon, 0.0, [CLEAN] * 130)
        self.assertFalse([l for l in lines if "DRIFT ALERT" in l])
        self.feed(mon, 1300.0, [FAULT] * 60)
        alerts = [l for l in lines if "DRIFT ALERT" in l]
        self.assertTrue(alerts)
        for a in alerts:
            self.assertRegex(a, r"^\[LOGIBRIDGE DRIFT ALERT\] PSI=\d+\.\d{3}$")

    def test_every_evaluation_prints_the_current_psi(self):
        mon, lines = self.make()
        self.feed(mon, 0.0, [CLEAN] * 130)
        psi_lines = [l for l in lines if "PSI=" in l and "ALERT" not in l]
        self.assertTrue(psi_lines)
        self.assertTrue(all(re.search(r"PSI=\d+\.\d{3}", l) for l in psi_lines))

    def test_threshold_is_strictly_greater_than_0_25(self):
        mon, _ = self.make(alert_threshold=0.25)
        events = self.feed(mon, 0.0, [CLEAN] * 100 + [FAULT] * 4)
        self.assertTrue(all(e["alert"] == (e["psi"] is not None and e["psi"] > 0.25) for e in events))

    def test_unknown_score_is_rejected(self):
        with self.assertRaises(ValueError):
            dm.PsiMonitor(clean_reference(), score="accuracy")

    def test_max_confidence_does_not_see_a_confident_fault(self):
        ref = clean_reference()
        mon = dm.PsiMonitor(ref, score="max_confidence", emit=lambda l: None)
        self.feed(mon, 0.0, [CLEAN] * 100 + [FAULT] * 100)
        self.assertAlmostEqual(mon.current_psi(), 0.0)      # a confident fault is as confident as clean data


class TestDemoEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lines = []
        cls.result = drift_demo.run_demo(model_path=os.path.join(HERE, "model_weights.npz"),
                                         stats_path=os.path.join(HERE, "training_stats.npy"),
                                         reference_path=REF_PATH, out=cls.lines.append)
        cls.m = cls.result["monitors"]["normal_confidence"]

    def test_psi_crosses_0_25_within_5_minutes_of_injection(self):
        self.assertIsNotNone(self.m["crossed_after_s"])
        self.assertLessEqual(self.m["crossed_after_s"], 300)

    def test_no_false_alarms_before_injection(self):
        self.assertEqual(self.m["false_alarms_before_injection"], 0)

    def test_psi_returns_below_0_10_after_clean_data_is_restored(self):
        self.assertIsNotNone(self.m["recovered_after_s"])
        self.assertLessEqual(self.m["recovered_after_s"], 1100)     # the 100-inference window needs about 1,000 s to flush

    def test_the_demo_output_contains_the_required_lines(self):
        text = "\\n".join(self.lines)
        self.assertRegex(text, r"\[LOGIBRIDGE DRIFT ALERT\] PSI=\d+\.\d{3}")
        self.assertIn("INJECT: --anomaly combined activated", text)
        self.assertIn("RESTORE: clean data back", text)

    def test_the_literal_confidence_score_never_crosses_for_this_model(self):
        lit = self.result["monitors"]["max_confidence"]
        self.assertLess(lit["peak_psi"], 0.01)
        self.assertIsNone(lit["crossed_after_s"])

    def test_script_exit_code_reflects_the_acceptance_criteria(self):
        self.assertEqual(drift_demo.main(["--pace", "0"]), 0)
        self.assertEqual(drift_demo.main(["--score", "max_confidence"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
