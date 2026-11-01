"""Tests for ota_strategy_cost.py (Task E3). Run with:  python3 -m unittest -v test_ota_strategy"""
import io
import contextlib
import unittest

import ota_strategy_cost as oc


class TestCosts(unittest.TestCase):
    def setUp(self):
        self.r = oc.costs()
        self.rows = self.r["rows"]

    def test_full_replacement_is_85_copies_of_280_kb(self):
        full = self.rows["full"]
        self.assertEqual(full["total_bytes"], 85 * 280_000)
        self.assertAlmostEqual(full["total_mb"], 23.8)
        self.assertAlmostEqual(full["cost"], 2.38)

    def test_canary_sends_the_model_to_every_truck_once_plus_a_little_telemetry(self):
        c = self.rows["canary (passes)"]
        self.assertEqual(c["model"], 85 * 280_000)
        self.assertEqual(c["telemetry"], 10 * 7 * 7_200)               # 10 trucks, 7 days, 300 B an hour
        self.assertAlmostEqual(c["cost"], 2.4304)

    def test_failed_canary_is_cheaper_because_75_trucks_never_receive_the_model(self):
        f = self.rows["canary (fails, rolled back)"]
        self.assertEqual(f["model"], 2 * 10 * 280_000)                  # 10 trucks out, 10 back
        self.assertLess(f["cost"], self.rows["full"]["cost"])

    def test_shadow_with_on_device_summaries(self):
        s = self.rows["shadow (on-device summary)"]
        per_truck_day = 24 * 300 + 8_640 * 0.005 * 150                  # 7,200 + 6,480 B
        self.assertEqual(self.r["per_truck_shadow_day_bytes"], per_truck_day)
        self.assertAlmostEqual(s["telemetry"], 85 * 7 * per_truck_day + 85 * 1_000)
        self.assertAlmostEqual(s["cost"], 3.2025, places=3)

    def test_shadow_streaming_every_window_is_costly_but_still_small(self):
        s = self.rows["shadow (every window streamed)"]
        self.assertAlmostEqual(s["telemetry"], 85 * 7 * 864_000 + 85_000)
        self.assertAlmostEqual(s["cost"], 53.7965, places=3)
        self.assertLess(s["cost"], 100)

    def test_cost_ordering(self):
        c = {k: v["cost"] for k, v in self.rows.items()}
        self.assertLess(c["full"], c["canary (passes)"])
        self.assertLess(c["canary (passes)"], c["shadow (on-device summary)"])
        self.assertLess(c["shadow (on-device summary)"], c["shadow (every window streamed)"])

    def test_yearly_cost_uses_a_6_week_cycle(self):
        self.assertAlmostEqual(self.r["cycles_per_year"], 52 / 6)
        self.assertAlmostEqual(self.rows["full"]["cost_per_year"], 2.38 * 52 / 6)

    def test_scales_to_the_full_fleet(self):
        big = oc.costs(trucks=265)["rows"]["full"]
        self.assertAlmostEqual(big["total_mb"], 265 * 0.28)

    def test_cli_prints_a_table(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(oc.main([]), 0)
        text = buf.getvalue()
        for name in ("full", "canary (passes)", "shadow (on-device summary)"):
            self.assertIn(name, text)
        self.assertIn("2.38", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
