#!/usr/bin/env python3
"""Bandwidth cost per OTA update cycle for three strategies (Task E3).

Inputs from the brief: model 280 KB (INT8 TFLite), M2M SIM at Rs 0.10 per MB, 85 trucks, a new model every
6 weeks. 1 MB = 1,000,000 bytes and 1 KB = 1,000 bytes, as in Task A1.

Strategies
  full    send the model to all trucks at once
  canary  send to 10 trucks, observe them, then send to the other 75 (or roll the 10 back)
  shadow  send to all trucks; the new model runs next to the old one on every truck without acting; trucks
          report how the two compare; promote with a small config message
Assumptions that are NOT in the brief (change them on the command line):
  shadow/canary observation period 7 days; one inference window every 10 s (8,640 a day)
  summary report 300 B an hour; a window where the models disagree is reported as a 150 B record; 0.5% of
  windows disagree; naive shadow reporting streams 100 B for every window; promote message 1 KB per truck

Run:  python3 ota_strategy_cost.py [--trucks 85] [--model-bytes 280000] [--days 7] ...
"""

from __future__ import annotations

import argparse
import sys

MB = 1_000_000


def costs(model_bytes=280_000, trucks=85, canary_trucks=10, tariff=0.10, days=7, windows_per_day=8_640,
          summary_bytes_per_hour=300, disagree_rate=0.005, disagree_record_bytes=150,
          naive_record_bytes=100, promote_bytes=1_000, cycle_weeks=6) -> dict:
    summary_day = 24 * summary_bytes_per_hour
    disagree_day = windows_per_day * disagree_rate * disagree_record_bytes
    shadow_day = summary_day + disagree_day
    naive_day = windows_per_day * naive_record_bytes
    cycles_per_year = 52 / cycle_weeks

    rows = {
        "full": {"model": trucks * model_bytes, "telemetry": 0},
        "canary (passes)": {"model": trucks * model_bytes, "telemetry": canary_trucks * days * summary_day},
        "canary (fails, rolled back)": {"model": 2 * canary_trucks * model_bytes,
                                        "telemetry": canary_trucks * days * summary_day},
        "shadow (on-device summary)": {"model": trucks * model_bytes,
                                       "telemetry": trucks * days * shadow_day + trucks * promote_bytes},
        "shadow (every window streamed)": {"model": trucks * model_bytes,
                                           "telemetry": trucks * days * naive_day + trucks * promote_bytes},
    }
    for r in rows.values():
        r["total_bytes"] = r["model"] + r["telemetry"]
        r["total_mb"] = r["total_bytes"] / MB
        r["cost"] = r["total_mb"] * tariff
        r["cost_per_year"] = r["cost"] * cycles_per_year
    return {"rows": rows, "cycles_per_year": cycles_per_year, "per_truck_shadow_day_bytes": shadow_day,
            "per_truck_naive_day_bytes": naive_day, "trucks": trucks, "model_bytes": model_bytes}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trucks", type=int, default=85)
    p.add_argument("--canary-trucks", type=int, default=10)
    p.add_argument("--model-bytes", type=int, default=280_000)
    p.add_argument("--tariff", type=float, default=0.10, help="Rs per MB")
    p.add_argument("--days", type=int, default=7, help="shadow / canary observation days")
    a = p.parse_args(argv)
    res = costs(a.model_bytes, a.trucks, a.canary_trucks, a.tariff, a.days)
    print(f"{a.trucks} trucks, model {a.model_bytes / 1000:.0f} KB, Rs {a.tariff:.2f} per MB, "
          f"{res['cycles_per_year']:.2f} cycles a year")
    print(f"{'strategy':32s} {'model MB':>9s} {'telemetry MB':>13s} {'total MB':>9s} {'Rs/cycle':>9s} {'Rs/year':>9s}")
    for name, r in res["rows"].items():
        print(f"{name:32s} {r['model'] / MB:9.2f} {r['telemetry'] / MB:13.3f} {r['total_mb']:9.3f} "
              f"{r['cost']:9.2f} {r['cost_per_year']:9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
