#!/usr/bin/env python3
"""Build results/benchmark_results.csv from everything that has been measured (Task F2).

Inputs
  variants_results.json     accuracy, Critical recall, noise test, sizes (numpy reference of the three variants)
  benchmark_results.json    OPTIONAL: the TFLite run from `benchmark.py --tflite ...` on your laptop (latency, p95,
                            file size, CPU%, energy). Variants without it are marked "pending".
  benchmark_selfcheck.json  the harness self-check on numpy callables (kept as clearly labelled extra rows)

Output columns: variant, method, mean_latency_ms, p95_latency_ms, file_kb, file_kb_est_low, file_kb_est_high,
payload_bytes, params, macs, accuracy_percent, critical_recall_percent, false_alarm_12sigma_percent,
critical_recall_12sigma_percent, cpu_percent, tdp_watts, energy_mj, status

Run:  python3 build_results_csv.py [--variants variants_results.json] [--tflite benchmark_results.json] [--out benchmark_results.csv]
"""
import argparse
import csv
import json
import os
import sys

COLUMNS = ["variant", "method", "mean_latency_ms", "p95_latency_ms", "file_kb", "file_kb_est_low", "file_kb_est_high",
           "payload_bytes", "params", "macs", "accuracy_percent", "critical_recall_percent",
           "false_alarm_12sigma_percent", "critical_recall_12sigma_percent", "cpu_percent", "tdp_watts", "energy_mj",
           "status"]
NAMES = {"M1": "M1 FP32 baseline", "M2": "M2 PTQ INT8", "M3": "M3 pruned + PTQ INT8"}


def build_rows(variants: dict, tflite: list | None = None, selfcheck: list | None = None) -> list:
    tflite_by_key = {}
    for r in tflite or []:
        tflite_by_key[r["variant"].split()[0]] = r
    noise12 = next(row for row in variants["noise"] if row["sigma"] == 12)
    rows = []
    for key in ("M1", "M2", "M3"):
        m = variants["models"][key]
        row = {"variant": NAMES[key], "method": "pending", "params": m["params"], "macs": m["macs"],
               "payload_bytes": m["payload"]["total"], "file_kb_est_low": round(m["file_kb_estimate"][0], 2),
               "file_kb_est_high": round(m["file_kb_estimate"][1], 2),
               "accuracy_percent": round(100 * m["val"]["accuracy"], 2),
               "critical_recall_percent": round(100 * m["val"]["critical_recall"], 2),
               "false_alarm_12sigma_percent": round(100 * noise12[key]["false_alarm"], 2),
               "critical_recall_12sigma_percent": round(100 * noise12[key]["critical_recall"], 2),
               "status": "pending: run benchmark.py --tflite on a laptop; accuracy from the numpy reference of TFLite arithmetic"}
        t = tflite_by_key.get(key)
        if t:
            row.update(method="tflite", mean_latency_ms=round(t["mean_latency_ms"], 5), p95_latency_ms=round(t["p95_latency_ms"], 5),
                       file_kb=round(t["file_kb"], 3), cpu_percent=round(t["cpu_percent"], 1), tdp_watts=t["tdp_watts"],
                       energy_mj=round(t["energy_mj"], 6), accuracy_percent=round(t["accuracy_percent"], 2),
                       critical_recall_percent=round(t["critical_recall_percent"], 2), status="measured")
        rows.append(row)
    for r in selfcheck or []:
        rows.append({"variant": r["variant"], "method": "numpy-self-check (not TFLite)",
                     "mean_latency_ms": round(r["mean_latency_ms"], 5), "p95_latency_ms": round(r["p95_latency_ms"], 5),
                     "accuracy_percent": round(r["accuracy_percent"], 2),
                     "critical_recall_percent": round(r["critical_recall_percent"], 2),
                     "cpu_percent": round(r["cpu_percent"], 1), "tdp_watts": r["tdp_watts"],
                     "energy_mj": round(r["energy_mj"], 6),
                     "status": "harness check only: numpy emulation on the sandbox CPU; not comparable with TFLite"})
    return rows


def write_rows(rows: list, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in COLUMNS})


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", default="variants_results.json")
    p.add_argument("--tflite", default="benchmark_results.json", help="TFLite run (optional; ignored if missing)")
    p.add_argument("--selfcheck", default="benchmark_selfcheck.json")
    p.add_argument("--out", default=os.path.join(here, "results", "benchmark_results.csv"))
    a = p.parse_args(argv)
    variants = json.load(open(a.variants))
    tflite = json.load(open(a.tflite)) if os.path.exists(a.tflite) else None
    selfcheck = json.load(open(a.selfcheck)) if os.path.exists(a.selfcheck) else None
    rows = build_rows(variants, tflite, selfcheck)
    write_rows(rows, a.out)
    measured = sum(1 for r in rows if r["status"] == "measured")
    print(f"Wrote {a.out}: {len(rows)} rows, {measured} of 3 variants measured with TFLite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
