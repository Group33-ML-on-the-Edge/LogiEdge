#!/usr/bin/env python3
"""Pareto chart of the three variants from benchmark_results.csv (Task F2).

Left panel (always available): model size against false alarms under 12-sigma input noise (lower-left is better).
Right panel: mean latency against false alarms, drawn only for variants measured with TFLite; until then it says so.
Points on the Pareto frontier (not beaten on both axes) are joined by a line. The chart is marked PROVISIONAL until
all three variants have TFLite latency.

Run:  python3 make_pareto.py [--csv benchmark_results.csv] [--out pareto_chart.png]
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"M1": "#2E7D32", "M2": "#0B5A80", "M3": "#EF6C00"}


def pareto_front(points: list) -> list:
    """Indices of points not dominated by another (both coordinates: lower is better)."""
    front = []
    for i, (xi, yi) in enumerate(points):
        if not any((xj <= xi and yj <= yi) and (xj < xi or yj < yi) for j, (xj, yj) in enumerate(points) if j != i):
            front.append(i)
    return front


def load(csv_path: str) -> list:
    with open(csv_path) as f:
        return [r for r in csv.DictReader(f) if not r["method"].startswith("numpy-self-check")]


def num(x):
    return float(x) if x not in ("", None) else None


def draw(rows: list, out: str) -> dict:
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3), dpi=200)
    sizes, fas, labels, keys = [], [], [], []
    for r in rows:
        kb = num(r["file_kb"]) if r["file_kb"] else (num(r["file_kb_est_low"]) + num(r["file_kb_est_high"])) / 2
        sizes.append(kb); fas.append(num(r["false_alarm_12sigma_percent"])); labels.append(r["variant"]); keys.append(r["variant"].split()[0])
    front = pareto_front(list(zip(sizes, fas)))
    for x, y, lab, k in zip(sizes, fas, labels, keys):
        a.scatter([x], [y], s=60, color=COLORS[k], zorder=3); a.annotate(lab, (x, y), (6, 6), textcoords="offset points", fontsize=6.4)
    fp = sorted(front, key=lambda i: sizes[i])
    a.plot([sizes[i] for i in fp], [fas[i] for i in fp], color="#546E7A", lw=1, ls="--", zorder=2)
    a.set_xlabel("Model size (KB; midpoint of the estimate until measured)", fontsize=7); a.set_ylabel("False alarms at 12σ noise (%)", fontsize=7)
    a.set_title("Size against robustness", fontsize=8, fontweight="bold"); a.tick_params(labelsize=6.5)
    measured = [r for r in rows if r["method"] == "tflite" and r["mean_latency_ms"]]
    if measured:
        lat = [num(r["mean_latency_ms"]) for r in measured]; fa2 = [num(r["false_alarm_12sigma_percent"]) for r in measured]
        for r, x, y in zip(measured, lat, fa2):
            k = r["variant"].split()[0]; b.scatter([x], [y], s=60, color=COLORS[k], zorder=3); b.annotate(r["variant"], (x, y), (6, 6), textcoords="offset points", fontsize=6.4)
        f2 = sorted(pareto_front(list(zip(lat, fa2))), key=lambda i: lat[i])
        b.plot([lat[i] for i in f2], [fa2[i] for i in f2], color="#546E7A", lw=1, ls="--")
        b.set_xlabel("Mean inference latency (ms, TFLite)", fontsize=7); b.set_ylabel("False alarms at 12σ noise (%)", fontsize=7)
    else:
        b.axis("off"); b.text(0.5, 0.5, "Latency not measured yet.\nRun on a laptop:\npython3 benchmark.py --tflite ...\nthen build_results_csv.py\nand make_pareto.py", ha="center", va="center", fontsize=7.5, color="#B71C1C")
    b.set_title("Latency against robustness", fontsize=8, fontweight="bold"); b.tick_params(labelsize=6.5)
    status = "complete" if len(measured) == 3 else "PROVISIONAL: TFLite latency and energy pending"
    fig.suptitle(f"LogiEdge model variants ({status})", fontsize=8.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out); plt.close(fig)
    return {"front": [keys[i] for i in front], "provisional": len(measured) < 3}


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=os.path.join(here, "results", "benchmark_results.csv"))
    p.add_argument("--out", default=os.path.join(here, "results", "pareto_chart.png"))
    a = p.parse_args(argv)
    info = draw(load(a.csv), a.out)
    print(f"Wrote {a.out}; Pareto-optimal on size and robustness: {', '.join(info['front'])}"
          + ("; PROVISIONAL" if info["provisional"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
