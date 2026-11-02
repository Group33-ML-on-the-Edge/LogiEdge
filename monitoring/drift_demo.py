#!/usr/bin/env python3
"""Drift demonstration for Task E1: clean data -> inject --anomaly combined -> restore clean data.

Runs the real chain (simulator -> preprocessing -> model -> PSI monitor) in one process on a fast clock, so it needs
no broker, and prints the PSI lines exactly as the inference service does:

    [LOGIBRIDGE] t=00:21:00 PSI=0.000 status=stable (last 100 inferences, score=normal_confidence)
    [LOGIBRIDGE DRIFT ALERT] PSI=1.234

Default timeline (simulated time):
    0 - 20 min     clean data (the 100-inference window fills after about 17 minutes)
    20 - 25 min    --anomaly combined injected
    25 - 48 min    clean data restored

Use --pace 0.25 to slow the printing down for screen recording, and --score max_confidence to see the literal
"confidence" score, which does not move for this model.

To do the same live with the container and a broker, see the Task E1 report (Section 6).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import generate_dataset as gd
from drift_monitor import SCORE_NAMES, PsiMonitor, load_reference
from inference_service import load_model
from preprocessing import Normaliser


def elapsed_label(ts: float) -> str:
    return f"{int(ts // 3600):02d}:{int(ts % 3600 // 60):02d}:{int(ts % 60):02d}"


def mmss(seconds: float) -> str:
    return f"{int(seconds // 60)} min {int(seconds % 60):02d} s"


def run_demo(model_path="model_weights.npz", stats_path="training_stats.npy", reference_path="reference_dist.json",
             score="normal_confidence", inject_at=1200.0, inject_for=300.0, total=2900.0, seed=777, eps=None,
             pace=0.0, out=print) -> dict:
    ref = load_reference(reference_path)
    model = load_model(model_path)
    norm = Normaliser.load(stats_path)
    restore_at = inject_at + inject_for

    printer = out
    if pace > 0:
        def printer(line):
            out(line)
            time.sleep(pace)
    monitors = {name: PsiMonitor(ref, name, eps=eps, emit=(printer if name == score else (lambda line: None)),
                                 time_label=elapsed_label) for name in SCORE_NAMES}

    msgs = gd.run_simulator("combined", total, seed, anomaly_start_s=inject_at, anomaly_end_s=restore_at)
    out(f"Reference: {ref['n_windows']} clean Normal-class windows, model {ref['model']['file']} ({ref['model']['sha256']})")
    out(f"Timeline: clean until {elapsed_label(inject_at)}, --anomaly combined until {elapsed_label(restore_at)}, "
        f"then clean until {elapsed_label(total)}; monitored score = {score}")
    out("")

    marks = {"inject": False, "restore": False}
    for window in gd.messages_to_windows(msgs, norm):
        t = window.end - gd.START_EPOCH
        if not marks["inject"] and t >= inject_at:
            marks["inject"] = True
            printer(f"=== t={elapsed_label(inject_at)}  INJECT: --anomaly combined activated ===")
        if not marks["restore"] and t >= restore_at:
            marks["restore"] = True
            printer(f"=== t={elapsed_label(restore_at)}  RESTORE: clean data back ===")
        probs = model.predict_proba(window.z)
        for m in monitors.values():
            m.add(t, probs)

    result = {"inject_at": inject_at, "restore_at": restore_at, "total": total, "score": score, "monitors": {}}
    for name, m in monitors.items():
        ev = [e for e in m.events if e["psi"] is not None]
        cross = next((e for e in ev if e["t"] >= inject_at and e["psi"] > 0.25), None)
        peak = max((e["psi"] for e in ev), default=0.0)
        rec = None
        if cross is not None:
            rec = next((e for e in ev if e["t"] >= restore_at and e["psi"] < 0.10), None)
        result["monitors"][name] = {
            "events": [{"t": e["t"], "psi": e["psi"], "alert": e["alert"]} for e in m.events],
            "peak_psi": peak,
            "crossed_after_s": None if cross is None else cross["t"] - inject_at,
            "recovered_after_s": None if rec is None else rec["t"] - restore_at,
            "false_alarms_before_injection": sum(1 for e in ev if e["t"] < inject_at and e["psi"] > 0.25),
        }
    out("")
    out("Summary")
    for name in SCORE_NAMES:
        r = result["monitors"][name]
        crossed = "never crossed 0.25" if r["crossed_after_s"] is None else f"crossed 0.25 {mmss(r['crossed_after_s'])} after injection"
        recovered = "" if r["recovered_after_s"] is None else f"; back below 0.10 {mmss(r['recovered_after_s'])} after restore"
        out(f"  {name:18s} peak PSI {r['peak_psi']:.3f}; {crossed}{recovered}")
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="PSI drift demonstration (Task E1)")
    p.add_argument("--model", default="model_weights.npz")
    p.add_argument("--stats", default="training_stats.npy")
    p.add_argument("--reference", default="reference_dist.json")
    p.add_argument("--score", choices=SCORE_NAMES, default="normal_confidence")
    p.add_argument("--inject-at", type=float, default=1200.0, help="seconds of clean data before injection")
    p.add_argument("--inject-for", type=float, default=300.0, help="seconds the anomaly lasts")
    p.add_argument("--total", type=float, default=2900.0, help="total simulated seconds")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--pace", type=float, default=0.0, help="seconds to pause after each printed line (for filming)")
    p.add_argument("--json", help="write the full result to this file")
    args = p.parse_args(argv)
    result = run_demo(args.model, args.stats, args.reference, args.score, args.inject_at, args.inject_for,
                      args.total, args.seed, pace=args.pace)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
    r = result["monitors"][args.score]
    ok = r["crossed_after_s"] is not None and r["crossed_after_s"] <= 300 and r["recovered_after_s"] is not None
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
