#!/usr/bin/env python3
"""Check the vibration-only gap (supports the Task F3 report).

The Task D1 training data contain Normal, temp_drift and combined runs only. A bearing-wear fault with normal
temperature (--anomaly vibration) was never shown to the model. This script (1) measures what the D1 models do
with it and (2) trains a model with vibration-only runs added (labelled Warning: "early refrigeration anomaly",
Task C scheme B) to see whether the gap is easy to close and whether anything else gets worse.

Run:  python3 vibration_gap_check.py        (writes vibration_gap_results.json; deterministic)
"""
import json

import numpy as np

import generate_dataset as gd
from model_variants import build_extended_test, load_splits, metrics
from train_model import MLP


def vibration_runs(norm, seeds, onsets=(0, 80, 160, 240, 320), duration=600):
    X, y = [], []
    for s in seeds:
        for j, onset in enumerate(onsets):
            wins = gd.messages_to_windows(gd.run_simulator("vibration", duration, 700 + s + 10 * j, anomaly_start_s=onset))
            for w in wins:
                X.append(norm.transform(w.raw))
                y.append(1 if (w.end - gd.START_EPOCH) - onset > 15 else 0)     # same majority rule as the onset runs
    return np.array(X), np.array(y)


def main() -> int:
    sp = load_splits()
    norm = sp["norm"]
    ext = build_extended_test(norm)
    Xv_train, yv_train = vibration_runs(norm, seeds=(0, 1))
    Xv_test, yv_test = vibration_runs(norm, seeds=(1000, 2000, 3000))
    out = {"vibration_train_windows": int(len(yv_train)), "vibration_test_windows": int(len(yv_test)),
           "vibration_test_anomalous": int(np.sum(yv_test == 1))}

    base = MLP.load("model_weights_onset.npz")
    X = np.vstack([sp["X_train"], Xv_train])
    y = np.concatenate([sp["y_train"], yv_train])
    fixed = MLP(seed=42)
    fixed.fit(X, y, seed=42)

    for name, mdl in (("D1 onset model", base), ("with vibration-only runs added", fixed)):
        pred = mdl.predict(Xv_test)
        anomalous = yv_test == 1
        out[name] = {
            "vibration_only_detected": float(np.mean(pred[anomalous] >= 1)),
            "vibration_only_false_alarm": float(np.mean(pred[~anomalous] != 0)),
            "val": metrics(mdl, sp["X_val"], sp["y_val"]),
            "extended_accuracy": metrics(mdl, ext["X"], ext["y"])["accuracy"],
            "extended_critical_recall": metrics(mdl, ext["X"], ext["y"])["critical_recall"]}
        print(f"{name}: vibration-only anomalous windows flagged {out[name]['vibration_only_detected']:.3f}; "
              f"false alarms {out[name]['vibration_only_false_alarm']:.3f}; val acc {out[name]['val']['accuracy']:.3f}; "
              f"extended acc {out[name]['extended_accuracy']:.3f}, critical recall {out[name]['extended_critical_recall']:.3f}")
    with open("vibration_gap_results.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
