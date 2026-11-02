#!/usr/bin/env python3
"""Why max-softmax confidence cannot show this drift (supports the Task E1 report).

Prints the confidence of the saved model on clean and faulty windows, raw and after temperature scaling.
The temperature is the one that minimises the loss on held-out windows (independent runs, including faults that
start mid-run). Deterministic: fixed seeds.

Run:  python3 confidence_check.py [--model model_weights.npz]
"""
import argparse
import json

import numpy as np

import generate_dataset as gd
from preprocessing import Normaliser
from train_model import MLP


def softmax(logits, temperature):
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model_weights.npz")
    p.add_argument("--stats", default="training_stats.npy")
    p.add_argument("--json", help="write the numbers to this file")
    args = p.parse_args(argv)
    norm, model = Normaliser.load(args.stats), MLP.load(args.model)
    logits = lambda z: model._forward(z)[-1]

    std = gd.build_dataset(seed_offset=7000)
    on = gd.build_onset_dataset(onsets=(80, 160, 240, 320), seed_offset=7000)
    Z = np.vstack([norm.transform(std["raw"]), norm.transform(on["raw"])])
    y = np.concatenate([std["y"], on["y"]])
    L = logits(Z)
    best = min(((T, -np.mean(np.log(softmax(L, T)[np.arange(len(y)), y] + 1e-12))) for T in np.logspace(0, 3.5, 80)),
               key=lambda t: t[1])
    T = float(best[0])
    out = {"calibration_windows": int(len(y)), "calibration_errors": int(np.sum(L.argmax(1) != y)), "temperature": T, "rows": []}
    print(f"Calibration set: {len(y)} held-out windows, {out['calibration_errors']} errors; NLL-optimal temperature T = {T:.1f}")
    print(f"{'windows':34s} {'mean':>7s} {'lowest':>7s}   bin shares [0,.25) [.25,.5) [.5,.75) [.75,1]")
    for label, mode, secs, seed, temp in (("Clean (--anomaly none)", "none", 3600, 9001, 1.0),
                                          ("Faulty (--anomaly combined)", "combined", 900, 9002, 1.0),
                                          (f"Clean, temperature-scaled (T={T:.1f})", "none", 3600, 9001, T),
                                          (f"Faulty, temperature-scaled (T={T:.1f})", "combined", 900, 9002, T)):
        wins = gd.messages_to_windows(gd.run_simulator(mode, secs, seed), norm)
        conf = softmax(logits(np.array([w.z for w in wins])), temp).max(axis=1)
        shares = np.histogram(conf, bins=[0, .25, .5, .75, 1.0])[0] / len(conf)
        print(f"{label:34s} {conf.mean():7.4f} {conf.min():7.4f}   {np.round(shares, 3).tolist()}")
        out["rows"].append({"label": label, "n": int(len(conf)), "mean": float(conf.mean()), "min": float(conf.min()),
                            "shares": shares.tolist()})
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
