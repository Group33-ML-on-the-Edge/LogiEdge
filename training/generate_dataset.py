#!/usr/bin/env python3
"""Generate the labelled training dataset for Task D1.

Runs the Task C1 simulator in each mode for the specified duration, passes the published
messages through the Task C2 preprocessing pipeline and saves one 6-value feature vector per
30 s window (step 10 s):

    class  label     simulator mode        duration   approx. windows
    0      Normal    --anomaly none        20 min     ~120
    1      Warning   --anomaly temp_drift  15 min     ~90
    2      Critical  --anomaly combined    15 min     ~90

The simulator is run in-process by calling the same `simulator.run()` loop that the
command line uses, on an accelerated clock, so the messages are identical to a live run but
no broker is needed. To record from a live broker instead, run (three terminals):

    mosquitto_sub -h localhost -t 'truck/T001/features' > features_normal.jsonl
    python3 preprocessing.py run --stats training_stats.npy
    python3 simulator.py --anomaly none --duration 1200 --speed 10

Outputs: training_dataset.npz and training_dataset.csv

Labels:
    y        class assigned by the simulator mode (the Task D1 definition; used for training)
    y_state  class implied by the noise-free simulator state at the end of each window
             (Task C2 "scheme B" thresholds), saved for the label-consistency check
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime

import numpy as np

import simulator as sim
from preprocessing import FEATURE_NAMES, Normaliser, PreprocessingPipeline

START_EPOCH = 1_800_000_000.0          # fixed start time so the dataset is reproducible
CLASS_NAMES = ("Normal", "Warning", "Critical")
SPECS = [  # class id, simulator mode, duration in seconds, default seed
    (0, "none", 20 * 60, 101),
    (1, "temp_drift", 15 * 60, 102),
    (2, "combined", 15 * 60, 103),
]


class Recorder:
    """Publisher that keeps messages instead of sending them."""

    def __init__(self):
        self.messages = []

    def connect(self): pass
    def close(self): pass

    def publish(self, topic, payload, qos):
        self.messages.append((topic, json.loads(payload)))
        return True


def run_simulator(mode: str, duration_s: float, seed: int, temp_cap_c=None, truck_id: str = "T001",
                  anomaly_start_s: float = 0.0, anomaly_end_s=None):
    """Run the C1 simulator loop on a fake clock and return the published messages."""
    now = {"t": 0.0}
    cfg = sim.SimConfig(anomaly=mode, truck_id=truck_id, duration_s=duration_s, seed=seed,
                        temp_cap_c=temp_cap_c, anomaly_start_s=anomaly_start_s,
                        anomaly_end_s=anomaly_end_s)
    rec = Recorder()
    sim.run(cfg, rec, clock=lambda: now["t"], wait=lambda s: now.__setitem__("t", now["t"] + s),
            start_epoch=START_EPOCH)
    return rec.messages


def messages_to_windows(messages, normaliser=None):
    """Feed simulator messages (temperature and vibration) through the preprocessing pipeline."""
    pipe = PreprocessingPipeline(normaliser)
    windows = []
    for topic, body in messages:
        kind = topic.rsplit("/", 1)[1]
        if kind in ("temp", "vib"):
            ts = datetime.fromisoformat(body["ts"].replace("Z", "+00:00")).timestamp()
            windows += pipe.push(kind, ts, body["value"])
    return windows


def state_label(mode: str, window_end_s: float) -> int:
    """Class implied by the noise-free simulator state at the end of a window (C2 scheme B)."""
    drift = mode in ("temp_drift", "combined")
    fault = mode == "combined"
    d = sim.TEMP_DRIFT_PER_READING_C * window_end_s if drift else 0.0   # last reading k = end
    t_cls = 0 if d <= 1.0 else (1 if d <= 3.0 else 2)
    if fault and d > 1.0:
        return 2
    return max(t_cls, 1 if fault else 0)


def build_dataset(specs=SPECS, seed_offset: int = 0, temp_cap_c=None) -> dict:
    raw, y, y_state, run_id, end = [], [], [], [], []
    for cls, mode, duration, seed in specs:
        msgs = run_simulator(mode, duration, seed + seed_offset, temp_cap_c)
        for w in messages_to_windows(msgs):
            raw.append(w.raw)
            y.append(cls)
            end_s = w.end - START_EPOCH
            y_state.append(state_label(mode, end_s))
            run_id.append(cls)
            end.append(end_s)
    return {"raw": np.array(raw), "y": np.array(y), "y_state": np.array(y_state),
            "run": np.array(run_id), "window_end_s": np.array(end)}


def build_onset_dataset(onsets=(100, 200, 300), duration: int = 600, seed_offset: int = 0) -> dict:
    """Extra runs in which the fault starts part-way through (--anomaly-start), so the data
    include the transition windows that the Task D1 runs (fault from t = 0) never contain.
    A window is labelled with the anomaly class when more than half of it lies after onset."""
    raw, y, end = [], [], []
    for cls, mode in ((1, "temp_drift"), (2, "combined")):
        for j, onset in enumerate(onsets):
            msgs = run_simulator(mode, duration, 500 + 10 * cls + j + seed_offset, anomaly_start_s=onset)
            for w in messages_to_windows(msgs):
                e = w.end - START_EPOCH
                raw.append(w.raw); end.append(e)
                y.append(cls if (e - onset) > 15 else 0)
    return {"raw": np.array(raw), "y": np.array(y), "window_end_s": np.array(end)}


def save_dataset(ds: dict, stem: str, normaliser: Normaliser) -> None:
    np.savez(stem + ".npz", raw=ds["raw"], z=normaliser.transform(ds["raw"]), y=ds["y"],
             y_state=ds["y_state"], run=ds["run"], window_end_s=ds["window_end_s"],
             feature_names=np.array(FEATURE_NAMES), class_names=np.array(CLASS_NAMES))
    with open(stem + ".csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["window_end_s", "mode_class", "label", "state_class", *FEATURE_NAMES])
        for e, c, s, r in zip(ds["window_end_s"], ds["y"], ds["y_state"], ds["raw"]):
            wr.writerow([f"{e:.0f}", c, CLASS_NAMES[c], s, *[f"{v:.6g}" for v in r]])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate the Task D1 labelled dataset")
    p.add_argument("--stats", default="training_stats.npy", help="saved C2 statistics (used for the z columns)")
    p.add_argument("--out", default="training_dataset", help="output file stem")
    p.add_argument("--temp-cap", type=float, help="optional temperature ceiling (default: none)")
    args = p.parse_args(argv)

    normaliser = Normaliser.load(args.stats)
    ds = build_dataset(temp_cap_c=args.temp_cap)
    save_dataset(ds, args.out, normaliser)
    print(f"Saved {args.out}.npz and {args.out}.csv")
    for cls, mode, duration, seed in SPECS:
        n = int(np.sum(ds["y"] == cls))
        print(f"  class {cls} {CLASS_NAMES[cls]:8s} --anomaly {mode:10s} {duration // 60:2d} min  seed {seed}  -> {n} windows")
    print(f"  total {len(ds['y'])} windows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
