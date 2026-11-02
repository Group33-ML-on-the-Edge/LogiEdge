#!/usr/bin/env python3
"""PSI drift monitoring on the model's output scores (Task E1).

Population Stability Index compares the recent distribution of a model output score with a
reference distribution captured on clean Normal-class data:

    PSI = sum over bins of (current_i - reference_i) * ln(current_i / reference_i)

Bins: [0, 0.25), [0.25, 0.50), [0.50, 0.75), [0.75, 1.0]. A proportion of zero is replaced by a small
floor (epsilon, default 1e-4) so the logarithm is defined. Usual reading: below 0.10 stable, 0.10 to 0.25
moderate, above 0.25 significant drift.

Monitoring rule: PSI over the last 100 inferences, evaluated every 60 seconds (data time, taken from the
inference timestamps, so it also works when the simulator is sped up). Above 0.25 the monitor prints

    [LOGIBRIDGE DRIFT ALERT] PSI=<value:.3f>

Which score is monitored
    max_confidence      highest class probability (the literal "confidence score"). For this model it is
                        saturated (0.999 or more) on clean AND on faulty windows, so PSI stays at 0.
    normal_confidence   the probability that the cargo is Normal (class 0). It is near 1 on clean data and near
                        0 when a fault appears, so PSI responds. This is the default.

Command line
    python3 drift_monitor.py make-reference --model model_weights.npz --out reference_dist.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

BIN_EDGES = (0.0, 0.25, 0.50, 0.75, 1.0)
SCORE_NAMES = ("max_confidence", "normal_confidence")
DEFAULT_SCORE = "normal_confidence"
WINDOW_SIZE = 100            # last 100 inferences
INTERVAL_S = 60.0            # evaluated every 60 s
ALERT_THRESHOLD = 0.25
STABLE_THRESHOLD = 0.10
EPSILON = 1e-4
REFERENCE_WINDOWS = 300      # clean Normal-class windows in the reference


# --------------------------------------------------------------------------
# PSI maths
# --------------------------------------------------------------------------
def scores_from_probs(probs) -> dict:
    """The monitored scores for one inference (probs = class probabilities, Normal first)."""
    probs = np.asarray(probs, dtype=float).ravel()
    return {"max_confidence": float(np.max(probs)), "normal_confidence": float(probs[0])}


def bin_counts(values) -> list:
    """Counts per bin; the last bin includes 1.0."""
    return np.histogram(np.asarray(values, dtype=float), bins=BIN_EDGES)[0].astype(int).tolist()


def proportions(counts) -> list:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        raise ValueError("cannot compute proportions of an empty sample")
    return (counts / total).tolist()


def psi(reference_props, current_props, eps: float = EPSILON) -> float:
    """Population Stability Index with a floor of eps on every proportion."""
    ref = np.maximum(np.asarray(reference_props, dtype=float), eps)
    cur = np.maximum(np.asarray(current_props, dtype=float), eps)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def status_for(value: float) -> str:
    return "drift" if value > ALERT_THRESHOLD else ("moderate" if value >= STABLE_THRESHOLD else "stable")


# --------------------------------------------------------------------------
# Reference distribution file
# --------------------------------------------------------------------------
def build_reference(prob_rows, model_file: str = "", model_sha256: str = "", source: str = "") -> dict:
    """Reference distribution from the class probabilities of clean Normal-class windows."""
    per_score = {name: [] for name in SCORE_NAMES}
    for p in prob_rows:
        s = scores_from_probs(p)
        for name in SCORE_NAMES:
            per_score[name].append(s[name])
    n = len(per_score[SCORE_NAMES[0]])
    dist = {}
    for name, vals in per_score.items():
        counts = bin_counts(vals)
        dist[name] = {"counts": counts, "proportions": proportions(counts),
                      "mean": float(np.mean(vals)), "min": float(np.min(vals))}
    return {"format": 1, "bin_edges": list(BIN_EDGES), "n_windows": n, "epsilon": EPSILON, "source": source,
            "model": {"file": model_file, "sha256": model_sha256}, "scores": dist}


def save_reference(ref: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(ref, f, indent=2)


def load_reference(path: str) -> dict:
    with open(path) as f:
        ref = json.load(f)
    if list(ref.get("bin_edges", [])) != list(BIN_EDGES):
        raise ValueError(f"{path}: unexpected bin edges {ref.get('bin_edges')}")
    for name in SCORE_NAMES:
        props = ref["scores"][name]["proportions"]
        if len(props) != len(BIN_EDGES) - 1 or abs(sum(props) - 1.0) > 1e-6:
            raise ValueError(f"{path}: bad proportions for {name}")
    return ref


# --------------------------------------------------------------------------
# The monitor
# --------------------------------------------------------------------------
def utc_label(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")


class PsiMonitor:
    """Rolling PSI over the last `window` inferences, evaluated every `interval_s` of data time.

    Feed every inference with add(timestamp, class_probabilities). It returns the evaluations that fell due
    (usually none or one) and prints a line for each through `emit`.
    """

    def __init__(self, reference: dict, score: str = DEFAULT_SCORE, window: int = WINDOW_SIZE,
                 interval_s: float = INTERVAL_S, alert_threshold: float = ALERT_THRESHOLD,
                 eps: Optional[float] = None, emit: Optional[Callable[[str], None]] = None,
                 time_label: Callable[[float], str] = utc_label):
        if score not in SCORE_NAMES:
            raise ValueError(f"score must be one of {SCORE_NAMES}")
        self.score, self.window, self.interval_s = score, window, interval_s
        self.alert_threshold = alert_threshold
        self.eps = reference.get("epsilon", EPSILON) if eps is None else eps
        self.ref_props = reference["scores"][score]["proportions"]
        self.emit = emit or (lambda line: print(line, flush=True))
        self.time_label = time_label
        self._buf: deque = deque(maxlen=window)
        self._next_eval: Optional[float] = None
        self.events: list = []

    def add(self, ts: float, probs) -> list:
        self._buf.append(scores_from_probs(probs)[self.score])
        if self._next_eval is None:
            self._next_eval = ts + self.interval_s
        due = []
        while ts >= self._next_eval:
            due.append(self._evaluate(self._next_eval))
            self._next_eval += self.interval_s
        return due

    def current_psi(self) -> Optional[float]:
        if len(self._buf) < self.window:
            return None
        return psi(self.ref_props, proportions(bin_counts(list(self._buf))), self.eps)

    def _evaluate(self, t: float) -> dict:
        n = len(self._buf)
        value = self.current_psi()
        label = self.time_label(t)
        if value is None:
            event = {"t": t, "psi": None, "n": n, "status": "warming_up", "alert": False, "score": self.score}
            self.emit(f"[LOGIBRIDGE] t={label} PSI n/a (collecting {n}/{self.window} inferences)")
        else:
            alert = value > self.alert_threshold
            event = {"t": t, "psi": value, "n": n, "status": status_for(value), "alert": alert, "score": self.score}
            self.emit(f"[LOGIBRIDGE] t={label} PSI={value:.3f} status={event['status']} "
                      f"(last {n} inferences, score={self.score})")
            if alert:
                self.emit(f"[LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}")
        self.events.append(event)
        return event


# --------------------------------------------------------------------------
# Offline: create reference_dist.json
# --------------------------------------------------------------------------
def make_reference(model_path: str, stats_path: str = "training_stats.npy", out_path: str = "reference_dist.json",
                   n_windows: int = REFERENCE_WINDOWS, seed: int = 5001) -> dict:
    """Run the model on `n_windows` clean Normal-class windows (simulator --anomaly none) and save the distribution."""
    import generate_dataset as gd                                  # offline tool: needs the simulator
    from inference_service import load_model
    from preprocessing import Normaliser

    model = load_model(model_path)
    norm = Normaliser.load(stats_path)
    seconds = (n_windows - 1) * 10 + 30 + 5                         # enough data for n_windows windows
    windows = gd.messages_to_windows(gd.run_simulator("none", seconds, seed), norm)[:n_windows]
    if len(windows) < n_windows:
        raise RuntimeError(f"only {len(windows)} windows produced")
    probs = [model.predict_proba(w.z) for w in windows]
    ref = build_reference(probs, model.file, model.sha256,
                          f"{n_windows} clean Normal-class windows, simulator --anomaly none, seed {seed}")
    save_reference(ref, out_path)
    return ref


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="PSI drift monitoring tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-reference", help="create reference_dist.json from 300 clean Normal-class windows")
    m.add_argument("--model", default="model_weights.npz")
    m.add_argument("--stats", default="training_stats.npy")
    m.add_argument("--out", default="reference_dist.json")
    m.add_argument("--windows", type=int, default=REFERENCE_WINDOWS)
    m.add_argument("--seed", type=int, default=5001)
    args = p.parse_args(argv)
    ref = make_reference(args.model, args.stats, args.out, args.windows, args.seed)
    print(f"Saved {args.out}: {ref['n_windows']} windows, model {ref['model']['file']} ({ref['model']['sha256']})")
    for name in SCORE_NAMES:
        d = ref["scores"][name]
        print(f"  {name:18s} counts {d['counts']}  proportions {[round(x, 3) for x in d['proportions']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
