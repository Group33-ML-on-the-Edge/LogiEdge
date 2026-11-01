#!/usr/bin/env python3
"""Five-metric benchmark for the three model variants (Task F2).

Metrics per variant
  1. mean inference latency (ms)  - 200 timed runs after 10 warm-up runs that are discarded
  2. p95 inference latency (ms)   - 95th percentile of the same 200 runs
  3. model file size (KB)         - size of the .tflite file (1 KB = 1,000 bytes; bytes are printed too)
  4. classification accuracy (%)  - on the held-out validation set (the 20% split from Task D1), plus Critical recall
  5. energy per inference (mJ)    - E = P x t, with P = TDP x CPU% / 100

Energy method (my reading of the Lab 2 methodology, please confirm): CPU% is the system-wide utilisation that
psutil reports while the model runs inference back to back for a few seconds; P is the laptop TDP estimate times
that fraction; t is the mean latency. A single-threaded loop on an 8-core laptop reads about 12%, so P is a
fraction of the TDP. Because W x ms = mJ, E(mJ) = P(W) x latency(ms).

Run on your laptop with the TFLite files from build_variants_tf.py:
    python3 benchmark.py --tflite M1=m1_fp32.tflite M2=m2_ptq_int8.tflite M3=m3_pruned_int8.tflite --tdp-watts 15

Harness self-check without TensorFlow (numpy callables; NOT TFLite, not comparable with it):
    python3 benchmark.py --numpy
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Callable, Optional

import numpy as np

RUNS = 200
WARMUP = 10


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------
def time_callable(fn: Callable, samples: np.ndarray, runs: int = RUNS, warmup: int = WARMUP) -> np.ndarray:
    """Latency of fn(sample) in ms for `runs` calls, after `warmup` discarded calls."""
    n = len(samples)
    for i in range(warmup):
        fn(samples[i % n])
    out = np.empty(runs)
    for i in range(runs):
        x = samples[i % n]
        t0 = time.perf_counter_ns()
        fn(x)
        out[i] = (time.perf_counter_ns() - t0) / 1e6
    return out


def cpu_percent_during(fn: Callable, samples: np.ndarray, seconds: float = 2.0) -> float:
    """System-wide CPU% (psutil) while fn runs back to back for `seconds`."""
    import psutil
    n, i = len(samples), 0
    psutil.cpu_percent(interval=None)                       # start the measurement window
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        fn(samples[i % n])
        i += 1
    return float(psutil.cpu_percent(interval=None))


def energy_mj(latency_ms: float, cpu_percent: float, tdp_watts: float) -> float:
    """E = P x t with P = TDP x CPU% / 100. Watts x milliseconds = millijoules."""
    return tdp_watts * (cpu_percent / 100.0) * latency_ms


def evaluate_callable(fn: Callable, X: np.ndarray, y: np.ndarray) -> dict:
    """Accuracy and Critical (class 2) recall; fn(x) returns class scores for one window."""
    pred = np.array([int(np.argmax(fn(x))) for x in X])
    crit = y == 2
    return {"accuracy": float(np.mean(pred == y)),
            "critical_recall": float(np.mean(pred[crit] == 2)) if crit.any() else float("nan"),
            "n": int(len(y)), "n_critical": int(crit.sum())}


def benchmark_variant(name: str, fn: Callable, X_val: np.ndarray, y_val: np.ndarray, file_bytes: Optional[int],
                      tdp_watts: float, runs: int = RUNS, warmup: int = WARMUP, cpu_seconds: float = 2.0) -> dict:
    lat = time_callable(fn, X_val, runs, warmup)
    cpu = cpu_percent_during(fn, X_val, cpu_seconds)
    mean = float(lat.mean())
    ev = evaluate_callable(fn, X_val, y_val)
    return {"variant": name, "runs": int(len(lat)), "warmup": warmup,
            "mean_latency_ms": mean, "p95_latency_ms": float(np.percentile(lat, 95)),
            "file_bytes": file_bytes, "file_kb": None if file_bytes is None else file_bytes / 1000.0,
            "accuracy_percent": 100 * ev["accuracy"], "critical_recall_percent": 100 * ev["critical_recall"],
            "cpu_percent": cpu, "tdp_watts": tdp_watts, "energy_mj": energy_mj(mean, cpu, tdp_watts)}


# --------------------------------------------------------------------------
# Callables
# --------------------------------------------------------------------------
def _make_interpreter(path: str):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    return Interpreter(model_path=path)


def tflite_callable(path: str) -> Callable:
    """fn(z) for a .tflite file; quantises the input and dequantises the output for int8 models."""
    interp = _make_interpreter(path)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

    def fn(z):
        x = np.asarray(z, dtype=np.float32).reshape(1, -1)
        if inp["dtype"] != np.float32:
            scale, zero = inp["quantization"]
            info = np.iinfo(inp["dtype"])
            x = np.clip(np.round(x / scale + zero), info.min, info.max).astype(inp["dtype"])
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = np.asarray(interp.get_tensor(out["index"]))[0]
        if out["dtype"] != np.float32:
            scale, zero = out["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        return y
    return fn


def numpy_callables() -> tuple:
    """Self-check callables built from the numpy reference variants (model_variants.py)."""
    import model_variants as mv
    sp = mv.load_splits()
    v = mv.build_variants(sp)
    fns = {"M1": lambda z: v["M1"]._forward(np.asarray(z, float)[None, :])[-1][0],
           "M2": lambda z: v["M2"].logits(np.asarray(z, float)[None, :])[0],
           "M3": lambda z: v["M3"].logits(np.asarray(z, float)[None, :])[0]}
    return fns, sp["X_val"], sp["y_val"]


CSV_COLUMNS = ["variant", "method", "mean_latency_ms", "p95_latency_ms", "file_kb", "accuracy_percent",
               "critical_recall_percent", "cpu_percent", "tdp_watts", "energy_mj"]


def write_csv(results: list, path: str, method: str = "tflite") -> None:
    """One row per variant with the five metrics (plus CPU% and the TDP used for the energy figure)."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in CSV_COLUMNS if k != "method"}
            row["variant"], row["method"] = r["variant"], method
            w.writerow(row)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tflite", nargs="+", metavar="NAME=FILE", help="variants to benchmark, e.g. M1=m1_fp32.tflite")
    p.add_argument("--numpy", action="store_true", help="harness self-check with numpy callables (not TFLite)")
    p.add_argument("--tdp-watts", type=float, default=15.0, help="laptop TDP estimate in watts (default 15)")
    p.add_argument("--runs", type=int, default=RUNS)
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--cpu-seconds", type=float, default=2.0)
    p.add_argument("--out", default="benchmark_results.json")
    p.add_argument("--csv", default="benchmark_results_raw.csv", help="also write the results as CSV")
    a = p.parse_args(argv)
    if not a.tflite and not a.numpy:
        p.error("give --tflite NAME=FILE ... or --numpy")

    results = []
    if a.numpy:
        fns, X_val, y_val = numpy_callables()
        for name, fn in fns.items():
            results.append(benchmark_variant(name + " (numpy)", fn, X_val, y_val, None, a.tdp_watts,
                                             a.runs, a.warmup, a.cpu_seconds))
    else:
        import model_variants as mv
        sp = mv.load_splits()
        for item in a.tflite:
            name, _, path = item.partition("=")
            results.append(benchmark_variant(name, tflite_callable(path), sp["X_val"], sp["y_val"],
                                             os.path.getsize(path), a.tdp_watts, a.runs, a.warmup, a.cpu_seconds))

    print(f"{'variant':14s} {'mean ms':>9s} {'p95 ms':>9s} {'file KB':>9s} {'acc %':>7s} {'crit recall %':>13s} {'CPU %':>6s} {'mJ/inf':>9s}")
    for r in results:
        kb = "-" if r["file_kb"] is None else f"{r['file_kb']:.2f}"
        print(f"{r['variant']:14s} {r['mean_latency_ms']:9.4f} {r['p95_latency_ms']:9.4f} {kb:>9s} "
              f"{r['accuracy_percent']:7.2f} {r['critical_recall_percent']:13.2f} {r['cpu_percent']:6.1f} {r['energy_mj']:9.5f}")
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    write_csv(results, a.csv, "numpy-self-check (not TFLite)" if a.numpy else "tflite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
