#!/usr/bin/env python3
"""Three model variants for Task F1, implemented in numpy so they run without TensorFlow.

    M1  FP32 baseline         the trained 6-32-16-3 MLP (model_weights_onset.npz from Task D1)
    M2  PTQ INT8              post-training quantisation, full integer, calibrated on >= 200 windows
    M3  Pruned + PTQ INT8     35% structured pruning (polynomial-decay schedule) of the hidden neurons,
                              fine-tuned, physically removed, then full-integer PTQ

The INT8 arithmetic follows the TFLite full-integer scheme: int8 activations with a per-tensor scale and
zero point taken from the min/max seen on the calibration data, symmetric int8 weights (per output channel),
int32 biases at scale (input scale x weight scale), int32 accumulation, and requantisation to int8 at every
layer. TFLite's fixed-point rounding differs from this float multiplier in the last bit only.

Structured pruning here removes whole hidden neurons (the "filters" of a dense layer): the importance of a
neuron is the mean absolute value of its incoming weights (tfmot's block pooling with AVG), the schedule is
tfmot's PolynomialDecay, and pruned neurons are physically removed, so the network gets smaller and faster.

The TensorFlow version of the same steps is build_variants_tf.py. Run this file to print the results:
    python3 model_variants.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

from preprocessing import Normaliser
from train_model import MLP, evaluate, stratified_split

CLASS_NAMES = ("Normal", "Warning", "Critical")
MIN_CALIBRATION_SAMPLES = 200


# --------------------------------------------------------------------------
# INT8 post-training quantisation
# --------------------------------------------------------------------------
def asym_qparams(lo: float, hi: float) -> tuple[float, int]:
    """Scale and zero point mapping [lo, hi] (extended to include 0) onto int8 [-128, 127]."""
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    scale = (hi - lo) / 255.0 or 1.0
    zero = int(round(-128 - lo / scale))
    return scale, max(-128, min(127, zero))


def quantize(x: np.ndarray, scale: float, zero: int) -> np.ndarray:
    return np.clip(np.round(np.asarray(x) / scale) + zero, -128, 127).astype(np.int32)


@dataclass
class QuantisedMLP:
    """Full-integer MLP: int8 weights, int32 biases, int8 activations."""
    in_q: tuple                    # (scale, zero point) of the input tensor
    layers: list                   # per layer: qw (int8), sw (per-channel scale), qb (int32), out_q (scale, zero)
    per_channel: bool = True
    n_calibration: int = 0

    @classmethod
    def from_float(cls, model: MLP, calibration: np.ndarray, per_channel: bool = True) -> "QuantisedMLP":
        calibration = np.asarray(calibration, dtype=float)
        if len(calibration) < MIN_CALIBRATION_SAMPLES:
            raise ValueError(f"need at least {MIN_CALIBRATION_SAMPLES} calibration samples, got {len(calibration)}")
        acts, a = [calibration], calibration
        for i, (w, b) in enumerate(zip(model.W, model.b)):
            a = a @ w + b
            if i < len(model.W) - 1:
                a = np.maximum(a, 0.0)
            acts.append(a)
        in_q = asym_qparams(acts[0].min(), acts[0].max())
        layers, s_in = [], in_q[0]
        for i, (w, b) in enumerate(zip(model.W, model.b)):
            if per_channel:
                sw = np.maximum(np.abs(w).max(axis=0), 1e-12) / 127.0
            else:
                sw = np.full(w.shape[1], max(np.abs(w).max(), 1e-12) / 127.0)
            qw = np.clip(np.round(w / sw), -127, 127).astype(np.int8)
            qb = np.round(b / (s_in * sw)).astype(np.int32)
            out_q = asym_qparams(acts[i + 1].min(), acts[i + 1].max())
            layers.append({"qw": qw, "sw": sw, "qb": qb, "out_q": out_q})
            s_in = out_q[0]
        return cls(in_q, layers, per_channel, len(calibration))

    def logits(self, X: np.ndarray) -> np.ndarray:
        s_prev, z_prev = self.in_q
        a = quantize(X, s_prev, z_prev)
        for i, L in enumerate(self.layers):
            acc = (a - z_prev).astype(np.int64) @ L["qw"].astype(np.int64) + L["qb"]       # int32 accumulation
            real = acc * (s_prev * L["sw"])                                                  # requantisation
            if i < len(self.layers) - 1:
                real = np.maximum(real, 0.0)                                                 # fused ReLU
            s_out, z_out = L["out_q"]
            a = np.clip(np.round(real / s_out) + z_out, -128, 127).astype(np.int32)
            s_prev, z_prev = s_out, z_out
        return (a - z_prev) * s_prev

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.logits(np.atleast_2d(X)), axis=1)

    def predict_proba(self, X) -> np.ndarray:
        z = self.logits(np.atleast_2d(X))
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    @property
    def n_params(self) -> int:
        return sum(L["qw"].size + L["qb"].size for L in self.layers)

    @property
    def sizes(self) -> list:
        return [L["qw"].shape[0] for L in self.layers] + [self.layers[-1]["qw"].shape[1]]


# --------------------------------------------------------------------------
# Structured pruning with a polynomial-decay schedule
# --------------------------------------------------------------------------
def polynomial_decay(step: int, begin: int, end: int, initial: float, final: float, power: float = 3.0) -> float:
    """tfmot PolynomialDecay: sparsity as a function of the training step."""
    if step < begin:
        return initial
    if step >= end:
        return final
    frac = (step - begin) / (end - begin)
    return final + (initial - final) * (1.0 - frac) ** power


def neurons_kept(units: int, sparsity: float) -> int:
    return int(round(units * (1.0 - sparsity)))


def neuron_scores(W: np.ndarray) -> np.ndarray:
    """Importance of each output neuron: mean |incoming weight| (block pooling AVG)."""
    return np.abs(W).mean(axis=0)


def prune_structured(model: MLP, X: np.ndarray, y: np.ndarray, final_sparsity: float = 0.35, epochs: int = 200,
                     lr: float = 1e-3, batch: int = 32, seed: int = 0, prune_end_fraction: float = 0.6,
                     frequency: int = 10) -> tuple[MLP, dict]:
    """Prune `final_sparsity` of the neurons of every hidden layer during fine-tuning, then remove them.

    The output layer is not pruned. Returns the smaller MLP and a log of the schedule.
    """
    rng = np.random.default_rng(seed)
    W = [w.copy() for w in model.W]
    b = [x.copy() for x in model.b]
    n_hidden = len(W) - 1
    masks = [np.ones(W[i].shape[1], dtype=bool) for i in range(n_hidden)]
    Y = np.eye(model.layers[-1])[y]
    steps_per_epoch = int(np.ceil(len(X) / batch))
    total_steps = epochs * steps_per_epoch
    end_step = int(prune_end_fraction * total_steps)
    params = W + b
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    step, log = 0, []

    def apply_masks():
        for i in range(n_hidden):
            W[i][:, ~masks[i]] = 0.0
            b[i][~masks[i]] = 0.0
            W[i + 1][~masks[i], :] = 0.0              # a removed neuron feeds nothing forward

    def forward(xb):
        acts = [xb]
        for i in range(len(W)):
            z = acts[-1] @ W[i] + b[i]
            acts.append(z if i == len(W) - 1 else np.maximum(z, 0.0))
        return acts

    for _ in range(epochs):
        order = rng.permutation(len(X))
        for k in range(0, len(X), batch):
            if step % frequency == 0:
                s = polynomial_decay(step, 0, end_step, 0.0, final_sparsity)
                for i in range(n_hidden):
                    keep = neurons_kept(len(masks[i]), s)
                    alive = np.where(masks[i])[0]
                    if len(alive) > keep:                                   # prune the weakest alive neurons
                        drop = alive[np.argsort(neuron_scores(W[i])[alive])[: len(alive) - keep]]
                        masks[i][drop] = False
                apply_masks()
                if step % (frequency * 10) == 0 or step >= end_step:
                    log.append({"step": step, "sparsity_target": round(s, 4),
                                "kept": [int(mk.sum()) for mk in masks]})
            idx = order[k:k + batch]
            acts = forward(X[idx])
            logits = acts[-1]
            p = np.exp(logits - logits.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            delta = (p - Y[idx]) / len(idx)
            gW, gb = [None] * len(W), [None] * len(W)
            for i in range(len(W) - 1, -1, -1):
                gW[i], gb[i] = acts[i].T @ delta, delta.sum(axis=0)
                if i > 0:
                    delta = (delta @ W[i].T) * (acts[i] > 0)
            step += 1
            for j, (p_, g_) in enumerate(zip(params, gW + gb)):
                m[j] = 0.9 * m[j] + 0.1 * g_
                v[j] = 0.999 * v[j] + 0.001 * g_ * g_
                p_ -= lr * (m[j] / (1 - 0.9 ** step)) / (np.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)
            apply_masks()                                                     # masked weights stay zero

    small = remove_neurons(W, b, masks)
    return small, {"schedule": log, "kept": [int(mk.sum()) for mk in masks], "removed": [int((~mk).sum()) for mk in masks],
                   "total_steps": total_steps, "end_step": end_step}


def remove_neurons(W: list, b: list, masks: list) -> MLP:
    """Physically delete masked hidden neurons: their columns, biases and the next layer's matching rows."""
    keep = [np.where(mk)[0] for mk in masks]
    layers = [W[0].shape[0]] + [len(k) for k in keep] + [W[-1].shape[1]]
    small = MLP(layers)
    for i in range(len(W)):
        rows = slice(None) if i == 0 else keep[i - 1]
        cols = keep[i] if i < len(keep) else slice(None)
        small.W[i] = W[i][rows][:, cols].copy()
        small.b[i] = b[i][cols].copy()
    return small


def shrink_dead_neurons(W: list, b: list) -> tuple:
    """Remove hidden neurons whose incoming weights are all zero (as left by tfmot structured pruning).

    A removed neuron still outputs relu(bias), a constant; that constant is folded into the next layer's bias so the
    smaller network computes exactly the same function. Returns new weight and bias lists.
    """
    W = [np.array(w, dtype=float) for w in W]
    b = [np.array(x, dtype=float) for x in b]
    for i in range(len(W) - 1):
        dead = np.all(W[i] == 0, axis=0)
        if dead.any():
            b[i + 1] = b[i + 1] + np.maximum(b[i][dead], 0.0) @ W[i + 1][dead, :]
        keep = ~dead
        W[i], b[i], W[i + 1] = W[i][:, keep], b[i][keep], W[i + 1][keep, :]
    return W, b


# --------------------------------------------------------------------------
# Data and the three variants
# --------------------------------------------------------------------------
def load_splits(dataset="training_dataset.npz", stats="training_stats.npy", seed: int = 42) -> dict:
    """Training windows (with the mid-run-onset runs the D1 recommended model used), the required 20% validation
    split, and the calibration set (the training windows)."""
    from generate_dataset import build_onset_dataset
    norm = Normaliser.load(stats)
    d = np.load(dataset)
    X, y = norm.transform(d["raw"]), d["y"]
    tr, va = stratified_split(y, 0.2, seed)
    extra = build_onset_dataset()
    X_train = np.vstack([X[tr], norm.transform(extra["raw"])])
    y_train = np.concatenate([y[tr], extra["y"]])
    return {"norm": norm, "X_val": X[va], "y_val": y[va], "X_train": X_train, "y_train": y_train,
            "X_calib": X[tr], "n_original_train": len(tr)}


def build_extended_test(norm: Normaliser) -> dict:
    """Independent held-out runs: 5 standard sets (fault from t = 0) and 5 sets with faults starting mid-run."""
    from generate_dataset import build_dataset, build_onset_dataset
    Xs, ys, kinds = [], [], []
    for off in (1000, 2000, 3000, 4000, 5000):
        s = build_dataset(seed_offset=off)
        Xs.append(norm.transform(s["raw"])); ys.append(s["y"]); kinds += ["standard"] * len(s["y"])
        o = build_onset_dataset(onsets=(80, 160, 240, 320), seed_offset=off)
        Xs.append(norm.transform(o["raw"])); ys.append(o["y"]); kinds += ["onset"] * len(o["y"])
    return {"X": np.vstack(Xs), "y": np.concatenate(ys), "kind": np.array(kinds)}


def build_variants(splits: dict, weights_path: str = "model_weights_onset.npz", seed: int = 0) -> dict:
    m1 = MLP.load(weights_path)
    m2 = QuantisedMLP.from_float(m1, splits["X_calib"])
    pruned, info = prune_structured(m1, splits["X_train"], splits["y_train"], 0.35, seed=seed)
    # the pruned network is fine-tuned inside prune_structured; quantise the final network
    m3 = QuantisedMLP.from_float(pruned, splits["X_calib"])
    return {"M1": m1, "M2": m2, "M3": m3, "M3_float": pruned, "prune_info": info}


def metrics(model, X, y) -> dict:
    pred = model.predict(X)
    r = evaluate(y, pred)
    r["critical_recall"] = r["recall"][2]
    r["critical_precision"] = r["precision"][2]
    return r


# --------------------------------------------------------------------------
# Size, operation and memory accounting
# --------------------------------------------------------------------------
def macs(sizes) -> int:
    """Multiply-accumulates for one inference of a dense network with these layer sizes."""
    return int(sum(a * b for a, b in zip(sizes[:-1], sizes[1:])))


def n_params(sizes) -> int:
    return int(sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:])))


def payload_bytes(kind: str, sizes, per_channel: bool = True) -> dict:
    """Bytes of numbers a model must store. kind is 'fp32' or 'int8' (int8 weights, int32 biases)."""
    weights = sum(a * b for a, b in zip(sizes[:-1], sizes[1:]))
    biases = sum(sizes[1:])
    if kind == "fp32":
        return {"weights": 4 * weights, "biases": 4 * biases, "quant_params": 0, "total": 4 * (weights + biases)}
    quant = (4 + 8) * sum(sizes[1:]) if per_channel else (4 + 8) * (len(sizes) - 1)   # float scale + int64 zero point
    return {"weights": weights, "biases": 4 * biases, "quant_params": quant, "total": weights + 4 * biases + quant}


def estimated_file_kb(kind: str, sizes, per_channel: bool = True) -> tuple:
    """Rough range for the .tflite file size: payload plus 1.2-2.2 KB (fp32) or 1.6-2.8 KB (int8) of FlatBuffer
    structure. These overheads are ESTIMATES; measure the real file with ls -l."""
    payload = payload_bytes(kind, sizes, per_channel)["total"]
    lo, hi = (1200, 2200) if kind == "fp32" else (1600, 2800)
    return (payload + lo) / 1000.0, (payload + hi) / 1000.0


def arena_bytes(kind: str, sizes) -> int:
    """Peak activation memory: the largest (layer input + layer output) pair, at 4 bytes (fp32) or 1 byte (int8)."""
    width = 4 if kind == "fp32" else 1
    return int(max(a + b for a, b in zip(sizes[:-1], sizes[1:])) * width)


# --------------------------------------------------------------------------
# Extra comparisons
# --------------------------------------------------------------------------
def noise_robustness(models: dict, X, y, sigmas=(0, 2, 5, 8, 12, 20), reps: int = 5, seed: int = 0) -> list:
    """Accuracy, false-alarm rate (Normal windows called anything else) and Critical recall when Gaussian noise
    of the given size (in normalised units) is added to every input feature."""
    rng = np.random.default_rng(seed)
    rows = []
    for sig in sigmas:
        noisy = [X + rng.normal(0, sig, X.shape) for _ in range(reps)]
        row = {"sigma": sig}
        for name, mdl in models.items():
            preds = [mdl.predict(Xn) for Xn in noisy]
            row[name] = {"accuracy": float(np.mean([np.mean(p == y) for p in preds])),
                         "false_alarm": float(np.mean([np.mean(p[y == 0] != 0) for p in preds])),
                         "critical_recall": float(np.mean([np.mean(p[y == 2] == 2) for p in preds]))}
        rows.append(row)
    return rows


def detection_delay(models: dict, norm: Normaliser, modes=("temp_drift", "combined"),
                    onsets=(80, 160, 240, 320), offsets=(1000, 2000, 3000), duration: int = 600) -> dict:
    """Seconds from fault onset to the first window flagged Warning or Critical, per variant and mode."""
    from generate_dataset import messages_to_windows, run_simulator, START_EPOCH
    out = {name: {m: [] for m in modes} for name in models}
    missed = {name: 0 for name in models}
    for mode in modes:
        for off in offsets:
            for j, onset in enumerate(onsets):
                msgs = run_simulator(mode, duration, 900 + off + 10 * j, anomaly_start_s=onset)
                wins = messages_to_windows(msgs)
                Z = norm.transform(np.array([w.raw for w in wins]))
                ends = np.array([w.end - START_EPOCH for w in wins])
                for name, mdl in models.items():
                    pred = mdl.predict(Z)
                    hit = np.where((ends > onset) & (pred >= 1))[0]
                    if len(hit):
                        out[name][mode].append(float(ends[hit[0]] - onset))
                    else:
                        missed[name] += 1
    return {name: {mode: {"mean": float(np.mean(v)) if v else None, "max": float(np.max(v)) if v else None, "n": len(v)}
                   for mode, v in d.items()} | {"missed": missed[name]} for name, d in out.items()}


def vibration_only_check(models: dict, norm: Normaliser, seconds: int = 900, seed: int = 321) -> dict:
    """Class counts predicted for a bearing-wear fault with normal temperature (--anomaly vibration).
    This case is NOT in the Task D1 training data (Normal, temp_drift, combined only)."""
    from generate_dataset import messages_to_windows, run_simulator
    wins = messages_to_windows(run_simulator("vibration", seconds, seed))
    Z = norm.transform(np.array([w.raw for w in wins]))
    return {name: np.bincount(m.predict(Z), minlength=3).tolist() for name, m in models.items()} | {"windows": len(Z)}


def agreement_with_baseline(models: dict, X) -> dict:
    base = models["M1"].predict(X)
    return {n: float(np.mean(m.predict(X) == base)) for n, m in models.items()}


def main() -> int:
    sp = load_splits()
    v = build_variants(sp)
    ext = build_extended_test(sp["norm"])
    models = {"M1": v["M1"], "M2": v["M2"], "M3": v["M3"]}
    sizes = {"M1": list(v["M1"].layers), "M2": list(v["M2"].sizes), "M3": list(v["M3"].sizes)}
    kinds = {"M1": "fp32", "M2": "int8", "M3": "int8"}
    out = {"n_calibration": int(sp["X_calib"].shape[0]), "sizes": sizes, "prune": {
               "kept": v["prune_info"]["kept"], "removed": v["prune_info"]["removed"],
               "total_steps": v["prune_info"]["total_steps"], "end_step": v["prune_info"]["end_step"]},
           "input_quantisation": {"scale": v["M2"].in_q[0], "zero_point": v["M2"].in_q[1]},
           "val_counts": np.bincount(sp["y_val"], minlength=3).tolist(),
           "ext_counts": np.bincount(ext["y"], minlength=3).tolist(), "models": {}}
    for name, mdl in models.items():
        a, e = metrics(mdl, sp["X_val"], sp["y_val"]), metrics(mdl, ext["X"], ext["y"])
        std = ext["kind"] == "standard"
        out["models"][name] = {
            "val": a, "extended": e,
            "extended_standard": metrics(mdl, ext["X"][std], ext["y"][std]),
            "extended_onset": metrics(mdl, ext["X"][~std], ext["y"][~std]),
            "params": n_params(sizes[name]), "macs": macs(sizes[name]), "kind": kinds[name],
            "payload": payload_bytes(kinds[name], sizes[name]),
            "file_kb_estimate": estimated_file_kb(kinds[name], sizes[name]),
            "arena_bytes": arena_bytes(kinds[name], sizes[name])}
        print(f"{name} {sizes[name]}: val acc {a['accuracy']:.4f} crit recall {a['critical_recall']:.4f} | "
              f"extended acc {e['accuracy']:.4f} crit recall {e['critical_recall']:.4f} (n={e['n']})")
    out["agreement"] = agreement_with_baseline(models, ext["X"])
    out["noise"] = noise_robustness(models, ext["X"], ext["y"])
    out["detection_delay"] = detection_delay(models, sp["norm"])
    out["vibration_only"] = vibration_only_check(models, sp["norm"])
    out["vibration_only"]["required_model"] = np.bincount(MLP.load("model_weights.npz").predict(
        sp["norm"].transform(np.array([w.raw for w in __import__("generate_dataset").messages_to_windows(
            __import__("generate_dataset").run_simulator("vibration", 900, 321))]))), minlength=3).tolist()
    with open("variants_results.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote variants_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
