#!/usr/bin/env python3
"""Train the LogiEdge cold-chain classifier (Task D1).

Model: 2-hidden-layer MLP, 6 -> 32 -> 16 -> 3, ReLU activations, softmax output.
Input: the 6-value feature vector from preprocessing.py, normalised with the saved
       training_stats.npy (loaded, never recomputed).
Split: stratified 80 / 20 train / validation.
Gate:  validation accuracy must exceed 88%. Below that the script prints a failure message,
       exits with code 1 and saves no model.

Run:   python3 train_model.py                       (needs training_dataset.npz, training_stats.npy)
Saves: model_weights.npz, model_metadata.json

Pure numpy, so it has no deep-learning framework dependency; the weights are plain arrays that
can be loaded into any framework for the conversion step.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional

import numpy as np

from preprocessing import FEATURE_NAMES, Normaliser

CLASS_NAMES = ("Normal", "Warning", "Critical")
LAYERS = (6, 32, 16, 3)
ACCURACY_THRESHOLD = 0.88


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class MLP:
    """Fully connected network with ReLU hidden layers and a softmax output."""

    def __init__(self, layers=LAYERS, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.layers = tuple(layers)
        # He initialisation. The first layer is scaled down because normalised anomalous
        # windows are far from the Normal class (hundreds of sigma for temperature mean).
        self.W, self.b = [], []
        for i, (n_in, n_out) in enumerate(zip(layers[:-1], layers[1:])):
            scale = np.sqrt(2.0 / n_in) * (0.05 if i == 0 else 1.0)
            self.W.append(rng.normal(0.0, scale, (n_in, n_out)))
            self.b.append(np.zeros(n_out))

    @property
    def n_params(self) -> int:
        return sum(w.size + b.size for w, b in zip(self.W, self.b))

    def _forward(self, X):
        acts = [np.asarray(X, dtype=float)]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ w + b
            acts.append(z if i == len(self.W) - 1 else np.maximum(z, 0.0))
        return acts

    def predict_proba(self, X) -> np.ndarray:
        logits = self._forward(X)[-1]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self._forward(X)[-1], axis=1)

    def fit(self, X, y, epochs: int = 300, lr: float = 1e-3, batch: int = 32, seed: int = 0,
            X_val=None, y_val=None) -> dict:
        """Adam on cross-entropy. The validation set is only logged, never used to stop or tune."""
        rng = np.random.default_rng(seed)
        X, y = np.asarray(X, float), np.asarray(y)
        Y = np.eye(self.layers[-1])[y]
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        step = 0
        hist = {"loss": [], "train_acc": [], "val_acc": []}
        n_layers = len(self.W)
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for i in range(0, len(X), batch):
                idx = order[i:i + batch]
                acts = self._forward(X[idx])
                logits = acts[-1]
                p = np.exp(logits - logits.max(axis=1, keepdims=True))
                p /= p.sum(axis=1, keepdims=True)
                delta = (p - Y[idx]) / len(idx)
                gW, gb = [None] * n_layers, [None] * n_layers
                for k in range(n_layers - 1, -1, -1):
                    gW[k], gb[k] = acts[k].T @ delta, delta.sum(axis=0)
                    if k > 0:
                        delta = (delta @ self.W[k].T) * (acts[k] > 0)
                grads = gW + gb
                step += 1
                for j, (p_, g_) in enumerate(zip(params, grads)):
                    m[j] = 0.9 * m[j] + 0.1 * g_
                    v[j] = 0.999 * v[j] + 0.001 * g_ * g_
                    p_ -= lr * (m[j] / (1 - 0.9 ** step)) / (np.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)
            probs = self.predict_proba(X)
            hist["loss"].append(float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))))
            hist["train_acc"].append(float(np.mean(probs.argmax(1) == y)))
            if X_val is not None:
                hist["val_acc"].append(float(np.mean(self.predict(X_val) == y_val)))
        return hist

    def save(self, path: str) -> None:
        arrays = {}
        for i, (w, b) in enumerate(zip(self.W, self.b), start=1):
            arrays[f"W{i}"], arrays[f"b{i}"] = w, b
        np.savez(path, layers=np.array(self.layers), **arrays)

    @classmethod
    def load(cls, path: str) -> "MLP":
        data = np.load(path)
        model = cls(tuple(int(x) for x in data["layers"]))
        for i in range(len(model.W)):
            model.W[i], model.b[i] = data[f"W{i + 1}"].copy(), data[f"b{i + 1}"].copy()
        return model


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def stratified_split(y, val_frac: float = 0.2, seed: int = 42):
    """Return (train_idx, val_idx) with the same class proportions in both parts."""
    rng = np.random.default_rng(seed)
    train, val = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_val = int(round(val_frac * len(idx)))
        val += idx[:n_val].tolist()
        train += idx[n_val:].tolist()
    return np.array(sorted(train)), np.array(sorted(val))


def wilson_interval(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def evaluate(y_true, y_pred, n_classes: int = 3) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.array([[int(np.sum((y_true == a) & (y_pred == b))) for b in range(n_classes)] for a in range(n_classes)])
    correct = int(np.trace(cm))
    prec = [float(cm[c, c] / cm[:, c].sum()) if cm[:, c].sum() else float("nan") for c in range(n_classes)]
    rec = [float(cm[c, c] / cm[c].sum()) if cm[c].sum() else float("nan") for c in range(n_classes)]
    lo, hi = wilson_interval(correct, len(y_true))
    return {"accuracy": correct / len(y_true), "n": int(len(y_true)), "correct": correct,
            "ci95": [lo, hi], "confusion": cm.tolist(), "precision": prec, "recall": rec}


def load_dataset(path: str, normaliser: Normaliser, label_set: str = "mode"):
    d = np.load(path)
    X = normaliser.transform(d["raw"])
    y = d["y"] if label_set == "mode" else d["y_state"]
    return X, np.asarray(y), d


# --------------------------------------------------------------------------
def train_and_validate(X, y, seed: int = 42, epochs: int = 300, lr: float = 1e-3, batch: int = 32,
                       val_frac: float = 0.2):
    tr, va = stratified_split(y, val_frac, seed)
    model = MLP(seed=seed)
    hist = model.fit(X[tr], y[tr], epochs=epochs, lr=lr, batch=batch, seed=seed, X_val=X[va], y_val=y[va])
    return model, hist, tr, va


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train the LogiEdge classifier (Task D1)")
    p.add_argument("--dataset", default="training_dataset.npz")
    p.add_argument("--stats", default="training_stats.npy")
    p.add_argument("--out", default="model_weights.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--label-set", choices=("mode", "state"), default="mode",
                   help="mode: class from the simulator mode (Task D1); state: from the noise-free state")
    p.add_argument("--threshold", type=float, default=ACCURACY_THRESHOLD)
    p.add_argument("--augment-onset", action="store_true",
                   help="add simulator runs whose fault starts mid-run to the TRAINING part only "
                        "(validation stays the same 20%% split); recommended, see the D1 report")
    p.add_argument("--meta", default="model_metadata.json")
    args = p.parse_args(argv)

    normaliser = Normaliser.load(args.stats)
    X, y, _ = load_dataset(args.dataset, normaliser, args.label_set)
    if args.augment_onset:
        from generate_dataset import build_onset_dataset
        tr, va = stratified_split(y, 0.2, args.seed)
        extra = build_onset_dataset()
        Xtr = np.vstack([X[tr], normaliser.transform(extra["raw"])])
        ytr = np.concatenate([y[tr], extra["y"]])
        model = MLP(seed=args.seed)
        hist = model.fit(Xtr, ytr, epochs=args.epochs, lr=args.lr, batch=args.batch, seed=args.seed,
                         X_val=X[va], y_val=y[va])
        print(f"Augmented training set: {len(tr)} original + {len(extra['y'])} mid-run-onset windows")
    else:
        model, hist, tr, va = train_and_validate(X, y, args.seed, args.epochs, args.lr, args.batch)
    res = evaluate(y[va], model.predict(X[va]))

    print(f"Dataset: {len(y)} windows, class counts {np.bincount(y).tolist()}")
    print(f"Model:   {' -> '.join(map(str, LAYERS))}, ReLU, {model.n_params} parameters")
    print(f"Split:   {len(tr)} train / {len(va)} validation (stratified 80/20, seed {args.seed})")
    print(f"Training accuracy {hist['train_acc'][-1]:.4f}")
    print(f"Validation accuracy {res['accuracy']:.4f}  ({res['correct']}/{res['n']}, 95% CI "
          f"{res['ci95'][0]:.3f}-{res['ci95'][1]:.3f})")
    print("Confusion matrix (rows = true, columns = predicted):")
    for name, row in zip(CLASS_NAMES, res["confusion"]):
        print(f"  {name:9s}{row}")

    if res["accuracy"] <= args.threshold:
        print(f"\nFAILED: validation accuracy {res['accuracy']:.1%} does not exceed {args.threshold:.0%}. "
              "The model is not fit for cold-chain monitoring. Revisit the feature extraction or the "
              "architecture before continuing. No model was saved.", file=sys.stderr)
        return 1

    model.save(args.out)
    meta = {"layers": list(LAYERS), "activation": "relu", "class_names": list(CLASS_NAMES),
            "feature_names": list(FEATURE_NAMES), "normalisation": "z = (x - mean) / std using training_stats.npy",
            "label_set": args.label_set, "seed": args.seed, "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
            "n_params": model.n_params, "n_train": int(len(tr)), "n_val": int(len(va)), "augment_onset": bool(args.augment_onset),
            "validation": res, "history": hist}
    with open(args.meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nPASSED the {args.threshold:.0%} gate. Saved {args.out} and {args.meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
