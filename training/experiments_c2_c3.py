#!/usr/bin/env python3
"""Experiments for Task C2 (normalisation-statistics shift) and Task C3 (fusion level).

Run:  python3 experiments_c2_c3.py            (needs preprocessing.py, simulator.py, numpy)
Writes experiment_results.json and prints a summary.

Data are generated with the Task C1 sensor models (no broker needed) and passed through the
real preprocessing pipeline. The classifier is a small reference MLP, a stand-in for the
project's trained model, so the experiments have something to run inference with.

Window labels (ground truth from the noise-free simulator state at the window end):
  deviation d = noise-free temperature offset from the 4 degC setpoint
  temperature class: 0 if d <= 1, 1 if 1 < d <= 3, 2 if d > 3
  vibration fault (bearing wear) = early refrigeration anomaly -> at least Class 1
  Scheme A ("OR"):  label = max(temperature class, vibration class)
  Scheme B (used for C2 and as the main C3 case): as A, but a vibration fault together with
            temperature more than 1 degC off setpoint is a unit-failure signature -> Class 2
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

import simulator as sim
from preprocessing import (FEATURE_NAMES, N_FEATURES, Normaliser, PreprocessingPipeline,
                           compute_training_stats)

MODES = ("none", "temp_drift", "vibration", "combined")
CLASS_NAMES = ("Normal", "Warning", "Critical")


# --------------------------------------------------------------------------
# Labelled data from the C1 sensor models
# --------------------------------------------------------------------------
def label_state(d: float, vib_fault: bool, scheme: str) -> int:
    t_cls = 0 if d <= 1.0 else (1 if d <= 3.0 else 2)
    v_cls = 1 if vib_fault else 0
    if scheme == "B" and vib_fault and d > 1.0:
        return 2
    return max(t_cls, v_cls)


def simulate_run(mode: str, duration: int, onset: int, seed: int) -> dict:
    rng_t, rng_v, _ = sim._make_rngs(seed)
    temp = sim.TemperatureSensor(rng_t, mode in ("temp_drift", "combined"))
    vib = sim.VibrationSensor(rng_v, mode in ("vibration", "combined"))
    run = {"temp": [], "vib": [], "d": [], "fault": []}
    for t in range(duration):
        active = mode != "none" and t >= onset
        run["temp"].append(temp.read(active))
        run["d"].append(sim.TEMP_DRIFT_PER_READING_C * temp.anomalous_readings)
        if t % 2 == 0:
            run["vib"].append(vib.read(active))
            run["fault"].append(bool(active and mode in ("vibration", "combined")))
    return run


def build_dataset(seeds: range, duration: int = 300, onset_range=(20, 150), base: int = 0) -> dict:
    raw, temp_s, vib_s, yA, yB, mode_of, onset_of, end_of = [], [], [], [], [], [], [], []
    for mi, mode in enumerate(MODES):
        for seed in seeds:
            onset = int(np.random.default_rng(base + seed * 7 + mi).integers(*onset_range))
            run = simulate_run(mode, duration, onset, base + seed * 10 + mi)
            pipe = PreprocessingPipeline(None, keep_samples=True)
            for t in range(duration):
                wins = pipe.push("temp", float(t), run["temp"][t])
                if t % 2 == 0:
                    wins += pipe.push("vib", float(t), run["vib"][t // 2])
                for w in wins:
                    end = int(w.end)
                    d, f = run["d"][end - 1], run["fault"][(end - 2) // 2]
                    raw.append(w.raw); temp_s.append(w.temp_samples); vib_s.append(w.vib_samples)
                    yA.append(label_state(d, f, "A")); yB.append(label_state(d, f, "B")); mode_of.append(mi)
                    onset_of.append(onset); end_of.append(end)
    return {"raw": np.array(raw), "temp_s": np.array(temp_s), "vib_s": np.array(vib_s),
            "yA": np.array(yA), "yB": np.array(yB), "mode": np.array(mode_of),
            "onset": np.array(onset_of), "end": np.array(end_of)}


def clean_sample_stats(seed: int = 2026, minutes: int = 10) -> dict:
    """Per-sample mean/std of the filtered Normal-class streams (for the data-level fusion variant)."""
    rng_t, rng_v, _ = sim._make_rngs(seed)
    temp, vib = sim.TemperatureSensor(rng_t, False), sim.VibrationSensor(rng_v, False)
    from preprocessing import MovingAverage
    ma_t, ma_v = MovingAverage(), MovingAverage()
    tx = [ma_t.update(temp.read(False)) for _ in range(minutes * 60)]
    vx = [ma_v.update(vib.read(False)) for _ in range(minutes * 30)]
    return {"t_mean": np.mean(tx), "t_std": np.std(tx, ddof=1), "v_mean": np.mean(vx), "v_std": np.std(vx, ddof=1)}


# --------------------------------------------------------------------------
# Reference classifier: small MLP in numpy (n_in -> 16 ReLU -> n_out), Adam
# --------------------------------------------------------------------------
class MLP:
    def __init__(self, n_in: int, n_out: int, hidden: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        # small initial weights: the z-scored inputs of anomalous windows are large
        self.W1 = rng.normal(0, 0.02, (n_in, hidden)); self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.1, (hidden, n_out)); self.b2 = np.zeros(n_out)
        self.n_in, self.n_out, self.hidden = n_in, n_out, hidden

    @property
    def n_params(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    @property
    def macs(self) -> int:
        return self.n_in * self.hidden + self.hidden * self.n_out

    def _fwd(self, X):
        h = np.maximum(0, X @ self.W1 + self.b1)
        return h, h @ self.W2 + self.b2

    def predict(self, X) -> np.ndarray:
        return np.argmax(self._fwd(np.asarray(X, float))[1], axis=1)

    def fit(self, X, y, epochs: int = 300, lr: float = 0.003, batch: int = 128, seed: int = 0):
        rng = np.random.default_rng(seed)
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
        step = 0
        Y = np.eye(self.n_out)[y]
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for i in range(0, len(X), batch):
                idx = order[i:i + batch]
                xb, yb = X[idx], Y[idx]
                h, logits = self._fwd(xb)
                p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                g = (p - yb) / len(idx)
                gW2, gb2 = h.T @ g, g.sum(0)
                gh = (g @ self.W2.T) * (h > 0)
                grads = [xb.T @ gh, gh.sum(0), gW2, gb2]
                step += 1
                for k, (p_, g_) in enumerate(zip(params, grads)):
                    m[k] = 0.9 * m[k] + 0.1 * g_
                    v[k] = 0.999 * v[k] + 0.001 * g_ * g_
                    p_ -= lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
        return self


def accuracy(pred, y) -> float:
    return float(np.mean(pred == y))


def recalls(pred, y) -> list:
    return [float(np.mean(pred[y == c] == c)) if np.any(y == c) else float("nan") for c in range(3)]


def confusion(pred, y) -> list:
    return [[int(np.sum((y == a) & (pred == b))) for b in range(3)] for a in range(3)]


# --------------------------------------------------------------------------
# C2: shifted normalisation statistics
# --------------------------------------------------------------------------
def experiment_c2(train, test, norm: Normaliser, n_models: int = 5) -> dict:
    Xtr = norm.transform(train["raw"])
    models = [MLP(N_FEATURES, 3, seed=s).fit(Xtr, train["yB"], seed=s) for s in range(n_models)]
    y = test["yB"]

    def evaluate(normaliser_for_test):
        accs, rec, cm = [], [], np.zeros((3, 3), int)
        for mdl in models:
            pred = mdl.predict(normaliser_for_test(test["raw"]))
            accs.append(accuracy(pred, y)); rec.append(recalls(pred, y)); cm += np.array(confusion(pred, y))
        return {"acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
                "recall": np.mean(rec, axis=0).tolist(), "confusion": cm.tolist()}

    out = {"n_train": int(len(Xtr)), "n_test": int(len(y)),
           "class_counts_test": np.bincount(y, minlength=3).tolist(),
           "majority_baseline": float(np.max(np.bincount(y)) / len(y))}
    out["correct"] = evaluate(norm.transform)
    out["plus3"] = evaluate(norm.shifted(+3.0).transform)
    out["minus3"] = evaluate(norm.shifted(-3.0).transform)
    out["per_feature_plus3"] = {FEATURE_NAMES[i]: evaluate(norm.shifted(+3.0, [i]).transform)["acc_mean"]
                                for i in range(N_FEATURES)}
    out["sweep"] = {str(k): evaluate(norm.shifted(float(k)).transform)["acc_mean"]
                    for k in (-30, -20, -10, -5, -3, -1, 0, 1, 3, 5, 10, 20, 30)}

    # What must never be done: recompute the statistics from live data (here: each run's own windows)
    accs = []
    for mdl in models:
        pred = np.zeros(len(y), int)
        for mi in range(len(MODES)):
            for rid in np.unique(test["run"][test["mode"] == mi]):
                sel = test["run"] == rid
                w = test["raw"][sel]
                mu, sd = w.mean(0), np.maximum(w.std(0), 1e-6)
                pred[sel] = mdl.predict((w - mu) / sd)
        accs.append(accuracy(pred, y))
    out["live_recompute"] = {"acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs))}

    # Detection delay: seconds from fault onset to the first window flagged Warning or Critical
    delays = {m: [] for m in MODES[1:]}
    for mdl in models:
        pred = mdl.predict(norm.transform(test["raw"]))
        for mi, mode in enumerate(MODES):
            if mode == "none":
                continue
            for rid in np.unique(test["run"][test["mode"] == mi]):
                sel = test["run"] == rid
                onset, ends, p = test["onset"][sel][0], test["end"][sel], pred[sel]
                hit = np.where((ends > onset) & (p >= 1))[0]
                delays[mode].append(float(ends[hit[0]] - onset) if len(hit) else float("nan"))
    out["detection_delay_s"] = {m: {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)),
                                    "max": float(np.nanmax(v)), "missed": int(np.sum(np.isnan(v)))}
                                for m, v in delays.items()}
    return out


# --------------------------------------------------------------------------
# C3: fusion levels
# --------------------------------------------------------------------------
def experiment_c3(train, test, norm: Normaliser, n_models: int = 5) -> dict:
    cs = clean_sample_stats()
    Ztr, Zte = norm.transform(train["raw"]), norm.transform(test["raw"])

    def samples(ds):   # data-level input: 30 temperature + 15 vibration filtered samples, z-scored per sensor
        return np.hstack([(ds["temp_s"] - cs["t_mean"]) / cs["t_std"], (ds["vib_s"] - cs["v_mean"]) / cs["v_std"]])

    Str, Ste = samples(train), samples(test)
    results = {}
    for scheme in ("A", "B"):
        ytr, yte = train["y" + scheme], test["y" + scheme]
        ttr = np.array([0 if d <= 1 else (1 if d <= 3 else 2) for d in train["d_end"]])
        vtr = train["fault_end"].astype(int)
        acc = {k: [] for k in ("data", "feature", "dec_max", "dec_rule")}
        miss = {k: {"vib": [], "temp": []} for k in acc}
        for s in range(n_models):
            data_m = MLP(Str.shape[1], 3, seed=s).fit(Str, ytr, seed=s)
            feat_m = MLP(6, 3, seed=s).fit(Ztr, ytr, seed=s)
            temp_m = MLP(3, 3, seed=s).fit(Ztr[:, :3], ttr, seed=s)
            vib_m = MLP(3, 2, seed=s).fit(Ztr[:, 3:], vtr, seed=s)

            def run_all(Z, S, drop=None):
                Z, S = Z.copy(), S.copy()
                if drop == "vib":
                    Z[:, 3:] = 0; S[:, 30:] = 0
                if drop == "temp":
                    Z[:, :3] = 0; S[:, :30] = 0
                t, v = temp_m.predict(Z[:, :3]), vib_m.predict(Z[:, 3:])
                if drop == "vib": v = np.zeros_like(v)
                if drop == "temp": t = np.zeros_like(t)
                rule = np.where((t == 2) | ((v == 1) & (t >= 1)), 2, np.maximum(t, v)) if scheme == "B" else np.maximum(t, v)
                return {"data": data_m.predict(S), "feature": feat_m.predict(Z),
                        "dec_max": np.maximum(t, v), "dec_rule": rule}

            for drop in (None, "vib", "temp"):
                preds = run_all(Zte, Ste, drop)
                for k, p in preds.items():
                    (acc[k] if drop is None else miss[k][drop]).append(accuracy(p, yte))
        sizes = {"data": (Str.shape[1] * 16 + 16 + 16 * 3 + 3, Str.shape[1] * 16 + 16 * 3),
                 "feature": (6 * 16 + 16 + 16 * 3 + 3, 6 * 16 + 16 * 3),
                 "dec_max": (3 * 16 + 16 + 16 * 3 + 3 + 3 * 16 + 16 + 16 * 2 + 2, 3 * 16 + 16 * 3 + 3 * 16 + 16 * 2),
                 "dec_rule": (3 * 16 + 16 + 16 * 3 + 3 + 3 * 16 + 16 + 16 * 2 + 2, 3 * 16 + 16 * 3 + 3 * 16 + 16 * 2)}
        results[scheme] = {k: {"acc_mean": float(np.mean(acc[k])), "acc_sd": float(np.std(acc[k])),
                               "vib_missing": float(np.mean(miss[k]["vib"])),
                               "temp_missing": float(np.mean(miss[k]["temp"])),
                               "params": sizes[k][0], "macs": sizes[k][1]} for k in acc}
    return results


# --------------------------------------------------------------------------
def attach_ground_truth(ds: dict, seeds: range, duration: int, onset_range, base: int) -> dict:
    """Add run ids and the end-of-window ground truth needed by the C3 comparison."""
    run, d_end, f_end = [], [], []
    rid = 0
    for mi, mode in enumerate(MODES):
        for seed in seeds:
            onset = int(np.random.default_rng(base + seed * 7 + mi).integers(*onset_range))
            r = simulate_run(mode, duration, onset, base + seed * 10 + mi)
            for end in range(30, duration + 1, 10):
                if end > duration - 1:      # window needs a sample at t >= end to be emitted
                    break
                run.append(rid); d_end.append(r["d"][end - 1]); f_end.append(r["fault"][(end - 2) // 2])
            rid += 1
    ds["run"], ds["d_end"], ds["fault_end"] = np.array(run), np.array(d_end), np.array(f_end)
    assert len(ds["run"]) == len(ds["raw"]), (len(ds["run"]), len(ds["raw"]))
    return ds


def main() -> int:
    t0 = time.time()
    norm = Normaliser.load("training_stats.npy")          # loaded, never recomputed
    train = attach_ground_truth(build_dataset(range(40), base=0), range(40), 300, (20, 150), 0)
    test = attach_ground_truth(build_dataset(range(30), base=100000), range(30), 300, (20, 150), 100000)
    print(f"train windows {len(train['raw'])}, test windows {len(test['raw'])}")
    res = {"c2": experiment_c2(train, test, norm), "c3": experiment_c3(train, test, norm)}
    res["data"] = {"train_class_counts_B": np.bincount(train["yB"], minlength=3).tolist(),
                   "test_class_counts_B": np.bincount(test["yB"], minlength=3).tolist(),
                   "test_class_counts_A": np.bincount(test["yA"], minlength=3).tolist()}
    with open("experiment_results.json", "w") as f:
        json.dump(res, f, indent=2)
    c2 = res["c2"]
    print("C2 correct %.4f | +3 sigma %.4f | -3 sigma %.4f | live recompute %.4f | majority %.4f" % (
        c2["correct"]["acc_mean"], c2["plus3"]["acc_mean"], c2["minus3"]["acc_mean"],
        c2["live_recompute"]["acc_mean"], c2["majority_baseline"]))
    for scheme in ("A", "B"):
        for k, v in res["c3"][scheme].items():
            print(scheme, k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
    print("done in %.0f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
