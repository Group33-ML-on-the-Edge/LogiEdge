#!/usr/bin/env python3
"""Extra checks on the Task D1 model, beyond the single required 80 / 20 split.

A random 20% split of overlapping windows is optimistic: neighbouring windows share 20 of
their 30 seconds. These checks ask how much the headline number can be trusted:

  1. repeated stratified splits   - spread of validation accuracy over 50 random splits
  2. 5-fold cross-validation
  3. blocked split                - train on the first 80% of each run, validate on the last 20%,
                                    with a 2-window gap so no samples overlap
  4. independent runs             - the saved model on freshly simulated runs (new seeds)
  5. scikit-learn cross-check     - an independent MLP (32, 16) implementation on the same split
  6. temperature capped at 35 degC- same pipeline with a physically plausible temperature ceiling
  7. label consistency            - the saved model against the labels implied by the Task C
                                    class thresholds (state labels)

Run:  python3 validation_checks.py      (writes validation_checks.json)
"""

from __future__ import annotations

import json
import sys

import numpy as np

from generate_dataset import build_dataset, build_onset_dataset
from preprocessing import Normaliser
from train_model import MLP, evaluate, stratified_split, train_and_validate


def run_all(stats="training_stats.npy", dataset="training_dataset.npz", weights="model_weights.npz") -> dict:
    norm = Normaliser.load(stats)
    d = np.load(dataset)
    X, y = norm.transform(d["raw"]), d["y"]
    out = {}

    # 1. repeated stratified splits
    accs, crit_to_normal = [], 0
    for s in range(50):
        m, _, tr, va = train_and_validate(X, y, seed=s)
        p = m.predict(X[va])
        accs.append(float(np.mean(p == y[va])))
        crit_to_normal += int(np.sum((y[va] == 2) & (p == 0)))
    out["repeated_splits"] = {"n": 50, "mean": float(np.mean(accs)), "sd": float(np.std(accs)),
                              "min": float(np.min(accs)), "max": float(np.max(accs)),
                              "below_88": int(np.sum(np.array(accs) <= 0.88)),
                              "critical_predicted_normal": crit_to_normal}

    # 2. five-fold cross-validation (stratified)
    rng = np.random.default_rng(7)
    folds = [[] for _ in range(5)]
    for c in np.unique(y):
        for i, idx in enumerate(rng.permutation(np.where(y == c)[0])):
            folds[i % 5].append(idx)
    fold_acc = []
    for k in range(5):
        va = np.array(folds[k]); tr = np.array([i for j in range(5) if j != k for i in folds[j]])
        m = MLP(seed=k); m.fit(X[tr], y[tr], seed=k)
        fold_acc.append(float(np.mean(m.predict(X[va]) == y[va])))
    out["five_fold"] = {"folds": fold_acc, "mean": float(np.mean(fold_acc))}

    # 3. blocked (temporal) split with a 2-window gap
    tr, va = [], []
    for c in np.unique(y):
        idx = np.where(d["run"] == c)[0]
        n_val = int(round(0.2 * len(idx)))
        tr += idx[: len(idx) - n_val - 2].tolist()
        va += idx[len(idx) - n_val:].tolist()
    tr, va = np.array(tr), np.array(va)
    m = MLP(seed=3); m.fit(X[tr], y[tr], seed=3)
    out["blocked_split"] = evaluate(y[va], m.predict(X[va]))

    # 4. independent runs, scored with the saved model
    saved = MLP.load(weights)
    accs, cm = [], np.zeros((3, 3), int)
    for off in (1000, 2000, 3000, 4000, 5000):
        ds = build_dataset(seed_offset=off)
        r = evaluate(ds["y"], saved.predict(norm.transform(ds["raw"])))
        accs.append(r["accuracy"]); cm += np.array(r["confusion"])
    out["independent_runs"] = {"n_datasets": 5, "accuracies": accs, "mean": float(np.mean(accs)),
                               "min": float(np.min(accs)), "confusion": cm.tolist()}

    # 5. scikit-learn cross-check on the same split as the saved model
    try:
        from sklearn.neural_network import MLPClassifier
        tr, va = stratified_split(y, 0.2, 42)
        sk = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", max_iter=2000, random_state=0)
        sk.fit(X[tr], y[tr])
        out["sklearn_crosscheck"] = {"accuracy": float(sk.score(X[va], y[va]))}
    except ImportError:
        out["sklearn_crosscheck"] = None

    # 6. temperature capped at 35 degC
    capped = build_dataset(temp_cap_c=35.0)
    Xc, yc = norm.transform(capped["raw"]), capped["y"]
    m, _, tr, va = train_and_validate(Xc, yc, seed=42)
    out["temp_cap_35"] = evaluate(yc[va], m.predict(Xc[va]))

    # 7. saved model against threshold-based (state) labels on the independent runs
    agree, total, cm = 0, 0, np.zeros((3, 3), int)
    for off in (1000, 2000, 3000):
        ds = build_dataset(seed_offset=off)
        p = saved.predict(norm.transform(ds["raw"]))
        r = evaluate(ds["y_state"], p)
        agree += r["correct"]; total += r["n"]; cm += np.array(r["confusion"])
    out["state_label_consistency"] = {"accuracy": agree / total, "n": total, "confusion": cm.tolist()}

    # 8. fault starting part-way through a run (transition windows), original model vs one trained
    #    with extra mid-run-onset runs added
    extra = build_onset_dataset()
    Xa = np.vstack([X[stratified_split(y, 0.2, 42)[0]], norm.transform(extra["raw"])])
    ya = np.concatenate([y[stratified_split(y, 0.2, 42)[0]], extra["y"]])
    aug = MLP(seed=42); aug.fit(Xa, ya, seed=42)
    res8 = {}
    for name, mdl in (("original", saved), ("with_onset_runs", aug)):
        on_acc, on_cm = [], np.zeros((3, 3), int)
        for off in (1000, 2000, 3000, 4000, 5000):
            t = build_onset_dataset(onsets=(80, 160, 240, 320), seed_offset=off)
            r = evaluate(t["y"], mdl.predict(norm.transform(t["raw"])))
            on_acc.append(r["accuracy"]); on_cm += np.array(r["confusion"])
        st_acc = []
        for off in (1000, 2000, 3000, 4000, 5000):
            t = build_dataset(seed_offset=off)
            st_acc.append(evaluate(t["y"], mdl.predict(norm.transform(t["raw"])))["accuracy"])
        res8[name] = {"onset_runs_accuracy": float(np.mean(on_acc)), "onset_confusion": on_cm.tolist(),
                      "standard_runs_accuracy": float(np.mean(st_acc))}
    res8["n_extra_windows"] = int(len(extra["y"]))
    out["onset_augmentation"] = res8
    return out


def main() -> int:
    res = run_all()
    with open("validation_checks.json", "w") as f:
        json.dump(res, f, indent=2)
    r = res["repeated_splits"]
    print(f"50 random splits: mean {r['mean']:.4f} sd {r['sd']:.4f} min {r['min']:.4f} max {r['max']:.4f}; "
          f"below 88%: {r['below_88']}; Critical predicted Normal: {r['critical_predicted_normal']}")
    print("5-fold CV:", [round(a, 4) for a in res["five_fold"]["folds"]], "mean", round(res["five_fold"]["mean"], 4))
    b = res["blocked_split"]; print(f"blocked split: {b['accuracy']:.4f} ({b['correct']}/{b['n']}) confusion {b['confusion']}")
    i = res["independent_runs"]; print("independent runs:", [round(a, 4) for a in i["accuracies"]], "confusion", i["confusion"])
    print("sklearn cross-check:", res["sklearn_crosscheck"])
    c = res["temp_cap_35"]; print(f"temp cap 35: {c['accuracy']:.4f} ({c['correct']}/{c['n']})")
    s = res["state_label_consistency"]; print(f"state-label agreement: {s['accuracy']:.4f} (n={s['n']}) confusion {s['confusion']}")
    o = res["onset_augmentation"]
    for k in ("original", "with_onset_runs"):
        print(f"onset test, {k}: {o[k]['onset_runs_accuracy']:.4f}, confusion {o[k]['onset_confusion']}, standard runs {o[k]['standard_runs_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
