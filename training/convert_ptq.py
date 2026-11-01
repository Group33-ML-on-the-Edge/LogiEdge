#!/usr/bin/env python3
"""M1 FP32 baseline and M2 post-training INT8 quantisation (Task F1), with TensorFlow.

NOT RUN where this was written: TensorFlow was not available there. The same variants run in numpy in
model_variants.py (the reported accuracy figures come from that); this script makes the real .tflite files.

Setup (Python 3.9-3.11; TF 2.14 matches tflite-runtime 2.14.0):
    python3.11 -m venv .venv && . .venv/bin/activate
    pip install tensorflow==2.14.1 numpy==1.26.4 psutil
    python3 convert_ptq.py --export model.tflite              # also copies the FP32 model to model.tflite

Outputs (in --outdir): m1_fp32_savedmodel/, m1_fp32.tflite, m2_ptq_int8.tflite, convert_ptq_results.json

M1  Keras network 6-32-16-3 loaded with the trained Task D1 weights (model_weights_onset.npz); SavedModel + float TFLite.
M2  TFLiteConverter, Optimize.DEFAULT, representative dataset of the training windows (at least 200 required),
    full integer (TFLITE_BUILTINS_INT8, int8 input and output).

--export-variant chooses which file --export copies: m1 (FP32, the Task F3 recommendation) or m2.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np


def keras_mlp(tf, sizes):
    layers = [tf.keras.Input(shape=(sizes[0],))]
    for i, n in enumerate(sizes[1:]):
        layers.append(tf.keras.layers.Dense(n, activation="relu" if i < len(sizes) - 2 else None))
    return tf.keras.Sequential(layers)


def numpy_logits(Ws, bs, X):
    a = np.asarray(X, dtype=np.float64)
    for i, (w, b) in enumerate(zip(Ws, bs)):
        a = a @ w + b
        if i < len(Ws) - 1:
            a = np.maximum(a, 0.0)
    return a


def to_float_tflite(tf, model, path):
    data = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    with open(path, "wb") as f:
        f.write(data)


def to_int8_tflite(tf, model, X_cal, path):
    """Full-integer post-training quantisation; needs at least 200 representative samples."""
    if len(X_cal) < 200:
        raise ValueError("full-integer PTQ needs at least 200 calibration samples")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for x in X_cal:
            yield [np.asarray(x, dtype=np.float32).reshape(1, -1)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    data = converter.convert()
    with open(path, "wb") as f:
        f.write(data)


def score_tflite(path, X_val, y_val):
    import benchmark
    ev = benchmark.evaluate_callable(benchmark.tflite_callable(path), X_val, y_val)
    return {"file": os.path.basename(path), "bytes": os.path.getsize(path), **ev}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="model_weights_onset.npz")
    p.add_argument("--outdir", default=".")
    p.add_argument("--export", help="copy the chosen variant here, e.g. model.tflite (the file the Dockerfile copies)")
    p.add_argument("--export-variant", choices=("m1", "m2"), default="m1")
    a = p.parse_args(argv)

    import tensorflow as tf
    import model_variants as mv

    sp = mv.load_splits()
    X_cal, X_val, y_val = sp["X_calib"], sp["X_val"], sp["y_val"]
    path = lambda name: os.path.join(a.outdir, name)
    os.makedirs(a.outdir, exist_ok=True)
    w = np.load(a.weights)
    sizes = [int(v) for v in w["layers"]]
    Ws = [w[f"W{i + 1}"] for i in range(len(sizes) - 1)]
    bs = [w[f"b{i + 1}"] for i in range(len(sizes) - 1)]

    m1 = keras_mlp(tf, sizes)
    for layer, W, b in zip(m1.layers, Ws, bs):
        layer.set_weights([W.astype(np.float32), b.astype(np.float32)])
    diff = float(np.max(np.abs(m1.predict(X_val.astype(np.float32), verbose=0) - numpy_logits(Ws, bs, X_val))))
    print(f"M1 Keras vs numpy logits, max difference {diff:.2e}")
    if diff > 1e-3:
        print("FAILED: the Keras model does not match the trained weights", file=sys.stderr)
        return 1
    m1.save(path("m1_fp32_savedmodel"), save_format="tf")
    to_float_tflite(tf, m1, path("m1_fp32.tflite"))
    to_int8_tflite(tf, m1, X_cal, path("m2_ptq_int8.tflite"))

    results = {"n_calibration": int(len(X_cal)), "M1": score_tflite(path("m1_fp32.tflite"), X_val, y_val),
               "M2": score_tflite(path("m2_ptq_int8.tflite"), X_val, y_val)}
    for name in ("M1", "M2"):
        r = results[name]
        print(f"{name} {r['file']}: {r['bytes']} bytes, validation accuracy {100 * r['accuracy']:.2f}%, "
              f"Critical recall {100 * r['critical_recall']:.2f}% (n={r['n']}, {r['n_critical']} Critical)")
    with open(path("convert_ptq_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    if a.export:
        shutil.copyfile(path("m1_fp32.tflite" if a.export_variant == "m1" else "m2_ptq_int8.tflite"), a.export)
        print(f"Copied the {a.export_variant.upper()} model to {a.export}")
    ok = all(results[n]["critical_recall"] > 0.95 for n in ("M1", "M2"))
    print("Both variants exceed 95% Critical recall" if ok else "WARNING: a variant is at or below 95% Critical recall")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
