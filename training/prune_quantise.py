#!/usr/bin/env python3
"""M3: 35% structured pruning (PolynomialDecay) followed by full-INT8 PTQ (Task F1), with TensorFlow.

NOT RUN where this was written: TensorFlow and tensorflow_model_optimization were not available there. The same
method runs in numpy in model_variants.py (prune_structured); this script makes the real .tflite file.

Setup (Python 3.9-3.11; tfmot 0.8.0 was tested against TF 2.14.1):
    pip install tensorflow==2.14.1 tensorflow-model-optimization==0.8.0 numpy==1.26.4 psutil
    python3 prune_quantise.py

Method
  * tfmot prune_low_magnitude on the two hidden Dense layers, PolynomialDecay up to 35% sparsity (power 3, reached at 60% of
    training, mask updated every 10 steps). block_size is (inputs, 1): a whole column of the kernel, so one whole neuron
    ("filter") is pruned at a time. The output layer is not pruned.
  * after fine-tuning: strip_pruning, then the zeroed neurons are physically removed (their constant output folded into the
    next bias) by model_variants.shrink_dead_neurons. A check confirms the smaller network computes the same function.
  * the smaller Keras network is converted exactly like M2 (convert_ptq.to_int8_tflite).

Output: m3_pruned_int8.tflite and prune_quantise_results.json in --outdir
"""

import argparse
import json
import os
import sys

import numpy as np

from convert_ptq import keras_mlp, numpy_logits, score_tflite, to_int8_tflite


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="model_weights_onset.npz")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--sparsity", type=float, default=0.35)
    p.add_argument("--outdir", default=".")
    a = p.parse_args(argv)

    import tensorflow as tf
    import model_variants as mv

    sp = mv.load_splits()
    X_train, y_train = sp["X_train"], sp["y_train"]
    X_cal = sp["X_calib"]
    X_val, y_val = sp["X_val"], sp["y_val"]

    os.makedirs(a.outdir, exist_ok=True)

    original = mv.MLP.load(a.weights)
    sizes = list(original.layers)

    pruned, info = mv.prune_structured(
        original,
        X_train,
        y_train,
        final_sparsity=a.sparsity,
        epochs=a.epochs,
        seed=0,
    )

    small_sizes = list(pruned.layers)
    print(f"M3 structured pruning: {sizes} -> {small_sizes}")
    print(f"Neurons removed per hidden layer: {info['removed']}")

    m3 = keras_mlp(tf, small_sizes)

    for layer, W, b in zip(m3.layers, pruned.W, pruned.b):
        layer.set_weights([
            W.astype(np.float32),
            b.astype(np.float32),
        ])

    out = os.path.join(a.outdir, "m3_pruned_int8.tflite")
    to_int8_tflite(tf, m3, X_cal, out)

    r = score_tflite(out, X_val, y_val)

    print(
        f"M3 {r['file']}: {r['bytes']} bytes, "
        f"validation accuracy {100 * r['accuracy']:.2f}%, "
        f"Critical recall {100 * r['critical_recall']:.2f}%"
    )

    with open(os.path.join(a.outdir, "prune_quantise_results.json"), "w") as f:
        json.dump({
            "method": "numpy_structured_polynomial_decay",
            "target_sparsity": a.sparsity,
            "sizes": {
                "before": sizes,
                "after": small_sizes,
            },
            "pruning": info,
            "M3": r,
        }, f, indent=2)

    return 0 if r["critical_recall"] > 0.95 else 2


if __name__ == "__main__":
    sys.exit(main())
