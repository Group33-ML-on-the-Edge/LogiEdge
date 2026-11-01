#!/usr/bin/env python3
"""Convert the trained numpy MLP (model_weights.npz) to model.tflite.

NOT RUN where this was written: it needs TensorFlow, which was not available there.
Run it on your machine first and read the verification output before building the image:

    python3.11 -m venv .venv && . .venv/bin/activate      # TensorFlow 2.14 has no Python 3.12 wheels
    pip install tensorflow==2.14.0 numpy==1.26.4 tflite-runtime==2.14.0
    python3 convert_to_tflite.py                                  # -> model.tflite
    python3 convert_to_tflite.py --weights model_weights_onset.npz --out model_onset.tflite

What it does:
  1. builds a Keras network with the same layers as train_model.py (6 -> 32 ReLU -> 16 ReLU -> 3 logits)
  2. copies the trained weights in and checks Keras matches the numpy model
  3. converts to a float32 TFLite model (no quantisation)
  4. runs the TFLite model on dataset windows and checks it matches the numpy model
  5. exits with an error if either check fails, so a bad conversion cannot be shipped silently

Use TensorFlow 2.14 so the operator versions in model.tflite match tflite-runtime 2.14.0 in the container.
A model made by a much newer TensorFlow can fail to load on the older runtime. Step 4b below tests exactly that.

The model outputs logits; inference_service.py applies softmax.
Note on INT8: the normalised inputs range from about -3 (Normal) to about 1,300 (a 70 degC window),
so naive full-integer quantisation of the input would crush the Normal range. Quantise only after
checking accuracy on the validation set.
"""

import argparse
import sys

import numpy as np


def numpy_logits(weights_path: str, X: np.ndarray) -> np.ndarray:
    d = np.load(weights_path)
    n = len(d["layers"]) - 1
    a = np.asarray(X, dtype=np.float64)
    for i in range(n):
        a = a @ d[f"W{i + 1}"] + d[f"b{i + 1}"]
        if i < n - 1:
            a = np.maximum(a, 0.0)
    return a


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="model_weights.npz")
    p.add_argument("--out", default="model.tflite")
    p.add_argument("--dataset", default="training_dataset.npz")
    p.add_argument("--stats", default="training_stats.npy")
    p.add_argument("--tolerance", type=float, default=1e-2, help="max allowed logit difference (relative to scale)")
    args = p.parse_args(argv)

    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow is not installed. Run: pip install tensorflow", file=sys.stderr)
        return 2
    from preprocessing import Normaliser

    w = np.load(args.weights)
    layers = [int(x) for x in w["layers"]]
    model = tf.keras.Sequential(
        [tf.keras.Input(shape=(layers[0],))]
        + [tf.keras.layers.Dense(n, activation="relu" if i < len(layers) - 2 else None)
           for i, n in enumerate(layers[1:])])
    for i, layer in enumerate(model.layers):
        layer.set_weights([w[f"W{i + 1}"].astype(np.float32), w[f"b{i + 1}"].astype(np.float32)])

    d = np.load(args.dataset)
    X = Normaliser.load(args.stats).transform(d["raw"]).astype(np.float32)
    ref = numpy_logits(args.weights, X)

    def check(name, got):
        scale = max(1.0, float(np.max(np.abs(ref))))
        diff = float(np.max(np.abs(got - ref))) / scale
        agree = float(np.mean(np.argmax(got, 1) == np.argmax(ref, 1)))
        print(f"{name}: max relative logit difference {diff:.2e}, class agreement {agree:.2%}")
        if diff > args.tolerance or agree < 1.0:
            print(f"FAILED: {name} does not match the numpy model", file=sys.stderr)
            return False
        return True

    if not check("Keras", model.predict(X, verbose=0)):
        return 1
    tflite_bytes = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    with open(args.out, "wb") as f:
        f.write(tflite_bytes)

    def run_interpreter(interp):
        interp.allocate_tensors()
        inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
        got = []
        for row in X:
            interp.set_tensor(inp["index"], row.reshape(1, -1))
            interp.invoke()
            got.append(interp.get_tensor(out["index"])[0])
        return np.array(got)

    if not check("TFLite (TensorFlow interpreter)", run_interpreter(tf.lite.Interpreter(model_content=tflite_bytes))):
        return 1
    # 4b. the runtime that actually runs in the container, if it is installed here
    try:
        from tflite_runtime.interpreter import Interpreter as RuntimeInterpreter
    except ImportError:
        print("NOTE: tflite_runtime is not installed here, so the container runtime was NOT tested. "
              "Install tflite-runtime==2.14.0 and rerun, or test inside the container.")
    else:
        if not check("tflite-runtime", run_interpreter(RuntimeInterpreter(model_path=args.out))):
            return 1
    print(f"Wrote {args.out}: {len(tflite_bytes):,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
