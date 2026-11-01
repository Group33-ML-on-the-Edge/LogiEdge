#!/usr/bin/env python3
"""Build all three TensorFlow variants (Task F1) by running convert_ptq.py and prune_quantise.py in turn.

NOT RUN where this was written (TensorFlow and tensorflow_model_optimization were not available).

    pip install tensorflow==2.14.1 tensorflow-model-optimization==0.8.0 numpy==1.26.4 psutil
    python3 build_variants_tf.py [--outdir DIR] [--export model.tflite]

M1 FP32 and M2 PTQ INT8 come from convert_ptq.py, M3 (35% structured pruning + PTQ INT8) from prune_quantise.py.
Exit code 0 only if every variant exceeds 95% Critical recall on the validation set.
"""

import argparse
import sys

import convert_ptq
import prune_quantise


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="model_weights_onset.npz")
    p.add_argument("--outdir", default=".")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--sparsity", type=float, default=0.35)
    p.add_argument("--export", help="copy the FP32 model here (the file the Dockerfile copies), e.g. model.tflite")
    a = p.parse_args(argv)
    first = ["--weights", a.weights, "--outdir", a.outdir] + (["--export", a.export] if a.export else [])
    code_a = convert_ptq.main(first)
    code_b = prune_quantise.main(["--weights", a.weights, "--outdir", a.outdir, "--epochs", str(a.epochs),
                                  "--sparsity", str(a.sparsity)])
    return max(code_a, code_b)


if __name__ == "__main__":
    sys.exit(main())
