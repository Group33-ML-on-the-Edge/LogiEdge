#!/usr/bin/env python3
"""Copy files generated in .work/ back to their home folders in the repository (only new or changed ones)."""
import filecmp
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, ".work")
HOME = {  # file name (or pattern by suffix) -> destination folder
    "training_dataset.npz": "training/data", "training_dataset.csv": "training/data",
    "training_stats.npy": "data_pipeline", "reference_dist.json": "monitoring",
    "model.tflite": "inference",
    "benchmark_results.csv": "optimisation/results", "benchmark_results.json": "optimisation/results",
    "benchmark_selfcheck.json": "optimisation/results", "pareto_chart.png": "optimisation/results",
    "drift_demo_results.json": "monitoring/results",
}
SUFFIX_HOME = [("_results.json", "training/results"), (".tflite", "training/models"), ("model_weights.npz", "training/models"),
               ("model_weights_onset.npz", "training/models"), ("model_metadata.json", "training/models"),
               ("model_metadata_onset.json", "training/models")]


def destination(name):
    if name in HOME:
        return HOME[name]
    for suffix, folder in SUFFIX_HOME:
        if name.endswith(suffix):
            return folder
    return None


def main() -> int:
    if not os.path.isdir(WORK):
        print("nothing to collect: run `make stage` first")
        return 1
    copied = []
    for name in sorted(os.listdir(WORK)):
        dest = destination(name)
        src = os.path.join(WORK, name)
        if dest is None or not os.path.isfile(src):
            continue
        target = os.path.join(ROOT, dest, name)
        if os.path.exists(target) and filecmp.cmp(src, target, shallow=False):
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(src, target)
        copied.append(os.path.relpath(target, ROOT))
    print("collected:" if copied else "nothing new to collect", *copied, sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
