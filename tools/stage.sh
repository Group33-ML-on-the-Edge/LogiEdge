#!/usr/bin/env bash
# Build .work/: a flat working directory with every script, data file and test side by side.
#
# The repository is organised by topic (data_pipeline/, training/, inference/, ...), but the scripts import each other by
# name and read their data from the current directory. Running them from .work/ is exactly how they were developed and
# tested, and it is the layout the commands in the reports assume. `make` does this for you.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/.work"
mkdir -p "$WORK"
# clear the previous staging (tolerant: some filesystems refuse to remove directories such as __pycache__)
find "$WORK" -mindepth 1 -maxdepth 1 ! -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
for d in data_pipeline training training/models training/data training/results inference monitoring monitoring/results \
         deployment optimisation optimisation/results tests; do
  [ -d "$ROOT/$d" ] || continue
  find "$ROOT/$d" -maxdepth 1 -type f ! -name '*.md' ! -name '.gitkeep' -exec cp {} "$WORK/" \;
done
echo "staged $(ls -A "$WORK" | wc -l) files in $WORK"
