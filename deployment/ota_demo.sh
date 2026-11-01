#!/usr/bin/env bash
# OTA layer-cache demonstration (Task D2). Run on a machine with Docker; record the screen.
#
# 1. build the image            (cold cache)
# 2. change ONLY model.tflite
# 3. rebuild                    (only the model layer is rebuilt)
# 4. prove it from the layer digests
# 5. measure sizes and state the bandwidth saving for the fleet
#
# Needs: docker, python3, model.tflite (python3 convert_to_tflite.py), and for step 2 either
#        model_v2.tflite or model_weights_onset.npz plus TensorFlow to convert it.
# NOT RUN where this was written (no Docker there); read the output before recording.

set -euo pipefail

IMAGE="${IMAGE:-logiedge/inference}"
TRUCKS="${TRUCKS:-85}"
NEW_MODEL="${NEW_MODEL:-model_v2.tflite}"

banner() { printf '\n==== %s ====\n' "$*"; }
need()   { command -v "$1" >/dev/null 2>&1 || { echo "Missing required tool: $1" >&2; exit 1; }; }

need docker
need python3
[ -f model.tflite ] || { echo "model.tflite not found. Run: python3 convert_to_tflite.py" >&2; exit 1; }

banner "1. Build v1 from scratch (--no-cache)"
cp model.tflite model_v1.tflite
docker build --no-cache --progress=plain -t "$IMAGE:v1" . > build_v1.log 2>&1   # stops here if the build fails
grep -E '^#[0-9]+ \[[0-9]+/[0-9]+\]' build_v1.log || true

banner "2. Change ONLY the model file"
if [ ! -f "$NEW_MODEL" ]; then
  echo "$NEW_MODEL not found; converting model_weights_onset.npz"
  python3 convert_to_tflite.py --weights model_weights_onset.npz --out "$NEW_MODEL"
fi
cp "$NEW_MODEL" model.tflite
sha256sum model_v1.tflite model.tflite

banner "3. Rebuild v2 (cache on): only the model layer should be rebuilt"
docker build --progress=plain -t "$IMAGE:v2" . > build_v2.log 2>&1
grep -E '^#[0-9]+ (\[[0-9]+/[0-9]+\]|CACHED|DONE)' build_v2.log || true
echo
echo "Cached steps in the v2 build: $(grep -c ' CACHED' build_v2.log || true)"

banner "4. Layer digests: v1 against v2"
docker image inspect "$IMAGE:v1" --format '{{range .RootFS.Layers}}{{println .}}{{end}}' | sed '/^$/d' > layers_v1.txt
docker image inspect "$IMAGE:v2" --format '{{range .RootFS.Layers}}{{println .}}{{end}}' | sed '/^$/d' > layers_v2.txt
same=$(comm -12 <(sort layers_v1.txt) <(sort layers_v2.txt) | wc -l)
new=$(comm -13 <(sort layers_v1.txt) <(sort layers_v2.txt) | wc -l)
echo "Layers in v1: $(wc -l < layers_v1.txt)   in v2: $(wc -l < layers_v2.txt)"
echo "Identical in both: $same   New in v2: $new"
echo "A truck holding v1 downloads only the $new new layer(s)."

banner "5. Sizes and bandwidth saving for $TRUCKS trucks"
FULL_BYTES=$(docker save "$IMAGE:v2" | gzip -c | wc -c)
LAYER_BYTES=$(tar -cf - model.tflite | gzip -c | wc -c)
echo "Full image, compressed (approx.): $FULL_BYTES bytes"
echo "Model layer, compressed (approx.): $LAYER_BYTES bytes"
python3 ota_bandwidth.py --full-bytes "$FULL_BYTES" --layer-bytes "$LAYER_BYTES" --trucks "$TRUCKS"
