#!/usr/bin/env bash
# Start a local Docker registry on port 5000 and push logiedge/inference:v1 (built in Task D2) to it.
# NOT RUN where this was written (no Docker there).
set -euo pipefail
TAG="${1:-v1}"
docker image inspect "logiedge/inference:${TAG}" >/dev/null 2>&1 || { echo "Build the image first: docker build -t logiedge/inference:${TAG} ." >&2; exit 1; }
docker ps --format '{{.Names}}' | grep -qx registry || docker run -d --restart unless-stopped -p 5000:5000 --name registry registry:2
docker tag "logiedge/inference:${TAG}" "localhost:5000/logiedge/inference:${TAG}"
docker push "localhost:5000/logiedge/inference:${TAG}"
echo "Pushed localhost:5000/logiedge/inference:${TAG}"
