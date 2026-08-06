#!/bin/sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_NAME="${LM_IDNET_IMAGE_NAME:-lm-idnet:latest}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11-slim}"

echo "Building app image..."
docker build \
  --build-arg "PYTHON_VERSION=$PYTHON_VERSION" \
  --tag "$IMAGE_NAME" \
  "$ROOT_DIR"

mkdir -p data artifacts reports logs

exec docker run --rm \
  --env PYTHONUNBUFFERED=1 \
  --user "$(id -u):$(id -g)" \
  --volume "$ROOT_DIR/configs:/app/configs:ro" \
  --volume "$ROOT_DIR/data:/app/data" \
  --volume "$ROOT_DIR/artifacts:/app/artifacts" \
  --volume "$ROOT_DIR/reports:/app/reports" \
  --volume "$ROOT_DIR/logs:/app/logs" \
  "$IMAGE_NAME" \
  "$@"
