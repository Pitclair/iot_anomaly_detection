#!/bin/sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="${LM_IDNET_STATE_DIR:-$ROOT_DIR/artifacts/edge-state}"
LOG_DIR="${LM_IDNET_LOG_DIR:-$ROOT_DIR/logs/edge}"
IMAGE_NAME="${LM_IDNET_IMAGE_NAME:-lm-idnet:edge}"

if [ ! -f "$STATE_DIR/current.json" ]; then
  echo "Prepare the detector state in $STATE_DIR before starting; see docs/edge-detector.md" >&2
  exit 2
fi
mkdir -p "$LOG_DIR"
STATE_DIR="$(cd "$STATE_DIR" && pwd)"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
if [ "$#" -eq 0 ]; then
  set -- run --config /app/configs/edge_example.json --state-dir /state
fi
exec docker run --rm -i --pull never --network none --read-only \
  --memory 512m --cpus 1 --pids-limit 64 \
  --tmpfs /tmp:rw,noexec,nosuid,size=16m \
  --log-driver local --log-opt max-size=5m --log-opt max-file=3 \
  --user "$(id -u):$(id -g)" \
  --env LM_IDNET_LOG_PATH=/logs/runtime.log \
  --volume "$ROOT_DIR/configs:/app/configs:ro" \
  --volume "$STATE_DIR:/state" --volume "$LOG_DIR:/logs" \
  "$IMAGE_NAME" "$@"
