#!/bin/sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

for config in configs/unsw_chromecast.json configs/unsw_samsung_camera.json
do
  ./scripts/run.sh train --config "$config"
  ./scripts/run.sh calibrate --config "$config"
  ./scripts/run.sh score --config "$config"
  ./scripts/run.sh evaluate --config "$config"
done

for config in configs/experiments/*_adaptive_threshold.json \
  configs/experiments/*_periodic_refit*.json
do
  ./scripts/run.sh adapt --config "$config"
  ./scripts/run.sh evaluate --config "$config"
done
