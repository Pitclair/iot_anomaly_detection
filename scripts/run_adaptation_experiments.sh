#!/bin/sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Reuse the saved starting model and threshold so historical comparisons stay fair.
for config in configs/experiments/*_adaptive_threshold.json \
  configs/experiments/*_periodic_refit*.json
do
  ./scripts/run.sh adapt --config "$config"
  ./scripts/run.sh evaluate --config "$config"
done
