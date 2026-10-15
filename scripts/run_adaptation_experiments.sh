#!/bin/sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [conservative|legacy]" >&2
  exit 2
fi

case "${1:-conservative}" in
  conservative) set -- configs/experiments/*_adaptive_threshold.json configs/experiments/*_periodic_refit*.json ;;
  legacy) set -- configs/experiments/legacy/*.json ;;
  *) echo "Usage: $0 [conservative|legacy]" >&2; exit 2 ;;
esac

# Reuse the saved starting model and threshold so historical comparisons stay fair.
for config in "$@"
do
  ./scripts/run.sh adapt --config "$config"
  ./scripts/run.sh evaluate --config "$config"
done
