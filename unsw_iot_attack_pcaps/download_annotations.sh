#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT

curl --fail --location --silent --show-error \
    --output "$temporary/annotations.zip" \
    https://iotanalytics.unsw.edu.au/anomaly-data/annotations.zip
python -m zipfile -e "$temporary/annotations.zip" "$temporary"
mkdir -p "$root/annotations"
cp "$temporary/annotations/f4f5d88f0a3c.csv" "$root/annotations/"
