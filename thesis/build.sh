#!/usr/bin/env bash
set -euo pipefail

THESIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$THESIS_DIR/.." && pwd)"
mkdir -p "$PROJECT_DIR/out/thesis"
export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
cd "$PROJECT_DIR"
python3 "$THESIS_DIR/generate_evidence.py"

exec docker compose -f "$PROJECT_DIR/compose.latex.yaml" run --rm latex \
  -cd -lualatex -outdir=/out/thesis \
  -interaction=nonstopmode -halt-on-error -file-line-error \
  thesis/thesis.tex
