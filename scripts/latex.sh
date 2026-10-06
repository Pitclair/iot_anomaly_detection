#!/bin/sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE=${1:-}

if [ "$#" -ne 1 ] || [ ! -f "$ROOT_DIR/$SOURCE" ]; then
  echo "Usage: $0 path/to/document.tex" >&2
  exit 2
fi

mkdir -p "$ROOT_DIR/out"
cd "$ROOT_DIR"

HOST_UID=$(id -u)
HOST_GID=$(id -g)
export HOST_UID HOST_GID

exec docker compose -f compose.latex.yaml run --rm latex \
  -cd -pdf -outdir=/out \
  -interaction=nonstopmode -halt-on-error -file-line-error \
  "$SOURCE"
