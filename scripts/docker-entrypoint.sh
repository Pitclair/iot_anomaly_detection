#!/bin/sh
set -e

MARKER="/srv/app/.deps_installed"
REQ_FILE="/srv/app/requirements.txt"
NATIVE_DIR="/srv/app/src/lm_idnet/algorithms/native"
LM_SOURCE="$NATIVE_DIR/LM_time_lib.c"
LM_LIBRARY="$NATIVE_DIR/LM_time_lib.so"

if ! command -v gcc >/dev/null 2>&1; then
  echo "Cannot compile the required LM library: gcc is not available" >&2
  exit 1
fi

if [ ! -f "$LM_SOURCE" ]; then
  echo "Cannot compile the required LM library: $LM_SOURCE was not found" >&2
  exit 1
fi

echo "Compiling native LM library..."
if ! gcc -O3 -fPIC -shared "$LM_SOURCE" -lm -o "$LM_LIBRARY"; then
  echo "Failed to compile the required LM library" >&2
  exit 1
fi

# Helper to check if path is writable
is_writable() {
  [ -w "$1" ] && return 0 || return 1
}

# If requirements.txt exists and we can write marker, try to install once
if [ -f "$REQ_FILE" ]; then
  if is_writable "/srv/app"; then
    if [ ! -f "$MARKER" ]; then
      echo "Installing dependencies inside container (first-run)..."
      pip install --no-cache-dir -r "$REQ_FILE" || true
      touch "$MARKER"
    else
      echo "Dependencies marker present, skipping pip install"
    fi
  else
    echo "/srv/app is not writable; skipping in-container pip install. Assuming image already has dependencies installed."
  fi
else
  echo "No requirements.txt found at $REQ_FILE; skipping dependency installation"
fi

# Execute the provided command
exec "$@"
