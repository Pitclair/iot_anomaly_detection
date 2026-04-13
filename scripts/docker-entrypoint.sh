#!/bin/sh
set -e

MARKER="/srv/app/.deps_installed"
REQ_FILE="/srv/app/requirements.txt"

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
