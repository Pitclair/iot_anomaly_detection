#!/bin/sh
set -e

NATIVE_DIR="/app/src/lm_idnet/algorithms/native"
LM_SOURCE="$NATIVE_DIR/LM_time_lib.c"
LM_LIBRARY="$NATIVE_DIR/LM_time_lib.so"
LM_BUILD_OUTPUT="$NATIVE_DIR/.LM_time_lib.so.tmp"

if ! command -v gcc >/dev/null 2>&1; then
  echo "Cannot compile the required LM library: gcc is not available" >&2
  exit 1
fi

if [ ! -f "$LM_SOURCE" ]; then
  echo "Cannot compile the required LM library: $LM_SOURCE was not found" >&2
  exit 1
fi

echo "Compiling native LM library..."
if ! gcc -O3 -fPIC -shared "$LM_SOURCE" -lm -o "$LM_BUILD_OUTPUT"; then
  echo "Failed to compile the required LM library" >&2
  exit 1
fi

mv "$LM_BUILD_OUTPUT" "$LM_LIBRARY"

if [ "$#" -eq 0 ]; then
  set -- --help
fi

exec lm-idnet "$@"
