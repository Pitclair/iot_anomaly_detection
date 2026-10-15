#!/bin/sh
set -eu

if [ "$#" -eq 0 ]; then
  set -- --help
fi

exec lm-idnet "$@"
