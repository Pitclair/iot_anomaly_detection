#!/usr/bin/env sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 CONFIG" >&2
    exit 2
fi

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
config=$1
case $config in
    /*) ;;
    *) config=$project_root/$config ;;
esac
lm_idnet_command=${LM_IDNET_COMMAND:-$project_root/.venv/bin/lm-idnet}

cd "$project_root"
"$lm_idnet_command" train --config "$config"
"$lm_idnet_command" calibrate --config "$config"
"$lm_idnet_command" score --config "$config"
"$lm_idnet_command" evaluate --config "$config"
