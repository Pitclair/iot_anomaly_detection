#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

source_root=$(cd "$(dirname "$0")" && pwd)
workspace=$(cd "$source_root/.." && pwd)
target="$workspace/data/raw/UNSW-Chromecast"
sources=("$source_root"/benign/2018-*.pcap "$source_root"/mixed/2018-*.pcap)

if ((${#sources[@]} != 17)); then
    echo "expected 17 retained PCAPs, found ${#sources[@]}" >&2
    exit 1
fi

mkdir -p "$target"
for source in "${sources[@]}"; do
    filename=${source##*/}
    group=${source%/*}
    group=${group##*/}
    ln -sfn "../../../unsw_iot_attack_pcaps/$group/$filename" "$target/$filename"
done
