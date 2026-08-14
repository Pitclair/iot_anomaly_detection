#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

root=$(cd "$(dirname "$0")" && pwd)
workspace=$(cd "$root/.." && pwd)

case ${1:-chromecast} in
    chromecast)
        source_root=$root
        dataset=UNSW-Chromecast
        expected=17
        ;;
    samsung-camera)
        source_root="$root/Samsung camera"
        dataset=UNSW-Samsung-Camera
        expected=12
        ;;
    *)
        echo "usage: $0 [chromecast|samsung-camera]" >&2
        exit 2
        ;;
esac

target="$workspace/data/raw/$dataset"
sources=("$source_root"/benign/2018-*.pcap "$source_root"/mixed/2018-*.pcap)

if ((${#sources[@]} != expected)); then
    echo "expected $expected retained PCAPs, found ${#sources[@]}" >&2
    exit 1
fi

mkdir -p "$target"
for source in "${sources[@]}"; do
    filename=${source##*/}
    group=${source%/*}
    group=${group##*/}
    ln -sfn "$(realpath --relative-to="$target" "$source")" "$target/$filename"
done
