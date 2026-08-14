#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
base=https://iotanalytics.unsw.edu.au/anomaly-data/pcap
pids=()

case ${1:-chromecast} in
    chromecast)
        destination=$root
        benign_dates=(
            2018-10-10 2018-10-11 2018-10-12 2018-10-13 2018-10-14
            2018-10-15 2018-10-16 2018-10-17 2018-10-18 2018-10-19
        )
        mixed_dates=(
            2018-10-20 2018-10-21 2018-10-22 2018-10-23
            2018-10-25 2018-10-26 2018-10-27
        )
        ;;
    samsung-camera)
        destination="$root/Samsung camera"
        benign_dates=(2018-05-28 2018-05-29 2018-05-30 2018-05-31)
        mixed_dates=(
            2018-06-01 2018-06-02 2018-06-03 2018-06-04
            2018-06-05 2018-06-06 2018-06-07 2018-06-08
        )
        ;;
    *)
        echo "usage: $0 [chromecast|samsung-camera]" >&2
        exit 2
        ;;
esac

wait_batch() {
    local status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=$?
    done
    pids=()
    return "$status"
}

download() {
    local group=$1 source=$2
    shift 2
    mkdir -p "$destination/$group"
    for date in "$@"; do
        local source_date=${date#20}
        curl --fail --location --continue-at - --retry 3 --silent --show-error \
            --output "$destination/$group/$date.pcap" "$base/$source/$source_date.pcap" &
        pids+=("$!")
        if ((${#pids[@]} == 4)); then
            wait_batch
        fi
    done
}

download benign benign "${benign_dates[@]}"
download mixed AttackAndBenign "${mixed_dates[@]}"

wait_batch
