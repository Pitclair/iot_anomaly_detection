#!/usr/bin/env python3
"""Create separate ten-minute ground-truth labels for the UNSW Chromecast data."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

from lm_idnet.config import load_config
from lm_idnet.partitioning import all_capture_ids
from scapy.utils import RawPcapReader

DATASET = "unsw-iot-attack-traces"
WINDOW_SECONDS = 600
ETHERNET = 1
ARP = 0x0806
VLAN_TYPES = {0x8100, 0x88A8}


@dataclass(frozen=True)
class AttackInterval:
    start: datetime
    end: datetime
    affected_features: tuple[str, ...]
    attack_type: str


def load_annotations(path: Path) -> tuple[AttackInterval, ...]:
    intervals = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        for line_number, row in enumerate(csv.reader(source), start=1):
            if len(row) != 4:
                raise ValueError(f"annotation line {line_number} must have four fields")
            start_seconds, end_seconds = map(int, row[:2])
            if end_seconds <= start_seconds:
                raise ValueError(f"annotation line {line_number} has an invalid interval")
            intervals.append(
                AttackInterval(
                    start=datetime.fromtimestamp(start_seconds, timezone.utc),
                    end=datetime.fromtimestamp(end_seconds, timezone.utc),
                    affected_features=tuple(row[2].split("|")),
                    attack_type=row[3],
                )
            )
    return tuple(intervals)


def _matches_device(packet: bytes, mac: bytes) -> bool:
    if len(packet) < 14:
        return False
    if packet[:6] == mac or packet[6:12] == mac:
        return True

    ethertype = int.from_bytes(packet[12:14])
    payload = 14
    while ethertype in VLAN_TYPES and len(packet) >= payload + 4:
        ethertype = int.from_bytes(packet[payload + 2 : payload + 4])
        payload += 4
    if ethertype == ARP and len(packet) >= payload + 28:
        return (
            packet[payload + 8 : payload + 14] == mac
            or packet[payload + 18 : payload + 24] == mac
        )
    return False


def _timestamp(metadata: object, reader: object) -> float:
    if hasattr(metadata, "tshigh"):
        return (
            (metadata.tshigh << 32) + metadata.tslow
        ) / metadata.tsresol
    resolution = 1_000_000_000 if reader.nano else 1_000_000
    return metadata.sec + metadata.usec / resolution


def capture_windows(config_path: Path, capture_id: str) -> tuple[tuple[datetime, datetime], ...]:
    config = load_config(config_path)
    ingest = config.ingest
    mac = bytes.fromhex(ingest.device_mac.replace(":", ""))
    pcap = ingest.raw_root / ingest.dataset_folder / f"{capture_id}.pcap"
    first = last = None
    with RawPcapReader(str(pcap)) as reader:
        for packet, metadata in reader:
            if metadata.linktype != ETHERNET:
                raise ValueError(f"capture is not Ethernet: {capture_id}")
            if not _matches_device(packet, mac):
                continue
            timestamp = _timestamp(metadata, reader)
            first = timestamp if first is None else first
            last = timestamp
    if first is None or last is None:
        raise ValueError(f"capture has no packets for the configured device: {capture_id}")

    first_epoch = int(first) // WINDOW_SECONDS * WINDOW_SECONDS
    last_epoch = int(last) // WINDOW_SECONDS * WINDOW_SECONDS
    return tuple(
        (
            datetime.fromtimestamp(start, timezone.utc),
            datetime.fromtimestamp(start + WINDOW_SECONDS, timezone.utc),
        )
        for start in range(first_epoch, last_epoch + 1, WINDOW_SECONDS)
    )


def _overlap_seconds(
    window_start: datetime,
    window_end: datetime,
    intervals: tuple[AttackInterval, ...],
) -> int:
    clipped = sorted(
        (max(window_start, interval.start), min(window_end, interval.end))
        for interval in intervals
    )
    if not clipped:
        return 0
    merged = [clipped[0]]
    for start, end in clipped[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return int(sum((end - start).total_seconds() for start, end in merged))


def label_window(
    start: datetime,
    end: datetime,
    annotations: tuple[AttackInterval, ...],
) -> tuple[dict[str, object], tuple[AttackInterval, ...]]:
    overlapping = tuple(
        interval
        for interval in annotations
        if interval.start < end and interval.end > start
    )
    return (
        {
            "window_start_utc": start.isoformat().replace("+00:00", "Z"),
            "window_end_utc": end.isoformat().replace("+00:00", "Z"),
            "is_attack": bool(overlapping),
            "attack_types": sorted({item.attack_type for item in overlapping}),
            "affected_features": sorted(
                {feature for item in overlapping for feature in item.affected_features}
            ),
            "attack_overlap_seconds": _overlap_seconds(start, end, overlapping),
        },
        overlapping,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/unsw_chromecast.json"))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("unsw_iot_attack_pcaps/annotations/f4f5d88f0a3c.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/labels/UNSW-Chromecast")
    )
    args = parser.parse_args()

    config = load_config(args.config)
    annotations = load_annotations(args.annotations)
    fingerprint = "sha256:" + hashlib.sha256(args.annotations.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    matched: set[AttackInterval] = set()
    capture_summaries = []

    for capture_id in all_capture_ids(config):
        labels = []
        for start, end in capture_windows(args.config, capture_id):
            label, overlaps = label_window(start, end, annotations)
            labels.append(label)
            matched.update(overlaps)
        document = {
            "dataset": DATASET,
            "device_id": config.ingest.device_name,
            "capture_id": capture_id,
            "annotation_fingerprint": fingerprint,
            "windows": labels,
        }
        (args.output / f"{capture_id}.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        capture_summaries.append(
            {
                "capture_id": capture_id,
                "window_count": len(labels),
                "attack_window_count": sum(label["is_attack"] for label in labels),
            }
        )

    unmatched = [interval for interval in annotations if interval not in matched]
    manifest = {
        "dataset": DATASET,
        "device_id": config.ingest.device_name,
        "source_annotation": str(args.annotations),
        "annotation_fingerprint": fingerprint,
        "annotation_interval_count": len(annotations),
        "matched_interval_count": len(matched),
        "unmatched_intervals": [
            {
                "start_utc": item.start.isoformat().replace("+00:00", "Z"),
                "end_utc": item.end.isoformat().replace("+00:00", "Z"),
                "attack_type": item.attack_type,
            }
            for item in unmatched
        ],
        "captures": capture_summaries,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
