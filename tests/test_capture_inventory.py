import hashlib
import json
import struct
from pathlib import Path
from typing import Callable

import pytest

from lm_idnet.config import AppConfig
from lm_idnet.exceptions import DataValidationError
from lm_idnet.processing.inventory import (
    build_capture_inventory,
    inventory_configured_captures,
)

pytestmark = pytest.mark.unit


def write_pcap(path: Path, packets: list[bytes]) -> None:
    """Write a minimal little-endian classic PCAP fixture."""
    global_header = struct.pack(
        "<IHHIIII",
        0xA1B2C3D4,
        2,
        4,
        0,
        0,
        65535,
        1,
    )
    with path.open("wb") as capture:
        capture.write(global_header)
        for index, packet in enumerate(packets, start=1):
            capture.write(struct.pack("<IIII", index, 0, len(packet), len(packet)))
            capture.write(packet)


def inventory_config(
    config_factory: Callable[..., AppConfig],
    raw_root: Path,
    *,
    capture_ids: list[str],
    allowed_duplicates: list[dict[str, object]] | None = None,
) -> AppConfig:
    if len(capture_ids) != 4:
        raise ValueError("inventory tests require one capture per partition")
    return config_factory(
        ingest={
            "raw_root": str(raw_root),
            "dataset_folder": "camera",
            "partitions": {
                "fit": [capture_ids[0]],
                "calibration": [capture_ids[1]],
                "development_test": [capture_ids[2]],
                "final_test": [capture_ids[3]],
            },
            "allowed_duplicate_captures": allowed_duplicates or [],
        }
    )


def test_inventory_matches_present_filesystem_entries(
    tmp_path: Path,
    config_factory: Callable[..., AppConfig],
) -> None:
    capture_ids = [f"camera-2020-10-{day:02d}" for day in range(8, 12)]
    capture_id = capture_ids[0]
    raw_directory = tmp_path / "raw" / "camera"
    raw_directory.mkdir(parents=True)
    capture_path = raw_directory / f"{capture_id}.pcap"
    write_pcap(capture_path, [b"first", b"second"])
    config = inventory_config(
        config_factory,
        tmp_path / "raw",
        capture_ids=capture_ids,
    )

    report = build_capture_inventory(config)
    present = report["captures"][0]
    missing = report["captures"][1]

    assert report["expected_count"] == 4
    assert report["present_count"] == 1
    assert report["missing_count"] == 3
    assert present["date"] == "2020-10-08"
    assert present["partition"] == "fit"
    assert present["byte_size"] == capture_path.stat().st_size
    assert present["packet_count"] == 2
    assert present["sha256"] == hashlib.sha256(capture_path.read_bytes()).hexdigest()
    assert present["parser_status"] == "readable"
    assert missing["present"] is False
    assert missing["parser_status"] == "missing"


def test_missing_required_capture_fails_after_writing_report(
    tmp_path: Path,
    config_factory: Callable[..., AppConfig],
) -> None:
    config = inventory_config(
        config_factory,
        tmp_path / "raw",
        capture_ids=[f"camera-2020-10-{day:02d}" for day in range(8, 12)],
    )
    output_path = tmp_path / "reports" / "inventory.json"

    with pytest.raises(DataValidationError, match="required capture.*missing"):
        inventory_configured_captures(config, output_path)

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["accepted"] is False
    assert report["missing_count"] == 4


def test_unexplained_duplicate_checksum_is_rejected(
    tmp_path: Path,
    config_factory: Callable[..., AppConfig],
) -> None:
    capture_ids = [f"camera-2020-10-{day:02d}" for day in range(8, 12)]
    raw_directory = tmp_path / "raw" / "camera"
    raw_directory.mkdir(parents=True)
    for index, capture_id in enumerate(capture_ids):
        packet = b"identical" if index < 2 else f"unique-{index}".encode()
        write_pcap(raw_directory / f"{capture_id}.pcap", [packet])
    config = inventory_config(
        config_factory,
        tmp_path / "raw",
        capture_ids=capture_ids,
    )
    output_path = tmp_path / "inventory.json"

    with pytest.raises(DataValidationError, match="unexplained duplicate"):
        inventory_configured_captures(config, output_path)

    duplicate = json.loads(output_path.read_text())["duplicate_groups"][0]
    assert duplicate["capture_ids"] == capture_ids[:2]
    assert duplicate["explained"] is False


def test_documented_duplicate_checksum_is_accepted(
    tmp_path: Path,
    config_factory: Callable[..., AppConfig],
) -> None:
    capture_ids = [f"camera-2020-10-{day:02d}" for day in range(8, 12)]
    raw_directory = tmp_path / "raw" / "camera"
    raw_directory.mkdir(parents=True)
    for index, capture_id in enumerate(capture_ids):
        packet = b"identical" if index < 2 else f"unique-{index}".encode()
        write_pcap(raw_directory / f"{capture_id}.pcap", [packet])
    config = inventory_config(
        config_factory,
        tmp_path / "raw",
        capture_ids=capture_ids,
        allowed_duplicates=[
            {"capture_ids": capture_ids[:2], "reason": "Known mirrored capture"}
        ],
    )

    report = inventory_configured_captures(config, tmp_path / "inventory.json")

    assert report["accepted"] is True
    assert report["duplicate_groups"][0]["explained"] is True
    assert report["duplicate_groups"][0]["explanation"] == "Known mirrored capture"


def test_unsupported_capture_is_recorded_not_skipped(
    tmp_path: Path,
    config_factory: Callable[..., AppConfig],
) -> None:
    capture_ids = [f"camera-2020-10-{day:02d}" for day in range(8, 12)]
    capture_id = capture_ids[0]
    raw_directory = tmp_path / "raw" / "camera"
    raw_directory.mkdir(parents=True)
    capture_path = raw_directory / f"{capture_id}.pcap"
    capture_path.write_bytes(b"not a pcap file but long enough")
    config = inventory_config(
        config_factory,
        tmp_path / "raw",
        capture_ids=capture_ids,
    )

    entry = build_capture_inventory(config)["captures"][0]

    assert entry["present"] is True
    assert entry["byte_size"] == capture_path.stat().st_size
    assert entry["sha256"] is not None
    assert entry["packet_count"] is None
    assert entry["parser_status"] == "unsupported_format"
