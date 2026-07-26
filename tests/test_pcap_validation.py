import json
import struct
from pathlib import Path
from typing import Callable

import pytest

from lm_idnet.config import AppConfig
from lm_idnet.exceptions import DataValidationError
from lm_idnet.processing.pcap_validation import (
    build_validation_report,
    quarantine_fixture,
    validate_configured_captures,
    validate_pcap_file,
)

pytestmark = [pytest.mark.unit, pytest.mark.pcap]

ROOT = Path(__file__).resolve().parents[1]


def write_pcap(
    path: Path,
    records: list[tuple[int, int, bytes, int | None]],
    *,
    snaplen: int = 65535,
) -> None:
    global_header = struct.pack(
        "<IHHIIII",
        0xA1B2C3D4,
        2,
        4,
        0,
        0,
        snaplen,
        1,
    )
    with path.open("wb") as capture:
        capture.write(global_header)
        for seconds, fraction, packet, original_length in records:
            original_length = original_length or len(packet)
            capture.write(
                struct.pack(
                    "<IIII",
                    seconds,
                    fraction,
                    len(packet),
                    original_length,
                )
            )
            capture.write(packet)


def validation_config(
    config_factory: Callable[..., AppConfig],
    raw_root: Path,
    capture_ids: list[str],
) -> AppConfig:
    return config_factory(
        ingest={
            "raw_root": str(raw_root),
            "dataset_folder": "camera",
            "training_dates": [capture_ids[0]],
            "testing_dates": capture_ids[1:],
            "allowed_duplicate_captures": [],
        }
    )


def test_complete_capture_parses_to_end_of_file(tmp_path: Path) -> None:
    capture_path = tmp_path / "valid.pcap"
    write_pcap(
        capture_path,
        [(1, 0, b"one", None), (2, 500_000, b"two", None)],
    )

    result = validate_pcap_file(capture_path)

    assert result["status"] == "valid"
    assert result["complete_to_eof"] is True
    assert result["packet_count"] == 2
    assert result["warning_count"] == 0
    assert result["fatal_error"] is None


def test_recoverable_warnings_are_separate_from_fatal_errors(tmp_path: Path) -> None:
    capture_path = tmp_path / "warnings.pcap"
    write_pcap(
        capture_path,
        [
            (2, 0, b"first", 10),
            (1, 0, b"", None),
        ],
    )

    result = validate_pcap_file(capture_path)

    assert result["status"] == "warning"
    assert result["complete_to_eof"] is True
    assert result["packet_count"] == 2
    assert result["fatal_error"] is None
    assert {warning["code"] for warning in result["warnings"]} == {
        "out_of_order_timestamp",
        "packet_snaplen_truncation",
        "zero_length_packet",
    }


def test_invalid_timestamp_is_fatal(tmp_path: Path) -> None:
    capture_path = tmp_path / "invalid_timestamp.pcap"
    write_pcap(capture_path, [(1, 1_000_000, b"packet", None)])

    result = validate_pcap_file(capture_path)

    assert result["status"] == "fatal"
    assert result["complete_to_eof"] is False
    assert result["packet_count"] == 0
    assert result["fatal_error"]["code"] == "invalid_timestamp"
    assert result["fatal_error"]["record_number"] == 1


def test_warnings_before_fatal_corruption_are_preserved(tmp_path: Path) -> None:
    capture_path = tmp_path / "warning_then_fatal.pcap"
    write_pcap(capture_path, [(1, 0, b"short", 10)])
    with capture_path.open("ab") as capture:
        capture.write(struct.pack("<IIII", 2, 1_000_000, 1, 1))
        capture.write(b"x")

    result = validate_pcap_file(capture_path)

    assert result["status"] == "fatal"
    assert result["packet_count"] == 1
    assert result["warning_count"] == 1
    assert result["warnings"][0]["code"] == "packet_snaplen_truncation"
    assert result["fatal_error"]["code"] == "invalid_timestamp"


def test_deliberately_truncated_fixture_is_quarantined_and_classified(
    tmp_path: Path,
) -> None:
    fixture_hex = (
        ROOT / "tests" / "data" / "quarantine" / "truncated_capture.hex"
    ).read_text(encoding="utf-8").strip()
    source = tmp_path / "truncated_capture.pcap"
    source.write_bytes(bytes.fromhex(fixture_hex))

    quarantined = quarantine_fixture(source, tmp_path / "quarantine")
    result = validate_pcap_file(quarantined)

    assert quarantined.parent.name == "quarantine"
    assert result["status"] == "fatal"
    assert result["complete_to_eof"] is False
    assert result["fatal_error"]["code"] == "truncated_packet_data"
    assert result["fatal_error"]["record_number"] == 1


def test_configured_validation_is_deterministic_across_runs(
    tmp_path: Path,
    config_factory: Callable[..., AppConfig],
) -> None:
    capture_ids = ["camera-2020-10-08", "camera-2020-10-09"]
    raw_directory = tmp_path / "raw" / "camera"
    raw_directory.mkdir(parents=True)
    write_pcap(raw_directory / f"{capture_ids[0]}.pcap", [(1, 0, b"one", None)])
    write_pcap(raw_directory / f"{capture_ids[1]}.pcap", [(2, 0, b"two", 5)])
    config = validation_config(config_factory, tmp_path / "raw", capture_ids)

    first = build_validation_report(config)
    second = build_validation_report(config)

    assert first["captures"] == second["captures"]
    assert first["summary"] == second["summary"]
    assert first["validation_fingerprint"] == second["validation_fingerprint"]
    assert first["summary"]["parsed_to_eof_count"] == 2
    assert first["summary"]["packet_count"] == 2
    assert first["summary"]["warning_count"] == 1


def test_fatal_report_is_written_before_command_failure(
    tmp_path: Path,
    config_factory: Callable[..., AppConfig],
) -> None:
    capture_ids = ["camera-2020-10-08", "camera-2020-10-09"]
    raw_directory = tmp_path / "raw" / "camera"
    raw_directory.mkdir(parents=True)
    write_pcap(raw_directory / f"{capture_ids[0]}.pcap", [(1, 0, b"one", None)])
    # The second configured capture is deliberately absent.
    config = validation_config(config_factory, tmp_path / "raw", capture_ids)
    output_path = tmp_path / "reports" / "validation.json"

    with pytest.raises(DataValidationError, match="rejected 1 configured source"):
        validate_configured_captures(config, output_path)

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["accepted"] is False
    assert report["summary"]["fatal_count"] == 1
    assert report["captures"][1]["fatal_error"]["code"] == "missing_file"
