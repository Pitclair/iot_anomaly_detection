"""Complete, bounded-memory validation for classic PCAP captures."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lm_idnet.config import AppConfig
from lm_idnet.exceptions import DataValidationError, IngestionError
from lm_idnet.partitioning import all_capture_ids, partition_name_for_capture

_PCAP_FORMATS = {
    b"\xd4\xc3\xb2\xa1": ("<", "microseconds", 1_000_000),
    b"\xa1\xb2\xc3\xd4": (">", "microseconds", 1_000_000),
    b"\x4d\x3c\xb2\xa1": ("<", "nanoseconds", 1_000_000_000),
    b"\xa1\xb2\x3c\x4d": (">", "nanoseconds", 1_000_000_000),
}
_WARNING_EXAMPLE_LIMIT = 10


@dataclass(frozen=True)
class FatalCaptureError(Exception):
    """A structural defect prevents a complete trustworthy parse."""

    code: str
    message: str
    byte_offset: int
    record_number: int | None = None
    completed_records: int = 0
    warning_count: int = 0
    warnings: list[dict[str, Any]] = field(default_factory=list)


class WarningCollector:
    """Aggregate warning counts while retaining only a few useful examples."""

    def __init__(self) -> None:
        self._warnings: dict[str, dict[str, Any]] = {}

    def add(self, code: str, message: str, record_number: int) -> None:
        warning = self._warnings.setdefault(
            code,
            {
                "code": code,
                "message": message,
                "count": 0,
                "example_record_numbers": [],
            },
        )
        warning["count"] += 1
        if len(warning["example_record_numbers"]) < _WARNING_EXAMPLE_LIMIT:
            warning["example_record_numbers"].append(record_number)

    def as_list(self) -> list[dict[str, Any]]:
        return [self._warnings[code] for code in sorted(self._warnings)]

    @property
    def count(self) -> int:
        return sum(warning["count"] for warning in self._warnings.values())


def _fatal(
    code: str,
    message: str,
    capture: Any,
    record_number: int | None = None,
) -> FatalCaptureError:
    return FatalCaptureError(code, message, capture.tell(), record_number)


def _fatal_record(
    code: str,
    message: str,
    byte_offset: int,
    record_number: int,
    completed_records: int,
    warnings: WarningCollector,
) -> FatalCaptureError:
    """Preserve recoverable warnings observed before a later fatal record."""
    return FatalCaptureError(
        code=code,
        message=message,
        byte_offset=byte_offset,
        record_number=record_number,
        completed_records=completed_records,
        warning_count=warnings.count,
        warnings=warnings.as_list(),
    )


def _parse_capture(path: Path) -> dict[str, Any]:
    warnings = WarningCollector()
    packet_count = 0
    previous_timestamp: tuple[int, int] | None = None

    with path.open("rb") as capture:
        global_header = capture.read(24)
        if len(global_header) != 24:
            raise _fatal(
                "truncated_global_header",
                "the 24-byte global header is incomplete",
                capture,
            )

        format_details = _PCAP_FORMATS.get(global_header[:4])
        if format_details is None:
            raise _fatal(
                "unsupported_format",
                "the file is not a supported classic PCAP capture",
                capture,
            )
        byte_order, timestamp_resolution, fractions_per_second = format_details

        version_major, version_minor = struct.unpack(
            f"{byte_order}HH",
            global_header[4:8],
        )
        if (version_major, version_minor) != (2, 4):
            raise _fatal(
                "unsupported_version",
                f"PCAP version {version_major}.{version_minor} is unsupported",
                capture,
            )

        snaplen = struct.unpack(f"{byte_order}I", global_header[16:20])[0]
        if snaplen == 0:
            raise _fatal(
                "invalid_global_header",
                "snapshot length must be greater than zero",
                capture,
            )

        record_header = struct.Struct(f"{byte_order}IIII")
        while True:
            record_offset = capture.tell()
            header = capture.read(record_header.size)
            if not header:
                break
            record_number = packet_count + 1
            if len(header) != record_header.size:
                raise _fatal_record(
                    "truncated_record_header",
                    "packet record header is incomplete",
                    record_offset,
                    record_number,
                    packet_count,
                    warnings,
                )

            seconds, fraction, captured_length, original_length = (
                record_header.unpack(header)
            )
            if fraction >= fractions_per_second:
                raise _fatal_record(
                    "invalid_timestamp",
                    f"timestamp fraction {fraction} exceeds {timestamp_resolution} range",
                    record_offset,
                    record_number,
                    packet_count,
                    warnings,
                )
            if captured_length > snaplen:
                raise _fatal_record(
                    "captured_length_exceeds_snaplen",
                    f"captured length {captured_length} exceeds snaplen {snaplen}",
                    record_offset,
                    record_number,
                    packet_count,
                    warnings,
                )
            if captured_length > original_length:
                raise _fatal_record(
                    "captured_length_exceeds_original",
                    "captured packet length exceeds original packet length",
                    record_offset,
                    record_number,
                    packet_count,
                    warnings,
                )

            packet_data = capture.read(captured_length)
            if len(packet_data) != captured_length:
                raise _fatal_record(
                    "truncated_packet_data",
                    "packet data ends before its declared captured length",
                    record_offset,
                    record_number,
                    packet_count,
                    warnings,
                )

            timestamp = (seconds, fraction)
            if previous_timestamp is not None and timestamp < previous_timestamp:
                warnings.add(
                    "out_of_order_timestamp",
                    "packet timestamp is earlier than the preceding record",
                    record_number,
                )
            if captured_length < original_length:
                warnings.add(
                    "packet_snaplen_truncation",
                    "capture stores fewer bytes than the original packet length",
                    record_number,
                )
            if captured_length == 0:
                warnings.add(
                    "zero_length_packet",
                    "packet record contains no captured bytes",
                    record_number,
                )

            previous_timestamp = timestamp
            packet_count += 1

    warning_list = warnings.as_list()
    return {
        "status": "warning" if warning_list else "valid",
        "complete_to_eof": True,
        "packet_count": packet_count,
        "timestamp_resolution": timestamp_resolution,
        "warning_count": warnings.count,
        "warnings": warning_list,
        "fatal_error": None,
    }


def validate_pcap_file(path: Path) -> dict[str, Any]:
    """Validate one capture and return a serializable result for every outcome."""
    result: dict[str, Any] = {
        "path": str(path),
        "byte_size": None,
        "status": "fatal",
        "complete_to_eof": False,
        "packet_count": 0,
        "timestamp_resolution": None,
        "warning_count": 0,
        "warnings": [],
        "fatal_error": None,
    }
    try:
        result["byte_size"] = path.stat().st_size
        result.update(_parse_capture(path))
    except FileNotFoundError:
        result["fatal_error"] = {
            "code": "missing_file",
            "message": "configured capture does not exist",
            "byte_offset": 0,
            "record_number": None,
        }
    except FatalCaptureError as error:
        result["fatal_error"] = {
            "code": error.code,
            "message": error.message,
            "byte_offset": error.byte_offset,
            "record_number": error.record_number,
        }
        # Counts and warnings before the fatal record remain useful diagnostic
        # evidence, but complete_to_eof stays false.
        result["packet_count"] = error.completed_records
        result["warning_count"] = error.warning_count
        result["warnings"] = error.warnings
    except OSError as error:
        result["fatal_error"] = {
            "code": "unreadable_file",
            "message": str(error),
            "byte_offset": 0,
            "record_number": None,
        }
    return result


def _deterministic_fingerprint(report: dict[str, Any]) -> str:
    stable_content = {
        "dataset_folder": report["dataset_folder"],
        "captures": report["captures"],
        "summary": report["summary"],
    }
    encoded = json.dumps(
        stable_content,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_validation_report(config: AppConfig) -> dict[str, Any]:
    """Parse every configured source and summarize warnings and fatal errors."""
    ingest = config.ingest
    raw_directory = ingest.raw_root / ingest.dataset_folder
    captures = []
    for capture_id in all_capture_ids(config):
        result = validate_pcap_file(raw_directory / f"{capture_id}.pcap")
        result["capture_id"] = capture_id
        result["partition"] = partition_name_for_capture(config, capture_id)
        captures.append(result)

    summary = {
        "configured_count": len(captures),
        "parsed_to_eof_count": sum(item["complete_to_eof"] for item in captures),
        "valid_count": sum(item["status"] == "valid" for item in captures),
        "warning_capture_count": sum(item["status"] == "warning" for item in captures),
        "fatal_count": sum(item["status"] == "fatal" for item in captures),
        "packet_count": sum(item["packet_count"] for item in captures),
        "warning_count": sum(item["warning_count"] for item in captures),
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_folder": ingest.dataset_folder,
        "raw_directory": str(raw_directory),
        "captures": captures,
        "summary": summary,
        "accepted": summary["fatal_count"] == 0,
    }
    report["validation_fingerprint"] = _deterministic_fingerprint(report)
    return report


def write_validation_report(report: dict[str, Any], output_path: Path) -> None:
    """Atomically persist validation evidence."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, output_path)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise IngestionError(
            f"could not write capture validation report {output_path}: {error}"
        ) from error


def validate_configured_captures(config: AppConfig, output_path: Path) -> dict[str, Any]:
    """Validate all sources, persist evidence, and fail on fatal corruption."""
    report = build_validation_report(config)
    write_validation_report(report, output_path)
    fatal_count = report["summary"]["fatal_count"]
    if fatal_count:
        raise DataValidationError(
            f"capture validation rejected {fatal_count} configured source(s); "
            f"report written to {output_path}"
        )
    return report


def quarantine_fixture(source: Path, quarantine_directory: Path) -> Path:
    """Copy a known-bad test fixture into an isolated quarantine directory."""
    quarantine_directory.mkdir(parents=True, exist_ok=True)
    destination = quarantine_directory / source.name
    try:
        shutil.copyfile(source, destination)
    except OSError as error:
        raise IngestionError(f"could not quarantine fixture {source}: {error}") from error
    return destination
