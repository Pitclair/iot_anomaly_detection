"""Inventory configured PCAP captures without loading them into memory."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from lm_idnet.config import AppConfig, DuplicateCaptureGroup
from lm_idnet.exceptions import DataValidationError, IngestionError

_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
_PCAP_BYTE_ORDERS = {
    b"\xd4\xc3\xb2\xa1": "<",  # little-endian, microsecond timestamps
    b"\xa1\xb2\xc3\xd4": ">",  # big-endian, microsecond timestamps
    b"\x4d\x3c\xb2\xa1": "<",  # little-endian, nanosecond timestamps
    b"\xa1\xb2\x3c\x4d": ">",  # big-endian, nanosecond timestamps
}


class CaptureFormatError(ValueError):
    """A capture cannot be counted as a complete classic PCAP file."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as capture:
        while chunk := capture.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def count_classic_pcap_packets(path: Path) -> int:
    """Count records in a classic PCAP while detecting incomplete records."""
    with path.open("rb") as capture:
        global_header = capture.read(24)
        if len(global_header) < 24:
            raise CaptureFormatError("truncated", "incomplete PCAP global header")

        byte_order = _PCAP_BYTE_ORDERS.get(global_header[:4])
        if byte_order is None:
            raise CaptureFormatError(
                "unsupported_format",
                "not a supported classic PCAP file",
            )

        # snaplen is the maximum record size declared by the capture. Checking
        # it before read() avoids trusting a corrupt length that could request
        # an excessive allocation.
        snaplen = struct.unpack(f"{byte_order}I", global_header[16:20])[0]

        packet_count = 0
        record_header = struct.Struct(f"{byte_order}IIII")
        while True:
            header = capture.read(record_header.size)
            if not header:
                return packet_count
            if len(header) != record_header.size:
                raise CaptureFormatError("truncated", "incomplete packet header")

            _seconds, _fraction, captured_length, original_length = (
                record_header.unpack(header)
            )
            if captured_length > snaplen or captured_length > original_length:
                raise CaptureFormatError(
                    "invalid_record",
                    "packet length exceeds the capture header limits",
                )
            packet_data = capture.read(captured_length)
            if len(packet_data) != captured_length:
                raise CaptureFormatError("truncated", "incomplete packet data")
            packet_count += 1


def _capture_date(capture_id: str) -> str | None:
    match = _DATE_PATTERN.search(capture_id)
    return match.group(1) if match else None


def inspect_capture(capture_id: str, path: Path) -> dict[str, Any]:
    """Return one inventory entry, preserving failures as explicit statuses."""
    entry: dict[str, Any] = {
        "capture_id": capture_id,
        "date": _capture_date(capture_id),
        "path": str(path),
        "present": path.is_file(),
        "byte_size": None,
        "packet_count": None,
        "sha256": None,
        "parser_status": "missing",
        "parser_message": None,
    }
    if not entry["present"]:
        return entry

    try:
        entry["byte_size"] = path.stat().st_size
        entry["sha256"] = _sha256_file(path)
        entry["packet_count"] = count_classic_pcap_packets(path)
        entry["parser_status"] = "readable"
    except CaptureFormatError as error:
        entry["parser_status"] = error.status
        entry["parser_message"] = str(error)
    except OSError as error:
        entry["parser_status"] = "unreadable"
        entry["parser_message"] = str(error)
    return entry


def _find_duplicate_groups(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    captures_by_checksum: dict[str, list[str]] = {}
    for entry in entries:
        checksum = entry["sha256"]
        if checksum:
            captures_by_checksum.setdefault(checksum, []).append(entry["capture_id"])

    return [
        {"sha256": checksum, "capture_ids": sorted(capture_ids)}
        for checksum, capture_ids in sorted(captures_by_checksum.items())
        if len(capture_ids) > 1
    ]


def _explanation_for(
    duplicate_ids: list[str],
    allowed_groups: tuple[DuplicateCaptureGroup, ...],
) -> str | None:
    duplicate_set = set(duplicate_ids)
    for allowed_group in allowed_groups:
        if duplicate_set == set(allowed_group.capture_ids):
            return allowed_group.reason
    return None


def build_capture_inventory(config: AppConfig) -> dict[str, Any]:
    """Inspect every configured capture and summarize missing and duplicate data."""
    ingest = config.ingest
    raw_directory = ingest.raw_root / ingest.dataset_folder
    capture_ids = ingest.training_dates + ingest.testing_dates
    entries = [
        inspect_capture(capture_id, raw_directory / f"{capture_id}.pcap")
        for capture_id in capture_ids
    ]

    duplicate_groups = _find_duplicate_groups(entries)
    for group in duplicate_groups:
        explanation = _explanation_for(
            group["capture_ids"],
            ingest.allowed_duplicate_captures,
        )
        group["explained"] = explanation is not None
        group["explanation"] = explanation

    missing = [entry["capture_id"] for entry in entries if not entry["present"]]
    unexplained_duplicates = [
        group for group in duplicate_groups if not group["explained"]
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_folder": ingest.dataset_folder,
        "raw_directory": str(raw_directory),
        "expected_count": len(entries),
        "present_count": len(entries) - len(missing),
        "missing_count": len(missing),
        "missing_capture_ids": missing,
        "duplicate_groups": duplicate_groups,
        "unexplained_duplicate_count": len(unexplained_duplicates),
        "captures": entries,
        "accepted": not missing and not unexplained_duplicates,
    }


def write_inventory(report: dict[str, Any], output_path: Path) -> None:
    """Write the report atomically so an interrupted write is never promoted."""
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
            f"could not write capture inventory {output_path}: {error}"
        ) from error


def inventory_configured_captures(config: AppConfig, output_path: Path) -> dict[str, Any]:
    """Build and persist an inventory, then enforce its acceptance conditions."""
    report = build_capture_inventory(config)
    write_inventory(report, output_path)

    failures: list[str] = []
    if report["missing_count"]:
        failures.append(f"{report['missing_count']} required capture(s) missing")
    if report["unexplained_duplicate_count"]:
        failures.append(
            f"{report['unexplained_duplicate_count']} unexplained duplicate group(s)"
        )
    if failures:
        raise DataValidationError(
            f"capture inventory rejected: {', '.join(failures)}; "
            f"report written to {output_path}"
        )
    return report
