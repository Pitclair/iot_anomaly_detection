"""Deterministic fingerprints for configured source datasets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from lm_idnet.config import AppConfig
from lm_idnet.exceptions import DataValidationError, IngestionError
from lm_idnet.partitioning import all_capture_ids, partition_name_for_capture


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> tuple[str, int]:
    try:
        initial_stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(chunk_size):
                digest.update(chunk)
        final_stat = path.stat()
    except FileNotFoundError as error:
        raise DataValidationError(f"dataset source is missing: {path}") from error
    except OSError as error:
        raise IngestionError(f"could not hash dataset source {path}: {error}") from error

    if (
        initial_stat.st_size != final_stat.st_size
        or initial_stat.st_mtime_ns != final_stat.st_mtime_ns
    ):
        raise DataValidationError(f"dataset source changed while hashing: {path}")
    return digest.hexdigest(), final_stat.st_size


def build_dataset_manifest(config: AppConfig) -> dict[str, Any]:
    """Build a canonical manifest from file bytes and all data-shaping policy."""
    ingest = config.ingest
    raw_directory = ingest.raw_root / ingest.dataset_folder

    ordered_files = []
    for capture_id in all_capture_ids(config):
        filename = f"{capture_id}.pcap"
        checksum, byte_size = _hash_file(raw_directory / filename)
        ordered_files.append(
            {
                "capture_id": capture_id,
                "partition": partition_name_for_capture(config, capture_id),
                # Relative names keep the identifier stable across clone locations.
                "relative_path": f"{ingest.dataset_folder}/{filename}",
                "byte_size": byte_size,
                "sha256": checksum,
            }
        )

    feature_taxonomy = {
        "ordered_categories": list(ingest.categories),
        "time_column": ingest.time_col,
        "protocol_column": ingest.protocol_col,
    }
    window_policy = {
        "window_minutes": ingest.window_minutes,
        "interval": "left_closed_right_open",
        "timestamp_timezone": "UTC",
    }
    partition_definition = config.ingest.partitions.model_dump(mode="json")
    device_identity = {
        "id": ingest.device_id,
        "mac": ingest.device_mac,
    }

    components = {
        "ordered_files": ordered_files,
        "feature_taxonomy": feature_taxonomy,
        "window_policy": window_policy,
        "partition_definition": partition_definition,
        "device_identity": device_identity,
    }
    component_hashes = {
        name: _canonical_hash(value) for name, value in components.items()
    }
    dataset_digest = _canonical_hash(components)
    return {
        "dataset_version": f"sha256:{dataset_digest}",
        **components,
        "component_hashes": component_hashes,
    }


def write_dataset_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    """Write a deterministic manifest atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, output_path)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise IngestionError(
            f"could not write dataset fingerprint manifest {output_path}: {error}"
        ) from error


def fingerprint_configured_dataset(
    config: AppConfig,
    output_path: Path,
) -> dict[str, Any]:
    manifest = build_dataset_manifest(config)
    write_dataset_manifest(manifest, output_path)
    return manifest
