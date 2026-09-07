"""Build and validate one timestamp-unique processed timeline."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from lm_idnet.config import AppConfig
from lm_idnet.exceptions import DataValidationError, IngestionError
from lm_idnet.partitioning import all_capture_ids, partition_name_for_capture

from .schemas import ProcessedDataset, WindowRecord
from .storage import load_processed_dataset

WindowIdentity = tuple[datetime, datetime]


def _load_datasets(
    config: AppConfig,
    capture_ids: Iterable[str],
) -> dict[str, ProcessedDataset]:
    processed_dir = config.ingest.processed_root / config.ingest.dataset_folder
    datasets = {}
    for capture_id in capture_ids:
        dataset = load_processed_dataset(processed_dir / f"{capture_id}.json")
        if dataset.metadata.capture_id != capture_id:
            raise DataValidationError(
                f"processed capture ID does not match filename: {capture_id}"
            )
        if dataset.metadata.device_id != config.ingest.device_id:
            raise DataValidationError(
                f"processed capture device does not match configuration: {capture_id}"
            )
        datasets[capture_id] = dataset
    return datasets


def _window_map(
    datasets: dict[str, ProcessedDataset],
) -> dict[WindowIdentity, list[tuple[str, WindowRecord]]]:
    windows: dict[WindowIdentity, list[tuple[str, WindowRecord]]] = {}
    for capture_id, dataset in datasets.items():
        for window in dataset.windows:
            windows.setdefault((window.start_utc, window.end_utc), []).append(
                (capture_id, window)
            )
    return windows


def _merge_fragments(
    config: AppConfig,
    datasets: dict[str, ProcessedDataset],
) -> tuple[dict[str, ProcessedDataset], list[dict[str, Any]], list[str]]:
    capture_ids = set(datasets)
    relevant_merges = [
        merge
        for merge in config.ingest.window_fragment_merges
        if set(merge.capture_ids) <= capture_ids
    ]
    partial_merges = [
        merge
        for merge in config.ingest.window_fragment_merges
        if set(merge.capture_ids) & capture_ids
        and not set(merge.capture_ids) <= capture_ids
    ]
    errors = [
        "window fragment merge is only partly present: " + ", ".join(merge.capture_ids)
        for merge in partial_merges
    ]
    resolutions = {
        (
            merge.window_start_utc,
            merge.window_end_utc,
            frozenset(merge.capture_ids),
        ): merge
        for merge in relevant_merges
    }
    used = set()
    resolved = []
    windows_by_capture = {
        capture_id: list(dataset.windows)
        for capture_id, dataset in datasets.items()
    }

    for identity, fragments in _window_map(datasets).items():
        if len(fragments) < 2:
            continue
        key = (*identity, frozenset(capture_id for capture_id, _ in fragments))
        merge = resolutions.get(key)
        if merge is None:
            continue
        used.add(key)
        if len(fragments) != len(merge.capture_ids):
            errors.append(
                f"window fragment merge has repeated windows at {identity[0].isoformat()}"
            )
            continue
        categories = {window.categories for _, window in fragments}
        if len(categories) != 1 or any(window.counts is None for _, window in fragments):
            errors.append(
                f"window fragments cannot be summed at {identity[0].isoformat()}"
            )
            continue

        counts = tuple(
            sum(values)
            for values in zip(*(window.counts for _, window in fragments))
        )
        target_window = WindowRecord(
            start_utc=identity[0],
            end_utc=identity[1],
            categories=fragments[0][1].categories,
            counts=counts,
            state="observed" if sum(counts) else "observed-silent",
        )
        for capture_id, _window in fragments:
            windows_by_capture[capture_id] = [
                window
                for window in windows_by_capture[capture_id]
                if (window.start_utc, window.end_utc) != identity
            ]
        windows_by_capture[merge.target_capture_id].append(target_window)
        windows_by_capture[merge.target_capture_id].sort(
            key=lambda window: (window.start_utc, window.end_utc)
        )
        resolved.append(
            {
                "capture_ids": list(merge.capture_ids),
                "target_capture_id": merge.target_capture_id,
                "window_start_utc": identity[0].isoformat(),
                "window_end_utc": identity[1].isoformat(),
                "source_counts": {
                    capture_id: list(window.counts)
                    for capture_id, window in fragments
                },
                "merged_counts": list(counts),
                "reason": merge.reason,
            }
        )

    for key, merge in resolutions.items():
        if key not in used:
            errors.append(
                "configured window fragment merge did not match processed data: "
                f"{merge.window_start_utc.isoformat()}"
            )

    canonical = {
        capture_id: ProcessedDataset(
            metadata=dataset.metadata,
            windows=tuple(windows_by_capture[capture_id]),
        )
        for capture_id, dataset in datasets.items()
    }
    return canonical, resolved, errors


def _overlaps(
    config: AppConfig,
    datasets: dict[str, ProcessedDataset],
) -> list[dict[str, Any]]:
    records = sorted(
        [
            (
                window.start_utc,
                window.end_utc,
                capture_id,
                window,
            )
            for capture_id, dataset in datasets.items()
            for window in dataset.windows
        ],
        key=lambda record: record[:3],
    )
    active: list[tuple[datetime, datetime, str, WindowRecord]] = []
    overlaps = []
    for record in records:
        start, end, capture_id, window = record
        active = [previous for previous in active if previous[1] > start]
        for previous_start, previous_end, previous_capture, previous_window in active:
            overlaps.append(
                {
                    "first_capture_id": previous_capture,
                    "first_partition": partition_name_for_capture(
                        config, previous_capture
                    ),
                    "first_start_utc": previous_start.isoformat(),
                    "first_end_utc": previous_end.isoformat(),
                    "first_counts": (
                        list(previous_window.counts)
                        if previous_window.counts is not None
                        else None
                    ),
                    "second_capture_id": capture_id,
                    "second_partition": partition_name_for_capture(config, capture_id),
                    "second_start_utc": start.isoformat(),
                    "second_end_utc": end.isoformat(),
                    "second_counts": (
                        list(window.counts) if window.counts is not None else None
                    ),
                    "identical_bounds": (
                        previous_start == start and previous_end == end
                    ),
                    "identical_counts": previous_window.counts == window.counts,
                }
            )
        active.append(record)
    return overlaps


def build_canonical_timeline(
    config: AppConfig,
    capture_ids: Iterable[str] | None = None,
) -> tuple[dict[str, ProcessedDataset], dict[str, Any]]:
    """Load configured captures, apply documented merges, and report overlaps."""
    selected_ids = tuple(capture_ids or all_capture_ids(config))
    datasets = _load_datasets(config, selected_ids)
    canonical, resolved, errors = _merge_fragments(config, datasets)
    overlaps = _overlaps(config, canonical)
    report = {
        "dataset_folder": config.ingest.dataset_folder,
        "device_id": config.ingest.device_id,
        "capture_count": len(canonical),
        "window_count": sum(len(dataset.windows) for dataset in canonical.values()),
        "resolved_window_count": len(resolved),
        "resolved_windows": resolved,
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
        "errors": errors,
        "accepted": not errors and not overlaps,
    }
    return canonical, report


def load_canonical_partition(
    config: AppConfig,
    capture_ids: Iterable[str],
) -> dict[str, ProcessedDataset]:
    """Return one complete, overlap-free capture partition."""
    datasets, report = build_canonical_timeline(config, capture_ids)
    if not report["accepted"]:
        raise DataValidationError(
            "processed timeline rejected: "
            f"{report['overlap_count']} unresolved overlap(s), "
            f"{len(report['errors'])} resolution error(s)"
        )
    return datasets


def _partition_ranges(
    config: AppConfig,
    datasets: dict[str, ProcessedDataset],
) -> tuple[dict[str, dict[str, str | None]], list[str]]:
    ranges = {}
    bounds = {}
    violations = []
    names = ("fit", "calibration", "development_test", "final_test")
    for name in names:
        windows = [
            window
            for capture_id in getattr(config.ingest.partitions, name)
            for window in datasets[capture_id].windows
        ]
        if not windows:
            ranges[name] = {"start_utc": None, "end_utc": None}
            violations.append(f"{name} contains no processed windows")
            continue
        start = min(window.start_utc for window in windows)
        end = max(window.end_utc for window in windows)
        bounds[name] = (start, end)
        ranges[name] = {"start_utc": start.isoformat(), "end_utc": end.isoformat()}
    for earlier, later in zip(names, names[1:]):
        if earlier in bounds and later in bounds and bounds[earlier][1] > bounds[later][0]:
            violations.append(
                f"{earlier} actual timestamps do not end before {later} begins"
            )
    return ranges, violations


def _write_report(report: dict[str, Any], output_path: Path) -> None:
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
            f"could not write timeline validation report {output_path}: {error}"
        ) from error


def validate_unique_timeline(config: AppConfig, output_path: Path) -> dict[str, Any]:
    """Write a global timeline report and reject unresolved or reordered windows."""
    datasets, report = build_canonical_timeline(config)
    ranges, violations = _partition_ranges(config, datasets)
    report["partition_ranges"] = ranges
    report["partition_order_violations"] = violations
    report["accepted"] = report["accepted"] and not violations
    _write_report(report, output_path)
    if not report["accepted"]:
        raise DataValidationError(
            "processed timeline rejected: "
            f"{report['overlap_count']} unresolved overlap(s), "
            f"{len(report['errors'])} resolution error(s), "
            f"{len(violations)} partition-order violation(s); "
            f"report written to {output_path}"
        )
    return report
