"""Tiny deterministic infrastructure experiment for reproducibility checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.random import Generator

from lm_idnet.config import load_config
from lm_idnet.exceptions import DataValidationError, IngestionError
from lm_idnet.processing.categories import validate_categories
from lm_idnet.randomness import create_named_generators


def load_smoke_fixture(path: Path) -> dict[str, Any]:
    try:
        fixture = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IngestionError(f"could not read smoke fixture {path}: {error}") from error
    if not isinstance(fixture, dict):
        raise DataValidationError("smoke fixture must be a JSON object")
    required_fields = {
        "metadata",
        "start_utc",
        "window_minutes",
        "window_count",
        "events",
        "capture_discontinuities",
        "expected_windows",
    }
    missing_fields = sorted(required_fields - fixture.keys())
    if missing_fields:
        raise DataValidationError(
            f"smoke fixture is missing required fields: {', '.join(missing_fields)}"
        )

    metadata = fixture["metadata"]
    if not isinstance(metadata, dict):
        raise DataValidationError("smoke fixture metadata must be an object")
    if metadata.get("synthetic") is not True or metadata.get("payload_free") is not True:
        raise DataValidationError("smoke fixture must be synthetic and payload-free")
    if metadata.get("license") != "CC0-1.0":
        raise DataValidationError("smoke fixture must declare the CC0-1.0 license")
    if not isinstance(fixture["events"], list):
        raise DataValidationError("smoke fixture events must be a list")
    return fixture


def _utc_timestamp(value: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise DataValidationError(f"invalid smoke timestamp: {value}") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise DataValidationError(f"smoke timestamp is not timezone-aware: {value}")
    return timestamp.astimezone(timezone.utc)


def aggregate_smoke_fixture(
    fixture: dict[str, Any],
    categories: Sequence[str],
) -> list[dict[str, Any]]:
    """Aggregate synthetic events into exact half-open windows, including silence."""
    categories = validate_categories(categories)
    window_minutes = fixture["window_minutes"]
    if not isinstance(window_minutes, int) or window_minutes <= 0:
        raise DataValidationError("smoke window_minutes must be positive")

    start = _utc_timestamp(fixture["start_utc"])
    window_count = fixture["window_count"]
    if not isinstance(window_count, int) or window_count <= 0:
        raise DataValidationError("smoke window_count must be positive")
    duration = timedelta(minutes=window_minutes)
    discontinuities = fixture["capture_discontinuities"]
    if not isinstance(discontinuities, list):
        raise DataValidationError("capture_discontinuities must be a list")
    missing_ranges: list[tuple[datetime, datetime]] = []
    for discontinuity in discontinuities:
        if not isinstance(discontinuity, dict) or set(discontinuity) != {"start_utc", "end_utc"}:
            raise DataValidationError(
                "each capture discontinuity must contain start_utc and end_utc"
            )
        gap_start = _utc_timestamp(discontinuity["start_utc"])
        gap_end = _utc_timestamp(discontinuity["end_utc"])
        if gap_end <= gap_start:
            raise DataValidationError("capture discontinuity end must follow its start")
        missing_ranges.append((gap_start, gap_end))

    windows = []
    for index in range(window_count):
        window_start = start + index * duration
        window_end = window_start + duration
        is_missing = any(
            gap_start < window_end and gap_end > window_start
            for gap_start, gap_end in missing_ranges
        )
        windows.append(
            {
                "start_utc": window_start.isoformat().replace("+00:00", "Z"),
                "end_utc": window_end.isoformat().replace(
                    "+00:00", "Z"
                ),
                "state": "missing" if is_missing else "observed-silent",
                "counts": None if is_missing else {category: 0 for category in categories},
            }
        )

    for event in fixture["events"]:
        if not isinstance(event, dict) or set(event) != {"timestamp", "category"}:
            raise DataValidationError(
                "each smoke event must contain only timestamp and category"
            )
        category = event["category"]
        if category not in categories:
            raise DataValidationError(f"unknown smoke category: {category}")
        timestamp = _utc_timestamp(event["timestamp"])
        elapsed = timestamp - start
        window_index = int(elapsed.total_seconds() // duration.total_seconds())
        if elapsed.total_seconds() < 0 or window_index >= window_count:
            raise DataValidationError(
                f"smoke event falls outside declared coverage: {event['timestamp']}"
            )
        if windows[window_index]["state"] == "missing":
            raise DataValidationError(
                f"smoke event falls within missing coverage: {event['timestamp']}"
            )
        windows[window_index]["counts"][category] += 1
        windows[window_index]["state"] = "observed"
    return windows


def _count_matrix(
    windows: list[dict[str, Any]],
    categories: Sequence[str],
) -> np.ndarray:
    observed_windows = [window for window in windows if window["state"] != "missing"]
    return np.asarray(
        [
            [window["counts"][category] for category in categories]
            for window in observed_windows
        ],
        dtype=np.int64,
    )


def fit_smoke_profile(counts: np.ndarray, generator: Generator) -> list[float]:
    """Create deterministic toy parameters; this is not the thesis estimator."""
    category_totals = counts.sum(axis=0).astype(float) + 1.0
    initialization = generator.uniform(0.95, 1.05, size=counts.shape[1])
    weighted_totals = category_totals * initialization
    parameters = 10.0 * weighted_totals / weighted_totals.sum()
    return parameters.tolist()


def inject_smoke_anomaly(
    counts: np.ndarray,
    generator: Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    injected = counts.copy()
    window_index = int(generator.integers(0, counts.shape[0]))
    category_index = int(generator.integers(0, counts.shape[1]))
    magnitude = int(generator.integers(5, 11))
    injected[window_index, category_index] += magnitude
    return injected, {
        "window_index": window_index,
        "category_index": category_index,
        "magnitude": magnitude,
    }


def run_smoke_experiment(
    fixture_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    fixture = load_smoke_fixture(fixture_path)
    config = load_config(config_path)
    categories = config.ingest.categories
    windows = aggregate_smoke_fixture(fixture, categories)
    counts = _count_matrix(windows, categories)
    generators = create_named_generators(config.seeds)

    model_parameters = fit_smoke_profile(counts, generators.model_fitting)
    injected_counts, injection = inject_smoke_anomaly(counts, generators.simulation)
    metric = float(np.abs(injected_counts - counts).sum())
    fixture_hash = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    return {
        "report_type": "infrastructure_smoke_experiment",
        "scientific_claim": "none",
        "float_tolerance": 1e-12,
        "fixture_sha256": fixture_hash,
        "seeds": {
            "simulation": config.seeds.simulation,
            "model_fitting": config.seeds.model_fitting,
        },
        "categories": categories,
        "windows": windows,
        "modeled_window_count": int(counts.shape[0]),
        "model_parameters": model_parameters,
        "injected_anomaly": injection,
        "metrics": {"total_absolute_count_change": metric},
    }


def write_smoke_report(report: dict[str, Any], output_path: Path) -> None:
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
        raise IngestionError(f"could not write smoke report {output_path}: {error}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    report = run_smoke_experiment(arguments.fixture, arguments.config)
    write_smoke_report(report, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
