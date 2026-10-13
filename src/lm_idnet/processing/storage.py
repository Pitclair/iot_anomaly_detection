"""Save and load processed datasets in their single canonical format."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import ValidationError

from lm_idnet.exceptions import DataValidationError

from .schemas import ProcessedDataset, WindowRecord

if TYPE_CHECKING:
    from lm_idnet.config import IngestConfig


def validate_window_inputs(
    window: WindowRecord,
    categories: tuple[str, ...],
    window_minutes: int,
) -> None:
    """Reject window semantics or counts incompatible with the detector."""
    if window.categories != categories:
        raise DataValidationError("window categories do not match model")
    if window.end_utc - window.start_utc != timedelta(minutes=window_minutes):
        raise DataValidationError("window duration does not match model")
    if window.total_count is not None and window.total_count > np.iinfo(np.int64).max:
        raise DataValidationError(
            "window packet total exceeds the supported integer range"
        )


def validate_dataset_inputs(
    dataset: ProcessedDataset,
    ingest: "IngestConfig",
    capture_id: str,
    partition: str,
) -> None:
    """Apply the same input checks before fitting, calibration, and scoring."""
    if dataset.metadata.capture_id != capture_id:
        raise DataValidationError(
            f"processed capture ID does not match filename: {capture_id}"
        )
    if dataset.metadata.partition != partition:
        raise DataValidationError(
            f"processed capture is not marked as {partition}: {capture_id}"
        )
    if dataset.metadata.device_id != ingest.device_id:
        raise DataValidationError(
            f"processed device does not match configuration: {capture_id}"
        )
    if dataset.metadata.input_policy != ingest.input_policy:
        raise DataValidationError(
            f"processed input policy does not match configuration: {capture_id}"
        )
    for window in dataset.windows:
        validate_window_inputs(window, ingest.categories, ingest.window_minutes)


def save_processed_dataset(dataset: ProcessedDataset, path: str | Path) -> None:
    """Write a validated processed dataset as straightforward JSON."""
    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            dataset.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise DataValidationError(
            f"cannot save processed dataset: {output_path}"
        ) from error


def load_processed_dataset(path: str | Path) -> ProcessedDataset:
    """Load and validate a processed dataset from canonical JSON."""
    input_path = Path(path)
    try:
        return ProcessedDataset.model_validate_json(
            input_path.read_text(encoding="utf-8")
        )
    except FileNotFoundError as error:
        raise DataValidationError(
            f"processed dataset not found: {input_path}"
        ) from error
    except (OSError, UnicodeError, ValueError, ValidationError) as error:
        raise DataValidationError(
            f"invalid processed dataset: {input_path}: {error}"
        ) from error
