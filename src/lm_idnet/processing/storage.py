"""Save and load processed datasets in their single canonical format."""

import json
from pathlib import Path

from pydantic import ValidationError

from lm_idnet.exceptions import DataValidationError

from .schemas import ProcessedDataset


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
