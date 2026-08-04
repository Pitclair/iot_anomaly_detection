"""Load fit-partition counts and initialize the future DM model."""

from dataclasses import dataclass
import logging

import numpy as np

from lm_idnet.config import AppConfig
from lm_idnet.exceptions import DataValidationError
from lm_idnet.partitioning import fit_partition_for_training
from lm_idnet.processing.schemas import ProcessedDataset
from lm_idnet.processing.storage import load_processed_dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingMatrix:
    """Validated count rows selected for model fitting."""

    counts: np.ndarray
    capture_ids: tuple[str, ...]
    missing_window_count: int
    silent_window_count: int


def _validate_training_dataset(
    dataset: ProcessedDataset,
    capture_id: str,
    categories: tuple[str, ...],
) -> None:
    if dataset.metadata.capture_id != capture_id:
        raise DataValidationError(
            f"processed capture ID does not match filename: {capture_id}"
        )
    if dataset.metadata.partition != "fit":
        raise DataValidationError(
            f"training capture is not marked as fit: {capture_id}"
        )
    if any(window.categories != categories for window in dataset.windows):
        raise DataValidationError(
            f"training categories do not match configuration: {capture_id}"
        )


def load_training_matrix(config: AppConfig) -> TrainingMatrix:
    """Load non-missing count windows from every configured fit capture."""
    selection = fit_partition_for_training(config)
    processed_dir = config.ingest.processed_root / config.ingest.dataset_folder
    categories = config.ingest.categories
    rows: list[tuple[int, ...]] = []
    missing_window_count = 0
    silent_window_count = 0

    logger.info(
        "Loading %d fit captures from %s",
        len(selection.capture_ids),
        processed_dir,
    )
    for capture_id in selection.capture_ids:
        dataset_path = processed_dir / f"{capture_id}.json"
        logger.info("Loading training capture: %s", capture_id)
        dataset = load_processed_dataset(dataset_path)
        _validate_training_dataset(dataset, capture_id, categories)

        observed_windows = [
            window for window in dataset.windows if window.state != "missing"
        ]
        rows.extend(window.counts for window in observed_windows if window.counts is not None)
        missing_window_count += len(dataset.windows) - len(observed_windows)
        silent_window_count += sum(
            window.state == "observed-silent" for window in observed_windows
        )
        logger.info(
            "Loaded capture %s: %d model rows, %d missing windows",
            capture_id,
            len(observed_windows),
            len(dataset.windows) - len(observed_windows),
        )

    if not rows:
        raise DataValidationError("fit partition contains no observed count windows")

    counts = np.asarray(rows, dtype=np.int64)
    expected_shape = (len(rows), len(categories))
    if counts.shape != expected_shape:
        raise DataValidationError(
            f"training matrix has shape {counts.shape}; expected {expected_shape}"
        )

    logger.info("Training matrix ready with shape %s", counts.shape)
    logger.info("Training matrix contains %d silent rows", silent_window_count)
    return TrainingMatrix(
        counts=counts,
        capture_ids=selection.capture_ids,
        missing_window_count=missing_window_count,
        silent_window_count=silent_window_count,
    )


def create_initial_alpha(category_count: int) -> np.ndarray:
    """Create the simple neutral starting point alpha_k = 1."""
    if category_count < 2:
        raise ValueError("at least two categories are required")
    return np.ones(category_count, dtype=np.float64)


def initialize_modeling_stage(config: AppConfig) -> dict[str, object]:
    """Prepare model inputs without fitting or persisting a model."""
    training = load_training_matrix(config)
    initial_alpha = create_initial_alpha(len(config.ingest.categories))
    logger.info("Initial alpha: %s", initial_alpha.tolist())
    logger.info("Model fitting intentionally stops before fixed-point iteration")

    return {
        "stage": "dirichlet_multinomial_initialization",
        "partition": "fit",
        "capture_ids": list(training.capture_ids),
        "matrix_shape": list(training.counts.shape),
        "missing_window_count": training.missing_window_count,
        "silent_window_count": training.silent_window_count,
        "categories": list(config.ingest.categories),
        "initial_alpha": initial_alpha.tolist(),
        "fitting_performed": False,
        "model_saved": False,
    }
