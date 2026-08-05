"""Fit and save the Dirichlet-multinomial normal-traffic model."""

from dataclasses import dataclass
import logging
from pathlib import Path
from time import perf_counter

import numpy as np

from lm_idnet.algorithms.dirichlet import create_initial_alpha, fixed_point_dirichlet
from lm_idnet.algorithms.log_likelihood import initialize_log_likelihood
from lm_idnet.artifact_schemas import CURRENT_SCHEMA_VERSION, ModelArtifact
from lm_idnet.artifacts import artifact_checksum
from lm_idnet.config import AppConfig
from lm_idnet.exceptions import ConvergenceError, DataValidationError
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


def save_model(
    path: str | Path,
    categories: tuple[str, ...],
    alpha: np.ndarray,
) -> ModelArtifact:
    """Save fitted parameters as a checksummed model artifact."""
    identity = {"categories": categories, "alpha": alpha.tolist()}
    model_data = {
        "artifact_type": "model",
        "schema_version": CURRENT_SCHEMA_VERSION,
        "model_version": f"dm-{artifact_checksum(identity)[:12]}",
        **identity,
    }
    model_data["checksum"] = artifact_checksum(model_data)
    model = ModelArtifact.model_validate(model_data)

    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return model


def train_model(config: AppConfig) -> dict[str, object]:
    """Fit the configured normal-traffic model and save it."""
    training = load_training_matrix(config)
    initial_alpha = create_initial_alpha(
        training.counts,
        config.estimator.initial_alpha_concentration,
    )
    backend = config.estimator.log_likelihood_backend
    log_likelihood = initialize_log_likelihood(backend)

    logger.info(
        "Starting model fit: rows=%d, categories=%d, backend=%s, "
        "tolerance=%g, max_iterations=%d",
        training.counts.shape[0],
        training.counts.shape[1],
        backend,
        config.estimator.tolerance_delta,
        config.estimator.max_iterations,
    )
    logger.info("Initial alpha: %s", initial_alpha.tolist())
    started = perf_counter()
    alpha, diagnostics = fixed_point_dirichlet(
        training.counts,
        log_likelihood,
        initial_alpha,
        config.estimator.tolerance_delta,
        config.estimator.max_iterations,
    )
    duration_seconds = perf_counter() - started

    if not diagnostics["converged"]:
        message = (
            "model fitting did not converge after "
            f"{diagnostics['iterations']} iterations"
        )
        logger.error(message)
        raise ConvergenceError(message)

    concentration = float(alpha.sum())
    psi = 1.0 / concentration
    model = save_model(config.outputs.model_path, config.ingest.categories, alpha)
    logger.info(
        "Model converged after %d iterations in %.3f seconds",
        diagnostics["iterations"],
        duration_seconds,
    )
    logger.info("Fitted alpha: %s", alpha.tolist())
    logger.info("Fitted concentration: %s", concentration)
    logger.info("Fitted psi: %s", psi)
    logger.info("Final log-likelihood: %s", diagnostics["ll_history"][-1])
    logger.info("Saved model to %s", config.outputs.model_path)

    return {
        "stage": "dirichlet_multinomial_training",
        "partition": "fit",
        "capture_ids": list(training.capture_ids),
        "matrix_shape": list(training.counts.shape),
        "missing_window_count": training.missing_window_count,
        "silent_window_count": training.silent_window_count,
        "categories": list(config.ingest.categories),
        "initial_alpha": initial_alpha.tolist(),
        "alpha": alpha.tolist(),
        "concentration": concentration,
        "psi": psi,
        "log_likelihood_backend": backend,
        "iterations": diagnostics["iterations"],
        "initial_log_likelihood": diagnostics["ll_history"][0],
        "final_log_likelihood": diagnostics["ll_history"][-1],
        "duration_seconds": duration_seconds,
        "model_path": str(config.outputs.model_path),
        "model_version": model.model_version,
        "model_checksum": model.checksum,
    }
