"""Calibrate an anomaly threshold from normal calibration windows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from lm_idnet.algorithms.dirichlet_multinomial import log_probability
from lm_idnet.algorithms.log_likelihood import initialize_log_likelihood
from lm_idnet.artifact_schemas import (
    CURRENT_SCHEMA_VERSION,
    ThresholdArtifact,
)
from lm_idnet.artifacts import artifact_checksum, load_model_for_scoring
from lm_idnet.config import AppConfig
from lm_idnet.exceptions import DataValidationError
from lm_idnet.partitioning import calibration_partition_for_threshold
from lm_idnet.processing.schemas import WindowRecord
from lm_idnet.processing.storage import load_processed_dataset

logger = logging.getLogger(__name__)


def load_calibration_windows(
    config: AppConfig,
    model: dict[str, Any],
) -> tuple[WindowRecord, ...]:
    """Load observed windows from the configured calibration captures."""
    selection = calibration_partition_for_threshold(config)
    processed_dir = config.ingest.processed_root / config.ingest.dataset_folder
    model_categories = tuple(model["categories"])

    windows: list[WindowRecord] = []
    for capture_id in selection.capture_ids:
        dataset = load_processed_dataset(processed_dir / f"{capture_id}.json")

        if dataset.metadata.capture_id != capture_id:
            raise DataValidationError(
                f"processed capture ID does not match filename: {capture_id}"
            )
        if dataset.metadata.partition != "calibration":
            raise DataValidationError(
                f"calibration capture has the wrong partition: {capture_id}"
            )

        for window in dataset.windows:
            if window.categories != model_categories:
                raise DataValidationError(
                    f"calibration categories do not match model: {capture_id}"
                )
            if window.state != "missing":
                windows.append(window)

    return tuple(windows)


def save_threshold(
    path: str | Path,
    threshold_data: dict[str, Any],
) -> ThresholdArtifact:
    """Validate, checksum, and save a threshold artifact."""
    candidate = ThresholdArtifact.model_validate(
        {**threshold_data, "checksum": "0" * 64}
    )
    normalized = candidate.model_dump(mode="json")
    normalized["checksum"] = artifact_checksum(normalized)
    threshold = ThresholdArtifact.model_validate(normalized)

    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            threshold.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise DataValidationError(
            f"cannot save threshold artifact: {output_path}"
        ) from error

    return threshold


def calibrate_threshold(config: AppConfig) -> dict[str, object]:
    """Calculate and save the configured lower-quantile threshold."""
    model = load_model_for_scoring(config.outputs.model_path)
    windows = load_calibration_windows(config, model)

    if len(windows) < config.calibration.minimum_samples:
        raise DataValidationError(
            "calibration requires at least "
            f"{config.calibration.minimum_samples} windows; found {len(windows)}"
        )

    backend_name = model["log_likelihood_backend"]
    if not isinstance(backend_name, str):
        raise DataValidationError("model does not record a likelihood backend")

    log_likelihood = initialize_log_likelihood(backend_name)
    alpha = np.asarray(model["alpha"], dtype=np.float64)
    scores = np.asarray(
        [
            log_probability(
                np.asarray(window.counts, dtype=np.int64),
                alpha,
                log_likelihood,
            )
            for window in windows
        ],
        dtype=np.float64,
    )
    quantile = config.calibration.quantile
    threshold_value = float(np.quantile(scores, quantile, method="linear"))
    selection = calibration_partition_for_threshold(config)

    threshold = save_threshold(
        config.outputs.threshold_path,
        {
            "artifact_type": "threshold",
            "schema_version": CURRENT_SCHEMA_VERSION,
            "model_checksum": model["checksum"],
            "score_type": "raw_log_probability",
            "quantile": quantile,
            "threshold": threshold_value,
            "calibration_capture_ids": selection.capture_ids,
            "calibration_window_count": len(windows),
            "calibration_start_utc": min(
                window.start_utc for window in windows
            ),
            "calibration_end_utc": max(window.end_utc for window in windows),
            "quantile_method": "linear",
            "score_minimum": float(scores.min()),
            "score_median": float(np.median(scores)),
            "score_maximum": float(scores.max()),
        },
    )
    logger.info(
        "Saved calibration threshold %.12g from %d windows to %s",
        threshold.threshold,
        len(windows),
        config.outputs.threshold_path,
    )

    return {
        "partition": "calibration",
        "capture_ids": list(selection.capture_ids),
        "window_count": len(windows),
        "quantile": quantile,
        "threshold": threshold.threshold,
        "threshold_path": str(config.outputs.threshold_path),
        "threshold_checksum": threshold.checksum,
    }
