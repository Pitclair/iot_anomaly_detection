"""Calibrate an anomaly threshold from normal calibration windows.
The baseline thesis should use empirical quantiles because they
are interpretable and do not assume Gaussian score distributions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from lm_idnet.algorithms.dirichlet_multinomial import anomaly_score
from lm_idnet.algorithms.log_likelihood import initialize_log_likelihood
from lm_idnet.artifacts import artifact_fingerprint, load_model, save_artifact
from lm_idnet.config import AppConfig
from lm_idnet.exceptions import DataValidationError
from lm_idnet.models.schemas import ThresholdArtifact
from lm_idnet.partitioning import calibration_partition_for_threshold
from lm_idnet.processing.schemas import WindowRecord
from lm_idnet.processing.storage import load_processed_dataset

logger = logging.getLogger(__name__)


def load_calibration_windows(
    config: AppConfig,
    model_categories: tuple[str, ...],
) -> tuple[WindowRecord, ...]:
    """Load observed windows from the configured calibration captures."""
    selection = calibration_partition_for_threshold(config)
    processed_dir = config.ingest.processed_root / config.ingest.dataset_folder
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
    """Validate and save a threshold artifact."""
    threshold = ThresholdArtifact.model_validate(threshold_data)
    save_artifact(path, threshold)
    return threshold


def _score_iqr(scores: np.ndarray) -> float:
    quartile_1, quartile_3 = np.quantile(
        scores, (0.25, 0.75), method="linear"
    )
    score_iqr = float(quartile_3 - quartile_1)
    if score_iqr <= 0:
        raise DataValidationError("calibration score IQR must be positive")
    return score_iqr


def calibrate_threshold(config: AppConfig) -> dict[str, object]:
    """Calculate and save the configured lower-quantile threshold."""
    model = load_model(config)
    windows = load_calibration_windows(config, model.categories)

    if len(windows) < config.calibration.minimum_samples:
        raise DataValidationError(
            "calibration requires at least "
            f"{config.calibration.minimum_samples} windows; found {len(windows)}"
        )

    log_likelihood = initialize_log_likelihood(
        model.log_likelihood_backend,
        model.precision_digits,
    )
    alpha = np.asarray(model.alpha, dtype=np.float64)
    scores = np.asarray(
        [
            anomaly_score(
                np.asarray(window.counts, dtype=np.int64),
                alpha,
                log_likelihood,
                config.calibration.score_type,
            )
            for window in windows
        ],
        dtype=np.float64,
    )
    quantile = config.calibration.quantile
    threshold_value = float(np.quantile(scores, quantile, method="linear"))
    score_iqr = _score_iqr(scores)
    selection = calibration_partition_for_threshold(config)

    threshold = save_threshold(
        config.outputs.threshold_path,
        {
            "artifact_type": "threshold",
            "model_fingerprint": artifact_fingerprint(model),
            "score_type": config.calibration.score_type,
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
            "score_iqr": score_iqr,
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
        "score_type": config.calibration.score_type,
        "quantile": quantile,
        "threshold": threshold.threshold,
        "threshold_path": str(config.outputs.threshold_path),
    }
