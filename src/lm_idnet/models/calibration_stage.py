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
from lm_idnet.models.modeling_stage import load_training_matrix
from lm_idnet.partitioning import calibration_partition_for_threshold
from lm_idnet.processing.schemas import WindowRecord, validate_window_contract
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
        if dataset.metadata.device_id != config.ingest.device_id:
            raise DataValidationError(
                f"calibration device does not match configuration: {capture_id}"
            )

        for window in dataset.windows:
            if window.categories != model_categories:
                raise DataValidationError(
                    f"calibration categories do not match model: {capture_id}"
                )
            validate_window_contract(window, model_categories, config.ingest.window_minutes)
            if window.state != "missing":
                windows.append(window)

    return tuple(windows)


def load_calibration_packet_count_limit(
    config: AppConfig,
    model_categories: tuple[str, ...],
) -> float:
    """Derive the fixed learning-volume guard from benign calibration counts."""
    windows = load_calibration_windows(config, model_categories)
    if len(windows) < config.calibration.minimum_samples:
        raise DataValidationError("volume limit requires enough benign calibration windows")
    limit = float(np.quantile(
        [window.total_count for window in windows],
        config.adaptation.volume_quantile, method="linear",
    ))
    if not np.isfinite(limit):
        raise DataValidationError("calibration packet-count limit must be finite")
    return limit


def save_threshold(
    path: str | Path,
    threshold_data: dict[str, Any],
) -> ThresholdArtifact:
    """Validate and save a threshold artifact."""
    threshold = ThresholdArtifact.model_validate(threshold_data)
    save_artifact(path, threshold)
    return threshold


def calibrate_threshold(config: AppConfig) -> dict[str, object]:
    """Calibrate the joint composition, packet-volume, and ARP score on normal traffic."""
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
    fit_counts = load_training_matrix(config).counts
    calibration_counts = np.asarray([window.counts for window in windows], dtype=np.int64)
    arp_index = model.categories.index("arp")
    fit_scores = np.asarray([
        anomaly_score(row, alpha, log_likelihood, config.calibration.score_type)
        for row in fit_counts
    ])
    fit_features = np.column_stack((
        fit_scores, -np.log1p(fit_counts.sum(axis=1)), -np.log1p(fit_counts[:, arp_index]),
    ))
    calibration_features = np.column_stack((
        scores, -np.log1p(calibration_counts.sum(axis=1)),
        -np.log1p(calibration_counts[:, arp_index]),
    ))
    centers = np.median(fit_features, axis=0)
    q1, q3 = np.quantile(fit_features, (0.25, 0.75), axis=0)
    # shortcut: constant fit features use unit scale; calibration still sets the joint cutoff.
    scales = np.where(q3 > q1, q3 - q1, 1.)
    joint_scores = np.min((calibration_features - centers) / scales, axis=1)
    joint_threshold = float(np.quantile(joint_scores, quantile, method="linear"))
    threshold_value = float(centers[0] + joint_threshold * scales[0])
    score_iqr = float(scales[0])
    count_tails = {
        name: {
            "threshold": float(-centers[index] - joint_threshold * scales[index]),
            "score_iqr": float(scales[index]),
        }
        for index, name in enumerate(("total", "arp"), start=1)
    }
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
            "count_tails": count_tails,
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
