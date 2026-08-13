"""Score development-test windows with a calibrated model and threshold."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from lm_idnet.algorithms.dirichlet_multinomial import anomaly_score
from lm_idnet.algorithms.log_likelihood import (
    LogLikelihood,
    initialize_log_likelihood,
)
from lm_idnet.artifacts import load_artifact
from lm_idnet.config import AppConfig
from lm_idnet.exceptions import ArtifactCompatibilityError, DataValidationError
from lm_idnet.models.schemas import AnomalyEventArtifact
from lm_idnet.partitioning import development_partition_for_evaluation
from lm_idnet.processing.schemas import WindowRecord
from lm_idnet.processing.storage import load_processed_dataset

logger = logging.getLogger(__name__)


def score_window(
    window: WindowRecord,
    *,
    device_id: str,
    capture_id: str,
    model_categories: tuple[str, ...],
    expected_profile: tuple[float, ...],
    model_fingerprint: str,
    alpha: np.ndarray,
    log_likelihood: LogLikelihood,
    threshold: float,
    score_iqr: float,
    score_type: str,
) -> dict[str, object]:
    """Score one prepared window without reading or writing external state."""
    if window.categories != model_categories:
        raise DataValidationError(
            f"development-test categories do not match model: {capture_id}"
        )
    if window.counts is None:
        raise DataValidationError("cannot score a missing window")
    if score_iqr <= 0:
        raise DataValidationError("calibration score IQR must be positive")

    counts = np.asarray(window.counts, dtype=np.int64)
    score = anomaly_score(counts, alpha, log_likelihood, score_type)
    total = int(counts.sum())
    residuals = counts / total if total else np.zeros_like(counts)
    residuals = residuals - expected_profile
    return AnomalyEventArtifact(
        device_id=device_id,
        capture_id=capture_id,
        window_start_utc=window.start_utc,
        window_end_utc=window.end_utc,
        counts=dict(zip(model_categories, window.counts)),
        score_type=score_type,
        score=score,
        threshold=threshold,
        is_anomaly=score < threshold,
        severity=max(0.0, threshold - score) / score_iqr,
        expected_profile=dict(zip(model_categories, expected_profile)),
        category_residuals=dict(zip(model_categories, residuals)),
        model_fingerprint=model_fingerprint,
    ).model_dump(mode="json")


def score_windows(config: AppConfig) -> dict[str, object]:
    """Score every non-missing development-test window and save JSON Lines."""
    model = load_artifact(config.outputs.model_path, expected_type="model")
    threshold = load_artifact(
        config.outputs.threshold_path,
        expected_type="threshold",
    )
    score_type = config.calibration.score_type
    if threshold.score_type != score_type:
        raise ArtifactCompatibilityError(
            f"threshold score type must match configured {score_type!r}; "
            f"found {threshold.score_type!r}"
        )

    backend_name = model.log_likelihood_backend
    if not isinstance(backend_name, str):
        raise DataValidationError("model does not record a likelihood backend")
    log_likelihood = initialize_log_likelihood(backend_name)
    alpha = np.asarray(model.alpha, dtype=np.float64)
    model_categories = model.categories
    expected_profile = model.mean_probabilities
    model_fingerprint = hashlib.sha256(
        json.dumps(
            model.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    selection = development_partition_for_evaluation(config)
    processed_dir = config.ingest.processed_root / config.ingest.dataset_folder
    results: list[dict[str, object]] = []

    for capture_id in selection.capture_ids:
        dataset = load_processed_dataset(processed_dir / f"{capture_id}.json")
        if dataset.metadata.capture_id != capture_id:
            raise DataValidationError(
                f"processed capture ID does not match filename: {capture_id}"
            )
        if dataset.metadata.partition != selection.name:
            raise DataValidationError(
                f"development-test capture has the wrong partition: {capture_id}"
            )
        if any(window.categories != model_categories for window in dataset.windows):
            raise DataValidationError(
                f"development-test categories do not match model: {capture_id}"
            )

        for window in dataset.windows:
            if window.state == "missing":
                continue
            results.append(
                score_window(
                    window,
                    device_id=dataset.metadata.device_id,
                    capture_id=capture_id,
                    model_categories=model_categories,
                    expected_profile=expected_profile,
                    model_fingerprint=model_fingerprint,
                    alpha=alpha,
                    log_likelihood=log_likelihood,
                    threshold=threshold.threshold,
                    score_iqr=threshold.score_iqr,
                    score_type=score_type,
                )
            )

    output_path = Path(config.outputs.events_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "".join(
                json.dumps(result, sort_keys=True) + "\n" for result in results
            ),
            encoding="utf-8",
        )
    except OSError as error:
        raise DataValidationError(
            f"cannot save scoring results: {output_path}"
        ) from error

    anomaly_count = sum(result["is_anomaly"] is True for result in results)
    logger.info(
        "Saved %d development-test scores (%d anomalies) to %s",
        len(results),
        anomaly_count,
        output_path,
    )
    return {
        "partition": selection.name,
        "capture_ids": list(selection.capture_ids),
        "window_count": len(results),
        "anomaly_count": anomaly_count,
        "score_type": score_type,
        "events_path": str(output_path),
    }
