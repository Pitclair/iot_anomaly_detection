"""Score development-test windows with a calibrated model and threshold."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import numpy as np

from lm_idnet.algorithms.dirichlet_multinomial import anomaly_score
from lm_idnet.algorithms.log_likelihood import initialize_log_likelihood
from lm_idnet.artifacts import load_artifact
from lm_idnet.config import AppConfig
from lm_idnet.exceptions import ArtifactCompatibilityError, DataValidationError
from lm_idnet.partitioning import development_partition_for_evaluation
from lm_idnet.processing.storage import load_processed_dataset

logger = logging.getLogger(__name__)


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

        for window in dataset.windows:
            if window.categories != model_categories:
                raise DataValidationError(
                    f"development-test categories do not match model: {capture_id}"
                )
            if window.state == "missing":
                continue

            counts = np.asarray(window.counts, dtype=np.int64)
            score = anomaly_score(counts, alpha, log_likelihood, score_type)
            results.append(
                {
                    "capture_id": capture_id,
                    "window_start_utc": window.start_utc.isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "window_end_utc": window.end_utc.isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "counts": window.counts,
                    "score": score,
                    "threshold": threshold.threshold,
                    "is_anomaly": score < threshold.threshold,
                }
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
