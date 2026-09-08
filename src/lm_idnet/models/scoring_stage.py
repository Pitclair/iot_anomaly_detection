"""Score development-test windows with a calibrated model and threshold."""

from __future__ import annotations

from collections import deque
import json
import logging
from pathlib import Path

import numpy as np

from lm_idnet.algorithms.dirichlet_multinomial import anomaly_score
from lm_idnet.algorithms.estimator_factory import create_estimator
from lm_idnet.algorithms.log_likelihood import (
    LogLikelihood,
    initialize_log_likelihood,
)
from lm_idnet.artifacts import (
    artifact_fingerprint,
    load_artifact,
    load_model,
    model_parameter_fingerprint,
)
from lm_idnet.config import AppConfig
from lm_idnet.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
    LMIDNetError,
)
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


def _candidate_threshold(
    scores: deque[float],
    quantile: float,
) -> tuple[float, float] | None:
    values = np.asarray(scores, dtype=np.float64)
    quartile_1, quartile_3 = np.quantile(
        values, (0.25, 0.75), method="linear"
    )
    score_iqr = float(quartile_3 - quartile_1)
    if score_iqr <= 0:
        return None
    return (
        float(np.quantile(values, quantile, method="linear")),
        score_iqr,
    )


def _save_json(path: Path, value: object, description: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise DataValidationError(f"cannot save {description}: {path}") from error


def _run_detector(
    config: AppConfig,
    mode: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Run the configured detector over chronological development data."""
    model = load_model(config)
    threshold = load_artifact(
        config.outputs.threshold_path,
        expected_type="threshold",
    )
    model_fingerprint = artifact_fingerprint(model)
    if threshold.model_fingerprint != model_fingerprint:
        raise ArtifactCompatibilityError(
            "threshold model fingerprint does not match loaded model"
        )
    score_type = config.calibration.score_type
    if threshold.score_type != score_type:
        raise ArtifactCompatibilityError(
            f"threshold score type must match configured {score_type!r}; "
            f"found {threshold.score_type!r}"
        )

    log_likelihood = initialize_log_likelihood(
        model.log_likelihood_backend,
        model.precision_digits,
    )
    initial_alpha = np.asarray(model.alpha, dtype=np.float64)
    alpha = initial_alpha.copy()
    model_categories = model.categories
    expected_profile = model.mean_probabilities
    initial_threshold = threshold.threshold
    current_threshold = initial_threshold
    initial_score_iqr = threshold.score_iqr
    current_score_iqr = initial_score_iqr
    selection = development_partition_for_evaluation(config)
    processed_dir = config.ingest.processed_root / config.ingest.dataset_folder
    results: list[dict[str, object]] = []
    recent_scores: deque[float] = deque(maxlen=config.adaptation.buffer_size)
    recent_counts: deque[np.ndarray] = deque(maxlen=config.adaptation.buffer_size)
    safe_counts: deque[np.ndarray] = deque(maxlen=config.adaptation.buffer_size)
    threshold_updates: list[dict[str, object]] = []
    model_refit_attempts: list[dict[str, object]] = []
    current_model_fingerprint = model_fingerprint
    estimator = (
        create_estimator(config.estimator, config.precision_digits)
        if mode == "periodic_refit"
        else None
    )

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
                    model_fingerprint=current_model_fingerprint,
                    alpha=alpha,
                    log_likelihood=log_likelihood,
                    threshold=current_threshold,
                    score_iqr=current_score_iqr,
                    score_type=score_type,
                )
            )

            if mode == "static":
                continue

            score = float(results[-1]["score"])
            counts = np.asarray(window.counts, dtype=np.int64)
            recent_scores.append(score)
            if mode == "periodic_refit":
                recent_counts.append(counts)
                # ponytail: the immutable baseline limits poisoning; relax this
                # gate only if drift tests show that it rejects benign changes.
                baseline_score = (
                    score
                    if current_model_fingerprint == model_fingerprint
                    else anomaly_score(
                        counts,
                        initial_alpha,
                        log_likelihood,
                        score_type,
                    )
                )
                if baseline_score >= (
                    initial_threshold
                    + config.adaptation.safe_margin * initial_score_iqr
                ):
                    safe_counts.append(counts)

            if (
                len(results) % config.adaptation.update_every != 0
                or len(recent_scores) < config.calibration.minimum_samples
            ):
                continue

            update_context = {
                "after_window": len(results),
                "buffer_window_count": len(recent_scores),
                "capture_id": capture_id,
                "window_end_utc": window.end_utc.isoformat().replace(
                    "+00:00", "Z"
                ),
            }
            model_promoted = False
            if mode == "periodic_refit":
                refit: dict[str, object] = {
                    **update_context,
                    "safe_window_count": len(safe_counts),
                    "old_model_fingerprint": current_model_fingerprint,
                    "old_alpha": alpha.tolist(),
                    "promoted": False,
                }
                if len(safe_counts) < config.adaptation.minimum_refit_windows:
                    refit["reason"] = "not enough safe windows"
                else:
                    safe_matrix = np.asarray(safe_counts, dtype=np.int64)
                    # ponytail: one symmetric count per category keeps sparse
                    # buffers fit-able; make the prior configurable only if
                    # sensitivity analysis shows that its weight matters.
                    refit_matrix = np.vstack(
                        (
                            np.eye(len(model_categories), dtype=np.int64),
                            safe_matrix,
                        )
                    )
                    refit["smoothing_window_count"] = len(model_categories)
                    try:
                        assert estimator is not None
                        active_log_likelihood = log_likelihood.calculate(
                            refit_matrix,
                            alpha,
                        )
                        fit = estimator.fit(refit_matrix)
                        refit.update(
                            {
                                "active_log_likelihood": active_log_likelihood,
                                "candidate_log_likelihood": (
                                    fit.final_log_likelihood
                                ),
                                "candidate_iterations": fit.iterations,
                                "candidate_converged": fit.converged,
                            }
                        )
                        if not fit.converged:
                            refit["reason"] = "candidate did not converge"
                        # ponytail: this in-buffer guard is intentionally cheap;
                        # add a quarantined holdout if poisoning tests require it.
                        elif fit.final_log_likelihood < (
                            active_log_likelihood - config.estimator.tolerance_delta
                        ):
                            refit["reason"] = (
                                "candidate likelihood is worse than active model"
                            )
                        else:
                            candidate_scores = deque(
                                (
                                    anomaly_score(
                                        recent,
                                        fit.alpha,
                                        log_likelihood,
                                        score_type,
                                    )
                                    for recent in recent_counts
                                ),
                                maxlen=config.adaptation.buffer_size,
                            )
                            candidate_threshold = _candidate_threshold(
                                candidate_scores,
                                config.calibration.quantile,
                            )
                            if candidate_threshold is None:
                                refit["reason"] = (
                                    "candidate score IQR is not positive"
                                )
                            else:
                                new_threshold, new_score_iqr = candidate_threshold
                                new_model_fingerprint = model_parameter_fingerprint(
                                    alpha=tuple(float(value) for value in fit.alpha),
                                    categories=model_categories,
                                    log_likelihood_backend=(
                                        model.log_likelihood_backend
                                    ),
                                    precision_digits=model.precision_digits,
                                )
                                threshold_updates.append(
                                    {
                                        **update_context,
                                        "source": "periodic_refit",
                                        "old_model_fingerprint": (
                                            current_model_fingerprint
                                        ),
                                        "new_model_fingerprint": (
                                            new_model_fingerprint
                                        ),
                                        "old_score_iqr": current_score_iqr,
                                        "new_score_iqr": new_score_iqr,
                                        "old_threshold": current_threshold,
                                        "new_threshold": new_threshold,
                                        "promoted": True,
                                    }
                                )
                                refit.update(
                                    {
                                        "new_model_fingerprint": (
                                            new_model_fingerprint
                                        ),
                                        "new_alpha": fit.alpha.tolist(),
                                        "promoted": True,
                                    }
                                )
                                alpha = fit.alpha.copy()
                                expected_profile = tuple(
                                    float(value / fit.concentration)
                                    for value in fit.alpha
                                )
                                current_model_fingerprint = new_model_fingerprint
                                current_threshold = new_threshold
                                current_score_iqr = new_score_iqr
                                recent_scores = candidate_scores
                                model_promoted = True
                    except (LMIDNetError, ValueError) as error:
                        refit["reason"] = str(error)
                model_refit_attempts.append(refit)

            if model_promoted:
                continue

            candidate = _candidate_threshold(
                recent_scores,
                config.calibration.quantile,
            )
            update = {
                **update_context,
                "source": "recent_scores",
                "old_model_fingerprint": current_model_fingerprint,
                "new_model_fingerprint": current_model_fingerprint,
                "old_score_iqr": current_score_iqr,
                "old_threshold": current_threshold,
                "promoted": candidate is not None,
            }
            if candidate is None:
                update["reason"] = "score IQR is not positive"
            else:
                current_threshold, current_score_iqr = candidate
                update.update(
                    {
                        "new_score_iqr": current_score_iqr,
                        "new_threshold": current_threshold,
                    }
                )
            threshold_updates.append(update)

    output_path = Path(config.outputs.events_path)
    _save_json(output_path, results, "scoring results")

    anomaly_count = sum(result["is_anomaly"] is True for result in results)
    logger.info(
        "Saved %d %s development-test scores (%d anomalies) to %s",
        len(results),
        mode,
        anomaly_count,
        output_path,
    )
    summary = {
        "partition": selection.name,
        "capture_ids": list(selection.capture_ids),
        "window_count": len(results),
        "anomaly_count": anomaly_count,
        "score_type": score_type,
        "events_path": str(output_path),
    }
    report = {
        "report_type": "adaptation",
        "mode": mode,
        "partition": selection.name,
        "capture_ids": list(selection.capture_ids),
        "window_count": len(results),
        "anomaly_count": anomaly_count,
        "initial_model_fingerprint": model_fingerprint,
        "final_model_fingerprint": current_model_fingerprint,
        "initial_alpha": initial_alpha.tolist(),
        "final_alpha": alpha.tolist(),
        "initial_threshold": initial_threshold,
        "final_threshold": current_threshold,
        "initial_score_iqr": initial_score_iqr,
        "final_score_iqr": current_score_iqr,
        "quantile": config.calibration.quantile,
        "score_type": score_type,
        "buffer_size": config.adaptation.buffer_size,
        "update_every": config.adaptation.update_every,
        "minimum_refit_windows": config.adaptation.minimum_refit_windows,
        "safe_margin": config.adaptation.safe_margin,
        "safe_window_count": len(safe_counts),
        "threshold_update_count": sum(
            update["promoted"] is True for update in threshold_updates
        ),
        "threshold_update_attempts": threshold_updates,
        "model_refit_count": sum(
            attempt["promoted"] is True for attempt in model_refit_attempts
        ),
        "model_refit_attempts": model_refit_attempts,
        "events_path": str(output_path),
    }
    return summary, report


def score_windows(config: AppConfig) -> dict[str, object]:
    """Run the unchanged static detector over development-test windows."""
    summary, _report = _run_detector(config, "static")
    return summary


def adapt_windows(config: AppConfig) -> dict[str, object]:
    """Run the configured detector mode and save its state-change report."""
    summary, report = _run_detector(config, config.adaptation.mode)
    report_path = config.outputs.reports_dir / "adaptation.json"
    _save_json(report_path, report, "adaptation report")
    return {
        **summary,
        "adaptation_mode": config.adaptation.mode,
        "threshold_update_count": report["threshold_update_count"],
        "model_refit_count": report["model_refit_count"],
        "adaptation_report_path": str(report_path),
    }
