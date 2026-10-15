"""Score development-test windows with a calibrated model and threshold."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
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
from lm_idnet.config import AdaptationConfig, AppConfig
from lm_idnet.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
    LMIDNetError,
)
from lm_idnet.models.calibration_stage import load_calibration_packet_count_limit
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
    scores: Sequence[float],
    quantile: float,
) -> tuple[float, float]:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise DataValidationError("candidate scores must be non-empty and finite")
    quartile_1, quartile_3 = np.quantile(
        values, (0.25, 0.75), method="linear"
    )
    score_iqr = float(quartile_3 - quartile_1)
    if not np.isfinite(score_iqr) or score_iqr <= 0:
        raise DataValidationError("candidate score IQR must be finite and positive")
    threshold = float(np.quantile(values, quantile, method="linear"))
    if not np.isfinite(threshold):
        raise DataValidationError("candidate threshold must be finite")
    return threshold, score_iqr


def _validate_model_update(
    candidate_alpha: np.ndarray,
    active_alpha: np.ndarray,
    settings: AdaptationConfig,
) -> None:
    """Validate parameters and enforce movement limits for conservative updates."""
    if (
        candidate_alpha.shape != active_alpha.shape
        or not np.isfinite(candidate_alpha).all()
        or (candidate_alpha <= 0).any()
    ):
        raise DataValidationError(
            "candidate alpha must be finite, positive, and match categories"
        )
    concentration = float(candidate_alpha.sum())
    if not np.isfinite(concentration):
        raise DataValidationError("candidate concentration must be finite")
    if settings.policy == "legacy":
        return
    active_concentration = float(active_alpha.sum())
    ratio = concentration / active_concentration
    if not (
        1 / settings.max_concentration_ratio
        <= ratio
        <= settings.max_concentration_ratio
    ):
        raise DataValidationError(
            "candidate concentration change exceeds configured limit"
        )
    category_shift = np.max(
        np.abs(candidate_alpha / concentration - active_alpha / active_concentration)
    )
    if category_shift > settings.max_category_shift:
        raise DataValidationError(
            "candidate category proportion change exceeds configured limit"
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
    conservative = config.adaptation.policy == "conservative"
    volume_limit = None
    if mode != "static" and conservative:
        volume_limit = load_calibration_packet_count_limit(config, model_categories)
    selection = development_partition_for_evaluation(config)
    processed_dir = config.ingest.processed_root / config.ingest.dataset_folder
    results: list[dict[str, object]] = []
    safe_counts: deque[np.ndarray] = deque(maxlen=config.adaptation.buffer_size)
    recent_scores: deque[float] | None = (
        deque(maxlen=config.adaptation.buffer_size) if not conservative else None
    )
    recent_counts: deque[np.ndarray] | None = (
        deque(maxlen=config.adaptation.buffer_size) if not conservative else None
    )
    rejected_window_counts = {"excessive_volume": 0, "low_confidence": 0}
    threshold_updates: list[dict[str, object]] = []
    model_refit_attempts: list[dict[str, object]] = []
    current_model_fingerprint = model_fingerprint
    estimator = (
        create_estimator(
            config.estimator.model_copy(
                update={"log_likelihood_backend": model.log_likelihood_backend}
            ),
            model.precision_digits,
        )
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
            if not conservative:
                assert recent_scores is not None and recent_counts is not None
                recent_scores.append(score)
                recent_counts.append(counts)
            if volume_limit is not None and window.total_count > volume_limit:
                rejected_window_counts["excessive_volume"] += 1
            elif conservative or mode == "periodic_refit":
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
                if (
                    (
                        not conservative or score >= (
                            current_threshold
                            + config.adaptation.safe_margin * current_score_iqr
                        )
                    )
                    and baseline_score >= (
                        initial_threshold
                        + config.adaptation.safe_margin * initial_score_iqr
                    )
                ):
                    safe_counts.append(counts)
                else:
                    rejected_window_counts["low_confidence"] += 1

            if len(results) % config.adaptation.update_every != 0:
                continue
            if recent_scores is not None and len(recent_scores) < config.calibration.minimum_samples:
                continue

            update_context = {
                "after_window": len(results),
                "buffer_window_count": len(safe_counts) if conservative else len(recent_scores),
                "capture_id": capture_id,
                "window_end_utc": window.end_utc.isoformat().replace(
                    "+00:00", "Z"
                ),
            }
            update: dict[str, object] = {
                **update_context,
                "source": (
                    "periodic_refit" if mode == "periodic_refit"
                    else ("accepted_windows" if conservative else "recent_scores")
                ),
                "old_model_fingerprint": current_model_fingerprint,
                "old_score_iqr": current_score_iqr,
                "old_threshold": current_threshold,
                "promoted": False,
            }
            refit = None
            if mode == "periodic_refit":
                refit = {
                    **update_context,
                    "safe_window_count": len(safe_counts),
                    "old_model_fingerprint": current_model_fingerprint,
                    "old_alpha": alpha.tolist(),
                    "promoted": False,
                }
            try:
                if mode == "periodic_refit":
                    minimum_windows = (
                        max(
                            config.calibration.minimum_samples,
                            config.adaptation.minimum_refit_windows,
                        )
                        if conservative else config.adaptation.minimum_refit_windows
                    )
                    if len(safe_counts) < minimum_windows:
                        raise DataValidationError(
                            "not enough accepted windows" if conservative
                            else "not enough safe windows"
                        )
                elif conservative and len(safe_counts) < config.calibration.minimum_samples:
                    raise DataValidationError("not enough accepted windows")

                candidate_alpha = alpha
                if mode == "periodic_refit":
                    assert refit is not None and estimator is not None
                    safe_matrix = np.asarray(safe_counts, dtype=np.int64)
                    # One count per category keeps sparse accepted buffers fit-able.
                    refit_matrix = np.vstack(
                        (
                            np.eye(len(model_categories), dtype=np.int64),
                            safe_matrix,
                        )
                    )
                    refit["smoothing_window_count"] = len(model_categories)
                    active_log_likelihood = log_likelihood.calculate(
                        refit_matrix, alpha
                    )
                    fit = estimator.fit(refit_matrix)
                    refit.update(
                        {
                            "candidate_iterations": fit.iterations,
                            "candidate_converged": fit.converged,
                        }
                    )
                    if not fit.converged:
                        raise DataValidationError("candidate did not converge")
                    candidate_alpha = np.asarray(fit.alpha, dtype=np.float64)
                    _validate_model_update(candidate_alpha, alpha, config.adaptation)
                    if (
                        not np.isfinite(active_log_likelihood)
                        or not np.isfinite(fit.final_log_likelihood)
                    ):
                        raise DataValidationError("candidate likelihood must be finite")
                    refit.update(
                        {
                            "active_log_likelihood": active_log_likelihood,
                            "candidate_log_likelihood": fit.final_log_likelihood,
                        }
                    )
                    # shortcut: compare on fitting data; add a holdout if required.
                    if fit.final_log_likelihood < (
                        active_log_likelihood - config.estimator.tolerance_delta
                    ):
                        raise DataValidationError(
                            "candidate likelihood is worse than active model"
                        )

                if not conservative and mode == "adaptive_threshold":
                    candidate_scores = list(recent_scores)
                else:
                    candidate_scores = [
                        anomaly_score(counts, candidate_alpha, log_likelihood, score_type)
                        for counts in (safe_counts if conservative else recent_counts)
                    ]
                new_threshold, new_score_iqr = _candidate_threshold(
                    candidate_scores, config.calibration.quantile,
                )
                if conservative and abs(new_threshold - current_threshold) > (
                    config.adaptation.max_threshold_shift_iqr * initial_score_iqr
                ):
                    raise DataValidationError(
                        "candidate threshold change exceeds configured limit"
                    )
                new_model_fingerprint = model_parameter_fingerprint(
                    alpha=tuple(float(value) for value in candidate_alpha),
                    categories=model_categories,
                    log_likelihood_backend=model.log_likelihood_backend,
                    precision_digits=model.precision_digits,
                )
            except (LMIDNetError, ValueError, FloatingPointError) as error:
                update["reason"] = str(error)
                if refit is not None:
                    refit["reason"] = str(error)
                    if not conservative:
                        # Historical refit failures still update from all recent scores.
                        update["source"] = "recent_scores"
                        try:
                            new_threshold, new_score_iqr = _candidate_threshold(
                                recent_scores, config.calibration.quantile,
                            )
                        except (LMIDNetError, ValueError, FloatingPointError) as fallback_error:
                            update["reason"] = str(fallback_error)
                        else:
                            update.pop("reason", None)
                            update.update(
                                {
                                    "new_model_fingerprint": current_model_fingerprint,
                                    "new_threshold": new_threshold,
                                    "new_score_iqr": new_score_iqr,
                                    "promoted": True,
                                }
                            )
                            current_threshold, current_score_iqr = new_threshold, new_score_iqr
            else:
                update.update(
                    {
                        "new_model_fingerprint": new_model_fingerprint,
                        "new_threshold": new_threshold,
                        "new_score_iqr": new_score_iqr,
                        "promoted": True,
                    }
                )
                if refit is not None:
                    refit.update(
                        {
                            "new_model_fingerprint": new_model_fingerprint,
                            "new_alpha": candidate_alpha.tolist(),
                            "promoted": True,
                        }
                    )
                alpha = candidate_alpha.copy()
                expected_profile = tuple(float(value) for value in alpha / alpha.sum())
                current_model_fingerprint = new_model_fingerprint
                current_threshold, current_score_iqr = new_threshold, new_score_iqr
                if not conservative:
                    recent_scores = deque(candidate_scores, maxlen=config.adaptation.buffer_size)
            threshold_updates.append(update)
            if refit is not None:
                model_refit_attempts.append(refit)

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
        "policy": config.adaptation.policy,
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
        "volume_quantile": config.adaptation.volume_quantile if conservative else None,
        "packet_count_limit": volume_limit,
        "max_concentration_ratio": config.adaptation.max_concentration_ratio if conservative else None,
        "max_category_shift": config.adaptation.max_category_shift if conservative else None,
        "max_threshold_shift_iqr": config.adaptation.max_threshold_shift_iqr if conservative else None,
        "rejected_learning_windows": rejected_window_counts,
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
        "adaptation_policy": config.adaptation.policy,
        "threshold_update_count": report["threshold_update_count"],
        "model_refit_count": report["model_refit_count"],
        "adaptation_report_path": str(report_path),
    }
