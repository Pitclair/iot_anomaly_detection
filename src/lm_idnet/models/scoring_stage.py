"""Score development-test windows with a calibrated model and threshold."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from contextlib import nullcontext
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
    detector_state_lock,
    load_detector_state,
    load_model_threshold,
    model_parameter_fingerprint,
    record_detector_update,
    save_detector_state,
)
from lm_idnet.config import AdaptationConfig, AppConfig, EdgeConfig
from lm_idnet.exceptions import (
    DataValidationError,
    DetectorStateCommitError,
    LMIDNetError,
)
from lm_idnet.models.calibration_stage import load_calibration_packet_count_limit
from lm_idnet.models.schemas import AnomalyEventArtifact, DetectorStateArtifact
from lm_idnet.partitioning import development_partition_for_evaluation
from lm_idnet.processing.schemas import WindowRecord, validate_window_contract
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


def _save_detector_update(
    state_dir: str | Path | None,
    state: DetectorStateArtifact | None,
    *,
    alpha: np.ndarray,
    threshold: float,
    score_iqr: float,
    model_fingerprint: str,
    source: str,
) -> DetectorStateArtifact | None:
    if state is None:
        return None
    assert state_dir is not None
    raw = state.model_dump(exclude={"version_id", "created_utc"})
    raw.update(
        source=source, alpha=alpha.tolist(),
        threshold={
            "model_fingerprint": model_fingerprint,
            "threshold": threshold, "score_iqr": score_iqr,
        },
    )
    candidate = DetectorStateArtifact.model_validate(raw)
    save_detector_state(state_dir, candidate)
    return candidate


class WindowDetector:
    """Shared one-window detector; callers own the state lock and output history."""

    def __init__(
        self, config: AppConfig | EdgeConfig, *, mode: str | None = None,
        state_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.mode = mode or config.adaptation.mode
        self.state_dir = state_dir
        if isinstance(config, EdgeConfig) and state_dir is None:
            self.state_dir = config.state_dir
        self.state = (
            load_detector_state(self.state_dir, config) if self.state_dir is not None else None
        )
        self.starting_state = self.state
        if self.state is None:
            self.model, threshold = load_model_threshold(config)
        else:
            self.model, threshold = self.state.baseline_model, self.state.baseline_threshold
        self.model_fingerprint = artifact_fingerprint(self.model)
        self.score_type = config.calibration.score_type
        self.window_count = 0
        self.log_likelihood = initialize_log_likelihood(
            self.model.log_likelihood_backend,
            self.model.precision_digits,
        )
        self.initial_alpha = np.asarray(self.model.alpha, dtype=np.float64)
        self.alpha = (
            np.asarray(self.state.alpha, dtype=np.float64)
            if self.state is not None else self.initial_alpha.copy()
        )
        self.model_categories = self.model.categories
        self.expected_profile = (
            tuple(float(value) for value in self.alpha / self.alpha.sum())
            if self.state is not None else self.model.mean_probabilities
        )
        self.initial_threshold = threshold.threshold
        self.current_threshold = (
            self.state.threshold.threshold if self.state is not None else self.initial_threshold
        )
        self.initial_score_iqr = threshold.score_iqr
        self.current_score_iqr = (
            self.state.threshold.score_iqr if self.state is not None else self.initial_score_iqr
        )
        self.conservative = self.config.adaptation.policy == "conservative"
        self.volume_limit = self.state.packet_count_limit if self.state is not None else None
        if self.state is None and self.mode != "static" and self.conservative:
            self.volume_limit = load_calibration_packet_count_limit(self.config, self.model_categories)
        self.safe_counts: deque[np.ndarray] = deque(maxlen=self.config.adaptation.buffer_size)
        self.recent_scores: deque[float] | None = (
            deque(maxlen=self.config.adaptation.buffer_size) if not self.conservative else None
        )
        self.recent_counts: deque[np.ndarray] | None = (
            deque(maxlen=self.config.adaptation.buffer_size) if not self.conservative else None
        )
        self.rejected_window_counts = {"excessive_volume": 0, "low_confidence": 0}
        self.current_model_fingerprint = (
            self.state.threshold.model_fingerprint
            if self.state is not None else self.model_fingerprint
        )
        self.estimator = (
            create_estimator(
                self.config.estimator.model_copy(
                    update={"log_likelihood_backend": self.model.log_likelihood_backend}
                ),
                self.model.precision_digits,
            )
            if self.mode == "periodic_refit"
            else None
        )

    def process(
        self, window: WindowRecord, *, device_id: str, capture_id: str,
    ) -> tuple[dict[str, object] | None, dict[str, object] | None, dict[str, object] | None]:
        """Return this window's event, threshold attempt, and refit attempt only."""
        if device_id != self.model.device_id:
            raise DataValidationError("scoring device does not match model")
        validate_window_contract(
            window, self.model_categories, self.model.input_contract.window_minutes,
        )
        if window.state == "missing":
            return None, None, None
        event = score_window(
            window, device_id=device_id, capture_id=capture_id,
            model_categories=self.model_categories, expected_profile=self.expected_profile,
            model_fingerprint=self.current_model_fingerprint, alpha=self.alpha,
            log_likelihood=self.log_likelihood, threshold=self.current_threshold,
            score_iqr=self.current_score_iqr, score_type=self.score_type,
        )
        self.window_count += 1
        if self.mode == "static":
            return event, None, None

        score = float(event["score"])
        counts = np.asarray(window.counts, dtype=np.int64)
        if not self.conservative:
            assert self.recent_scores is not None and self.recent_counts is not None
            self.recent_scores.append(score)
            self.recent_counts.append(counts)
        if self.volume_limit is not None and window.total_count > self.volume_limit:
            self.rejected_window_counts["excessive_volume"] += 1
        elif self.conservative or self.mode == "periodic_refit":
            baseline_score = (
                score
                if self.current_model_fingerprint == self.model_fingerprint
                else anomaly_score(
                    counts,
                    self.initial_alpha,
                    self.log_likelihood,
                    self.score_type,
                )
            )
            if (
                (
                    not self.conservative or score >= (
                        self.current_threshold
                        + self.config.adaptation.safe_margin * self.current_score_iqr
                    )
                )
                and baseline_score >= (
                    self.initial_threshold
                    + self.config.adaptation.safe_margin * self.initial_score_iqr
                )
            ):
                self.safe_counts.append(counts)
            else:
                self.rejected_window_counts["low_confidence"] += 1

        if self.window_count % self.config.adaptation.update_every != 0:
            return event, None, None
        if self.recent_scores is not None and len(self.recent_scores) < self.config.calibration.minimum_samples:
            return event, None, None

        update_context = {
            "after_window": self.window_count,
            "buffer_window_count": len(self.safe_counts) if self.conservative else len(self.recent_scores),
            "capture_id": capture_id,
            "window_end_utc": window.end_utc.isoformat().replace(
                "+00:00", "Z"
            ),
        }
        update: dict[str, object] = {
            **update_context,
            "source": (
                "periodic_refit" if self.mode == "periodic_refit"
                else ("accepted_windows" if self.conservative else "recent_scores")
            ),
            "old_model_fingerprint": self.current_model_fingerprint,
            "old_score_iqr": self.current_score_iqr,
            "old_threshold": self.current_threshold,
            "promoted": False,
        }
        refit = None
        previous_state = self.state
        if self.mode == "periodic_refit":
            refit = {
                **update_context,
                "safe_window_count": len(self.safe_counts),
                "old_model_fingerprint": self.current_model_fingerprint,
                "old_alpha": self.alpha.tolist(),
                "promoted": False,
            }
        try:
            if self.mode == "periodic_refit":
                minimum_windows = (
                    max(
                        self.config.calibration.minimum_samples,
                        self.config.adaptation.minimum_refit_windows,
                    )
                    if self.conservative else self.config.adaptation.minimum_refit_windows
                )
                if len(self.safe_counts) < minimum_windows:
                    raise DataValidationError(
                        "not enough accepted windows" if self.conservative
                        else "not enough safe windows"
                    )
            elif self.conservative and len(self.safe_counts) < self.config.calibration.minimum_samples:
                raise DataValidationError("not enough accepted windows")

            candidate_alpha = self.alpha
            if self.mode == "periodic_refit":
                assert refit is not None and self.estimator is not None
                safe_matrix = np.asarray(self.safe_counts, dtype=np.int64)
                # One count per category keeps sparse accepted buffers fit-able.
                refit_matrix = np.vstack(
                    (
                        np.eye(len(self.model_categories), dtype=np.int64),
                        safe_matrix,
                    )
                )
                refit["smoothing_window_count"] = len(self.model_categories)
                active_log_likelihood = self.log_likelihood.calculate(
                    refit_matrix, self.alpha
                )
                fit = self.estimator.fit(refit_matrix)
                refit.update(
                    {
                        "candidate_iterations": fit.iterations,
                        "candidate_converged": fit.converged,
                    }
                )
                if not fit.converged:
                    raise DataValidationError("candidate did not converge")
                candidate_alpha = np.asarray(fit.alpha, dtype=np.float64)
                _validate_model_update(candidate_alpha, self.alpha, self.config.adaptation)
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
                    active_log_likelihood - self.config.estimator.tolerance_delta
                ):
                    raise DataValidationError(
                        "candidate likelihood is worse than active model"
                    )

            if not self.conservative and self.mode == "adaptive_threshold":
                candidate_scores = list(self.recent_scores)
            else:
                candidate_scores = [
                    anomaly_score(counts, candidate_alpha, self.log_likelihood, self.score_type)
                    for counts in (self.safe_counts if self.conservative else self.recent_counts)
                ]
            new_threshold, new_score_iqr = _candidate_threshold(
                candidate_scores, self.config.calibration.quantile,
            )
            if self.conservative and abs(new_threshold - self.current_threshold) > (
                self.config.adaptation.max_threshold_shift_iqr * self.initial_score_iqr
            ):
                raise DataValidationError(
                    "candidate threshold change exceeds configured limit"
                )
            new_model_fingerprint = model_parameter_fingerprint(
                alpha=tuple(float(value) for value in candidate_alpha),
                categories=self.model_categories,
                log_likelihood_backend=self.model.log_likelihood_backend,
                precision_digits=self.model.precision_digits,
                **self.model.input_contract.model_dump(exclude={"categories"}),
            )
        except (LMIDNetError, ValueError, FloatingPointError) as error:
            update["reason"] = str(error)
            if refit is not None:
                refit["reason"] = str(error)
                if not self.conservative:
                    # Historical refit failures still update from all recent scores.
                    update["source"] = "recent_scores"
                    try:
                        new_threshold, new_score_iqr = _candidate_threshold(
                            self.recent_scores, self.config.calibration.quantile,
                        )
                        self.state = _save_detector_update(
                            self.state_dir, self.state, alpha=self.alpha, threshold=new_threshold,
                            score_iqr=new_score_iqr, model_fingerprint=self.current_model_fingerprint,
                            source="adaptive_threshold",
                        )
                    except DetectorStateCommitError:
                        raise
                    except (LMIDNetError, ValueError, FloatingPointError) as fallback_error:
                        update["reason"] = str(fallback_error)
                    else:
                        update.pop("reason", None)
                        update.update(
                            {
                                "new_model_fingerprint": self.current_model_fingerprint,
                                "new_threshold": new_threshold,
                                "new_score_iqr": new_score_iqr,
                                "promoted": True,
                            }
                        )
                        self.current_threshold, self.current_score_iqr = new_threshold, new_score_iqr
        else:
            try:
                self.state = _save_detector_update(
                    self.state_dir, self.state, alpha=candidate_alpha, threshold=new_threshold,
                    score_iqr=new_score_iqr, model_fingerprint=new_model_fingerprint,
                    source=self.mode,
                )
            except DetectorStateCommitError:
                raise
            except (LMIDNetError, ValueError, FloatingPointError) as error:
                update["reason"] = str(error)
                if refit is not None:
                    refit["reason"] = str(error)
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
                self.alpha = candidate_alpha.copy()
                self.expected_profile = tuple(float(value) for value in self.alpha / self.alpha.sum())
                self.current_model_fingerprint = new_model_fingerprint
                self.current_threshold, self.current_score_iqr = new_threshold, new_score_iqr
                if not self.conservative:
                    self.recent_scores = deque(candidate_scores, maxlen=self.config.adaptation.buffer_size)
        if self.state is not None:
            assert previous_state is not None and self.state_dir is not None
            if refit is not None and not refit["promoted"]:
                record_detector_update(
                    self.state_dir, source="periodic_refit", old_version=previous_state.version_id,
                    promoted=False, reason=str(refit["reason"]), after_window=self.window_count,
                )
            if update["promoted"] or refit is None or update.get("reason") != refit.get("reason"):
                record_detector_update(
                    self.state_dir,
                    source="adaptive_threshold" if update["source"] == "recent_scores" else self.mode,
                    old_version=previous_state.version_id,
                    new_version=self.state.version_id if update["promoted"] else None,
                    promoted=bool(update["promoted"]), reason=update.get("reason"),
                    after_window=self.window_count,
                )
        return event, update, refit


def _run_detector(
    config: AppConfig, mode: str, *, state_dir: str | Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Collect batch reports around the same detector used by edge inference."""
    detector = WindowDetector(config, mode=mode, state_dir=state_dir)
    selection = development_partition_for_evaluation(config)
    processed_dir = config.ingest.processed_root / config.ingest.dataset_folder
    results: list[dict[str, object]] = []
    threshold_updates: list[dict[str, object]] = []
    model_refit_attempts: list[dict[str, object]] = []
    for capture_id in selection.capture_ids:
        dataset = load_processed_dataset(processed_dir / f"{capture_id}.json")
        if dataset.metadata.capture_id != capture_id:
            raise DataValidationError(f"processed capture ID does not match filename: {capture_id}")
        if dataset.metadata.partition != selection.name:
            raise DataValidationError(f"development-test capture has the wrong partition: {capture_id}")
        if dataset.metadata.device_id != config.ingest.device_id:
            raise DataValidationError(f"scoring device does not match configuration: {capture_id}")
        for window in dataset.windows:
            event, update, refit = detector.process(
                window, device_id=dataset.metadata.device_id, capture_id=capture_id,
            )
            if event is not None:
                results.append(event)
            if update is not None:
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
        "score_type": detector.score_type,
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
        "initial_model_fingerprint": detector.model_fingerprint,
        "final_model_fingerprint": detector.current_model_fingerprint,
        "initial_alpha": detector.initial_alpha.tolist(),
        "final_alpha": detector.alpha.tolist(),
        "initial_threshold": detector.initial_threshold,
        "final_threshold": detector.current_threshold,
        "initial_score_iqr": detector.initial_score_iqr,
        "final_score_iqr": detector.current_score_iqr,
        "quantile": config.calibration.quantile,
        "score_type": detector.score_type,
        "buffer_size": config.adaptation.buffer_size,
        "update_every": config.adaptation.update_every,
        "minimum_refit_windows": config.adaptation.minimum_refit_windows,
        "safe_margin": config.adaptation.safe_margin,
        "volume_quantile": config.adaptation.volume_quantile if detector.conservative else None,
        "packet_count_limit": detector.volume_limit,
        "max_concentration_ratio": config.adaptation.max_concentration_ratio if detector.conservative else None,
        "max_category_shift": config.adaptation.max_category_shift if detector.conservative else None,
        "max_threshold_shift_iqr": config.adaptation.max_threshold_shift_iqr if detector.conservative else None,
        "rejected_learning_windows": detector.rejected_window_counts,
        "safe_window_count": len(detector.safe_counts),
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
    if detector.state is not None:
        assert detector.starting_state is not None
        state_summary = {
            "state_dir": str(detector.state_dir),
            "starting_state_version": detector.starting_state.version_id,
            "final_state_version": detector.state.version_id,
        }
        summary.update(state_summary)
        report.update(state_summary)
        report.update(
            starting_alpha=list(detector.starting_state.alpha),
            starting_threshold=detector.starting_state.threshold.threshold,
            starting_score_iqr=detector.starting_state.threshold.score_iqr,
        )
    return summary, report


def score_windows(config: AppConfig) -> dict[str, object]:
    """Run the unchanged static detector over development-test windows."""
    summary, _report = _run_detector(config, "static")
    return summary


def adapt_windows(config: AppConfig, *, state_dir: str | Path | None = None) -> dict[str, object]:
    """Run the configured detector mode and save its state-change report."""
    guard = detector_state_lock(state_dir) if state_dir is not None else nullcontext()
    with guard:
        summary, report = _run_detector(config, config.adaptation.mode, state_dir=state_dir)
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
