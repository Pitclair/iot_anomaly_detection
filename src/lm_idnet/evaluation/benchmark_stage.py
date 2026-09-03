"""Measure LM and SciPy on the same production workload."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Callable

import numpy as np
import scipy

from lm_idnet.algorithms.dirichlet_multinomial import anomaly_score
from lm_idnet.algorithms.estimator_factory import create_estimator
from lm_idnet.algorithms.log_likelihood import (
    LogLikelihoodBackend,
    initialize_log_likelihood,
)
from lm_idnet.config import AppConfig
from lm_idnet.exceptions import DataValidationError
from lm_idnet.models.calibration_stage import calibrate_threshold
from lm_idnet.models.modeling_stage import load_training_matrix, train_model
from lm_idnet.models.schemas import AnomalyEventArtifact
from lm_idnet.models.scoring_stage import score_windows

_BACKENDS: tuple[LogLikelihoodBackend, ...] = ("scipy", "lm")
_MODEL_FIT_MEASUREMENTS = 5
_MEASUREMENT_BATCHES = 5
_LIKELIHOOD_CALLS = 200
_SCORE_CALLS = 1000


def _median_call_seconds(operation: Callable[[], object], calls: int) -> float:
    operation()
    samples = []
    for _ in range(_MEASUREMENT_BATCHES):
        started = perf_counter()
        for _ in range(calls):
            operation()
        samples.append((perf_counter() - started) / calls)
    return median(samples)


def _load_events(path: Path) -> list[AnomalyEventArtifact]:
    try:
        raw_events = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_events, list):
            raise ValueError("event artifact must be a JSON array")
        return [AnomalyEventArtifact.model_validate(event) for event in raw_events]
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise DataValidationError(f"cannot read benchmark events: {path}") from error


def benchmark_backends(config: AppConfig) -> dict[str, object]:
    """Compare both likelihood backends without replacing accepted artifacts."""
    training = load_training_matrix(config)
    run_results: dict[str, dict[str, object]] = {}

    with TemporaryDirectory(prefix="lm-idnet-benchmark-") as temporary:
        temporary_root = Path(temporary)
        for backend_name in _BACKENDS:
            run_root = temporary_root / backend_name
            run_config = config.model_copy(
                update={
                    "estimator": config.estimator.model_copy(
                        update={"log_likelihood_backend": backend_name}
                    ),
                    "outputs": config.outputs.model_copy(
                        update={
                            "model_path": run_root / "model.json",
                            "threshold_path": run_root / "threshold.json",
                            "events_path": run_root / "events.json",
                            "reports_dir": run_root / "reports",
                        }
                    ),
                }
            )
            training_result = train_model(run_config)
            calibration_result = calibrate_threshold(run_config)
            scoring_result = score_windows(run_config)
            run_results[backend_name] = {
                "training": training_result,
                "calibration": calibration_result,
                "scoring": scoring_result,
                "events": _load_events(run_config.outputs.events_path),
            }

        scipy_events = run_results["scipy"]["events"]
        lm_events = run_results["lm"]["events"]
        assert isinstance(scipy_events, list)
        assert isinstance(lm_events, list)
        if not scipy_events or len(scipy_events) != len(lm_events):
            raise DataValidationError(
                "backend benchmark requires matching non-empty scoring results"
            )
        event_identity = lambda event: (
            event.capture_id,
            event.window_start_utc,
            event.window_end_utc,
        )
        if [event_identity(event) for event in scipy_events] != [
            event_identity(event) for event in lm_events
        ]:
            raise DataValidationError("backend benchmark scored different windows")

        scipy_training = run_results["scipy"]["training"]
        lm_training = run_results["lm"]["training"]
        scipy_calibration = run_results["scipy"]["calibration"]
        lm_calibration = run_results["lm"]["calibration"]
        assert isinstance(scipy_training, dict)
        assert isinstance(lm_training, dict)
        assert isinstance(scipy_calibration, dict)
        assert isinstance(lm_calibration, dict)

        reference_alpha = np.asarray(scipy_training["alpha"], dtype=np.float64)
        representative_event = next(
            (
                event
                for event in scipy_events
                if sum(event.counts.values()) > 0
            ),
            scipy_events[0],
        )
        representative_counts = np.asarray(
            [
                representative_event.counts[category]
                for category in config.ingest.categories
            ],
            dtype=np.int64,
        )
        backend_metrics: dict[str, dict[str, object]] = {}
        likelihood_values: dict[str, float] = {}
        for backend_name in _BACKENDS:
            backend = initialize_log_likelihood(
                backend_name,
                config.precision_digits,
            )
            likelihood_operation = lambda: backend.calculate(
                training.counts,
                reference_alpha,
            )
            scoring_operation = lambda: anomaly_score(
                representative_counts,
                reference_alpha,
                backend,
                config.calibration.score_type,
            )
            likelihood_values[backend_name] = likelihood_operation()
            training_result = run_results[backend_name]["training"]
            calibration_result = run_results[backend_name]["calibration"]
            scoring_result = run_results[backend_name]["scoring"]
            assert isinstance(training_result, dict)
            assert isinstance(calibration_result, dict)
            assert isinstance(scoring_result, dict)
            model_fit_samples = [float(training_result["duration_seconds"])]
            estimator_config = config.estimator.model_copy(
                update={"log_likelihood_backend": backend_name}
            )
            for _ in range(_MODEL_FIT_MEASUREMENTS - 1):
                estimator = create_estimator(estimator_config, config.precision_digits)
                started = perf_counter()
                estimator.fit(training.counts)
                model_fit_samples.append(perf_counter() - started)
            backend_metrics[backend_name] = {
                "model_fit_median_seconds": median(model_fit_samples),
                "model_fit_samples_seconds": model_fit_samples,
                "model_fit_iterations": training_result["iterations"],
                "likelihood_matrix_median_seconds": _median_call_seconds(
                    likelihood_operation,
                    _LIKELIHOOD_CALLS,
                ),
                "single_window_score_median_seconds": _median_call_seconds(
                    scoring_operation,
                    _SCORE_CALLS,
                ),
                "initial_log_likelihood": training_result[
                    "initial_log_likelihood"
                ],
                "final_log_likelihood": training_result["final_log_likelihood"],
                "threshold": calibration_result["threshold"],
                "anomaly_count": scoring_result["anomaly_count"],
            }

        scipy_alpha = np.asarray(scipy_training["alpha"], dtype=np.float64)
        lm_alpha = np.asarray(lm_training["alpha"], dtype=np.float64)
        score_differences = np.abs(
            np.asarray([event.score for event in lm_events])
            - np.asarray([event.score for event in scipy_events])
        )
        comparisons = {
            "alpha_maximum_absolute_difference": float(
                np.max(np.abs(lm_alpha - scipy_alpha))
            ),
            "alpha_maximum_relative_difference": float(
                np.max(np.abs((lm_alpha - scipy_alpha) / scipy_alpha))
            ),
            "likelihood_absolute_difference": abs(
                likelihood_values["lm"] - likelihood_values["scipy"]
            ),
            "threshold_absolute_difference": abs(
                float(lm_calibration["threshold"])
                - float(scipy_calibration["threshold"])
            ),
            "score_maximum_absolute_difference": float(score_differences.max()),
            "score_median_absolute_difference": float(median(score_differences)),
            "decision_mismatch_count": sum(
                lm_event.is_anomaly != scipy_event.is_anomaly
                for lm_event, scipy_event in zip(lm_events, scipy_events)
            ),
            "lm_over_scipy_runtime_ratio": {
                metric: float(backend_metrics["lm"][metric])
                / float(backend_metrics["scipy"][metric])
                for metric in (
                    "model_fit_median_seconds",
                    "likelihood_matrix_median_seconds",
                    "single_window_score_median_seconds",
                )
            },
        }
        report = {
            "report_type": "lm_backend_benchmark",
            "dataset": config.ingest.dataset_folder,
            "precision_digits": config.precision_digits,
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "method": {
                "model_fit_measurements": _MODEL_FIT_MEASUREMENTS,
                "timing_batches": _MEASUREMENT_BATCHES,
                "likelihood_calls_per_batch": _LIKELIHOOD_CALLS,
                "score_calls_per_batch": _SCORE_CALLS,
                "model_fit_scope": "estimator fit only; excludes data loading and saving",
            },
            "workload": {
                "training_matrix_shape": list(training.counts.shape),
                "representative_window_counts": representative_counts.tolist(),
                "scored_window_count": len(scipy_events),
            },
            "backends": backend_metrics,
            "comparisons": comparisons,
        }

    output_path = config.outputs.reports_dir / "lm_backend_benchmark.json"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, allow_nan=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as error:
        raise DataValidationError(
            f"cannot save backend benchmark report: {output_path}"
        ) from error
    return {**report, "report_path": str(output_path)}
