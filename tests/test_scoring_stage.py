"""Tests for development-window anomaly scoring."""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from lm_idnet.algorithms.dirichlet import DirichletFit
from lm_idnet.algorithms.log_likelihood import ScipyLogLikelihood
from lm_idnet.artifacts import (
    artifact_fingerprint, detector_state_lock, load_artifact,
    load_detector_state, load_detector_version,
)
from lm_idnet.config import AdaptationConfig
from lm_idnet.exceptions import ArtifactCompatibilityError, DataValidationError, DetectorStateCommitError
from lm_idnet.models.calibration_stage import save_threshold
from lm_idnet.models import scoring_stage
from lm_idnet.models.modeling_stage import TrainingMatrix, save_model
from lm_idnet.models.scoring_stage import adapt_windows, score_window, score_windows
from lm_idnet.processing.schemas import Metadata, ProcessedDataset, WindowRecord
from lm_idnet.processing.storage import save_processed_dataset

pytestmark = pytest.mark.unit


def test_score_window_is_pure_and_rejects_missing_windows(window_factory) -> None:
    window = window_factory(tcp=1, udp=2, ssdp=3, arp=4)
    arguments = {
        "device_id": "camera-001",
        "capture_id": "capture-001",
        "model_categories": window.categories,
        "expected_profile": (0.1, 0.2, 0.3, 0.4),
        "model_fingerprint": "0" * 64,
        "alpha": np.asarray([1.0, 2.0, 3.0, 4.0]),
        "log_likelihood": ScipyLogLikelihood(),
        "threshold": 0.0,
        "score_iqr": 2.0,
        "score_type": "raw",
    }

    result = score_window(window, **arguments)

    assert result["counts"] == dict(zip(window.categories, window.counts))
    assert result["is_anomaly"] == (result["score"] < result["threshold"])
    assert result["severity"] == max(0.0, -result["score"]) / 2.0
    assert result["category_residuals"] == pytest.approx(
        dict(zip(window.categories, (0.0, 0.0, 0.0, 0.0)))
    )

    missing = WindowRecord(
        start_utc=window.start_utc,
        end_utc=window.end_utc,
        categories=window.categories,
        counts=None,
        state="missing",
    )
    with pytest.raises(DataValidationError, match="missing window"):
        score_window(missing, **arguments)

    arguments["score_iqr"] = 0.0
    with pytest.raises(DataValidationError, match="IQR must be positive"):
        score_window(window, **arguments)


def write_model(
    config,
    log_likelihood_backend: str = "scipy",
    precision_digits: int | None = None,
) -> None:
    alpha = np.asarray([1.0, 2.0, 3.0, 4.0])
    start = datetime(2020, 10, 8, tzinfo=timezone.utc)
    model_config = config.model_copy(
        update={
            "precision_digits": precision_digits or config.precision_digits,
            "estimator": config.estimator.model_copy(
                update={"log_likelihood_backend": log_likelihood_backend}
            ),
        }
    )
    save_model(
        model_config,
        DirichletFit(alpha.copy(), alpha, 10.0, 0.1, 1, True, -10.0, -9.0),
        TrainingMatrix(
            counts=np.ones((1, len(config.ingest.categories)), dtype=np.int64),
            capture_ids=config.ingest.partitions.fit,
            start_utc=start,
            end_utc=start + timedelta(minutes=10),
            missing_window_count=0,
            silent_window_count=0,
        ),
        0.1,
    )


def write_threshold(
    config,
    score_type: str,
    model_fingerprint: str | None = None,
    threshold_value: float = 0.0,
    score_iqr: float = 1.0,
) -> None:
    start = datetime(2020, 10, 14, tzinfo=timezone.utc)
    model = load_artifact(config.outputs.model_path, expected_type="model")
    save_threshold(
        config.outputs.threshold_path,
        {
            "artifact_type": "threshold",
            "model_fingerprint": model_fingerprint or artifact_fingerprint(model),
            "score_type": score_type,
            "quantile": config.calibration.quantile,
            "threshold": threshold_value,
            "calibration_capture_ids": config.ingest.partitions.calibration,
            "calibration_window_count": config.calibration.minimum_samples,
            "calibration_start_utc": start,
            "calibration_end_utc": start + timedelta(minutes=10),
            "quantile_method": "linear",
            "score_minimum": -1.0,
            "score_median": -0.5,
            "score_maximum": 0.0,
            "score_iqr": score_iqr,
        },
    )


def write_processed_captures(
    config,
    partition: str = "development_test",
    counts: tuple = ((1, 2, 3, 4), (0, 0, 0, 0), None),
) -> None:
    output_dir = config.ingest.processed_root / config.ingest.dataset_folder
    for index, capture_id in enumerate(getattr(config.ingest.partitions, partition)):
        start = datetime(2020, 10, 16 + index, tzinfo=timezone.utc)
        windows = tuple(
            WindowRecord(
                start_utc=start + timedelta(minutes=offset * config.ingest.window_minutes),
                end_utc=start + timedelta(minutes=(offset + 1) * config.ingest.window_minutes),
                categories=config.ingest.categories,
                counts=values,
                state=(
                    "missing" if values is None
                    else "observed" if sum(values) else "observed-silent"
                ),
            )
            for offset, values in enumerate(counts)
        )
        save_processed_dataset(
            ProcessedDataset(
                metadata=Metadata(
                    device_id=config.ingest.device_id,
                    capture_id=capture_id,
                    partition=partition,
                    date=capture_id,
                    file_source=f"{capture_id}.pcap",
                ),
                windows=windows,
            ),
            output_dir / f"{capture_id}.json",
        )


@pytest.mark.parametrize("score_type", ["raw", "normalized"])
def test_score_windows_writes_observed_and_silent_results(
    tmp_path,
    config_factory,
    score_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = config_factory(
        precision_digits=8,
        ingest={"processed_root": tmp_path},
        estimator={"log_likelihood_backend": "scipy"},
        calibration={"score_type": score_type},
        outputs={
            "model_path": tmp_path / "model.json",
            "threshold_path": tmp_path / "threshold.json",
            "events_path": tmp_path / "events.json",
        },
    )
    write_model(config, log_likelihood_backend="lm", precision_digits=7)
    write_threshold(config, score_type)
    write_processed_captures(config)
    backend_calls = []

    def initialize_backend(backend_name, precision_digits=6):
        backend_calls.append((backend_name, precision_digits))
        return ScipyLogLikelihood()

    monkeypatch.setattr(
        scoring_stage,
        "initialize_log_likelihood",
        initialize_backend,
    )

    summary = score_windows(config)
    results = json.loads(config.outputs.events_path.read_text(encoding="utf-8"))

    assert summary["partition"] == "development_test"
    assert backend_calls == [("lm", 7)]
    assert summary["score_type"] == score_type
    assert summary["window_count"] == len(results) == 4
    assert summary["anomaly_count"] == 2
    assert all(
        set(result)
        == {
            "artifact_type",
            "device_id",
            "capture_id",
            "window_start_utc",
            "window_end_utc",
            "counts",
            "score_type",
            "score",
            "threshold",
            "severity",
            "is_anomaly",
            "expected_profile",
            "category_residuals",
            "model_fingerprint",
        }
        for result in results
    )
    assert {result["capture_id"] for result in results} == set(
        config.ingest.partitions.development_test
    )
    assert all(
        result["device_id"] == config.ingest.device_id for result in results
    )
    assert all(result["score_type"] == score_type for result in results)
    threshold = load_artifact(
        config.outputs.threshold_path, expected_type="threshold"
    )
    assert all(
        result["model_fingerprint"] == threshold.model_fingerprint
        for result in results
    )
    assert all(
        result["expected_profile"]
        == {"tcp": 0.1, "udp": 0.2, "ssdp": 0.3, "arp": 0.4}
        for result in results
    )
    assert all(
        set(result["counts"]) == set(config.ingest.categories)
        for result in results
    )
    assert all(
        result["is_anomaly"] == (result["score"] < result["threshold"])
        for result in results
    )
    assert all(
        result["severity"] == max(0.0, result["threshold"] - result["score"])
        for result in results
    )
    assert all(
        result["is_anomaly"] is False
        for result in results
        if sum(result["counts"].values()) == 0
    )
    silent = next(
        result for result in results if not sum(result["counts"].values())
    )
    assert silent["category_residuals"] == pytest.approx(
        {category: -value for category, value in silent["expected_profile"].items()}
    )


def test_score_windows_rejects_unexpected_score_type(
    tmp_path,
    config_factory,
) -> None:
    config = config_factory(
        outputs={
            "model_path": tmp_path / "model.json",
            "threshold_path": tmp_path / "threshold.json",
            "events_path": tmp_path / "events.json",
        }
    )
    write_model(config)
    write_threshold(config, "normalized")

    with pytest.raises(ArtifactCompatibilityError, match="score type"):
        score_windows(config)

    assert not config.outputs.events_path.exists()


def test_score_windows_rejects_threshold_for_another_model(
    tmp_path,
    config_factory,
) -> None:
    config = config_factory(
        outputs={
            "model_path": tmp_path / "model.json",
            "threshold_path": tmp_path / "threshold.json",
            "events_path": tmp_path / "events.json",
        }
    )
    write_model(config)
    write_threshold(config, "raw", "f" * 64)

    with pytest.raises(ArtifactCompatibilityError, match="model fingerprint"):
        score_windows(config)

    assert not config.outputs.events_path.exists()


def test_static_adaptation_preserves_static_scoring(
    tmp_path,
    config_factory,
) -> None:
    config = config_factory(
        ingest={"processed_root": tmp_path},
        estimator={"log_likelihood_backend": "scipy"},
        outputs={
            "model_path": tmp_path / "model.json",
            "threshold_path": tmp_path / "threshold.json",
            "events_path": tmp_path / "events.json",
            "reports_dir": tmp_path / "reports",
        },
    )
    write_model(config)
    write_threshold(config, "raw")
    write_processed_captures(config)

    score_windows(config)
    static_events = config.outputs.events_path.read_text(encoding="utf-8")
    result = adapt_windows(config)

    assert config.outputs.events_path.read_text(encoding="utf-8") == static_events
    assert result["adaptation_mode"] == "static"
    assert result["threshold_update_count"] == 0


def test_adaptive_threshold_updates_only_after_scoring_current_window(
    tmp_path,
    config_factory,
) -> None:
    config = config_factory(
        ingest={"processed_root": tmp_path},
        estimator={"log_likelihood_backend": "scipy"},
        calibration={"minimum_samples": 2, "quantile": 0.25},
        adaptation={
            "mode": "adaptive_threshold",
            "buffer_size": 2,
            "update_every": 2,
            "safe_margin": 0.0,
            "max_threshold_shift_iqr": 10.0,
        },
        outputs={
            "model_path": tmp_path / "model.json",
            "threshold_path": tmp_path / "threshold.json",
            "events_path": tmp_path / "events.json",
            "reports_dir": tmp_path / "reports",
        },
    )
    write_model(config)
    write_threshold(config, "raw", threshold_value=-10.0)
    write_processed_captures(config, "calibration")
    write_processed_captures(config)

    result = adapt_windows(config)
    events = json.loads(config.outputs.events_path.read_text(encoding="utf-8"))
    report = json.loads(
        (config.outputs.reports_dir / "adaptation.json").read_text(
            encoding="utf-8"
        )
    )

    assert [event["threshold"] for event in events[:2]] == [-10.0, -10.0]
    assert events[2]["threshold"] == pytest.approx(
        report["threshold_update_attempts"][0]["new_threshold"]
    )
    assert result["threshold_update_count"] == 1
    assert report["threshold_update_attempts"][0]["promoted"] is True
    assert report["threshold_update_attempts"][1]["promoted"] is False
    assert report["initial_model_fingerprint"] == events[0]["model_fingerprint"]


def test_periodic_refit_promotes_model_and_matching_threshold(
    tmp_path,
    config_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = config_factory(
        ingest={"processed_root": tmp_path},
        estimator={"log_likelihood_backend": "scipy"},
        calibration={"minimum_samples": 2, "quantile": 0.25},
        adaptation={
            "mode": "periodic_refit",
            "buffer_size": 2,
            "update_every": 2,
            "minimum_refit_windows": 2,
            "safe_margin": 0.0,
            "max_threshold_shift_iqr": 10.0,
        },
        outputs={
            "model_path": tmp_path / "model.json",
            "threshold_path": tmp_path / "threshold.json",
            "events_path": tmp_path / "events.json",
            "reports_dir": tmp_path / "reports",
        },
    )
    write_model(config)
    write_threshold(config, "raw", threshold_value=-10.0)
    write_processed_captures(config, "calibration")
    write_processed_captures(config)
    fit_matrices = []

    class SuccessfulEstimator:
        def fit(self, counts):
            fit_matrices.append(counts.copy())
            alpha = np.asarray([1.1, 2.05, 3.05, 4.0])
            return DirichletFit(
                alpha.copy(), alpha, 10.2, 1.0 / 10.2, 1, True, -10.0,
                ScipyLogLikelihood().calculate(counts, alpha),
            )

    monkeypatch.setattr(
        scoring_stage,
        "create_estimator",
        lambda *_args: SuccessfulEstimator(),
    )

    result = adapt_windows(config)
    events = json.loads(config.outputs.events_path.read_text(encoding="utf-8"))
    report = json.loads(
        (config.outputs.reports_dir / "adaptation.json").read_text(
            encoding="utf-8"
        )
    )

    assert [event["model_fingerprint"] for event in events[:2]] == [
        report["initial_model_fingerprint"],
        report["initial_model_fingerprint"],
    ]
    assert events[2]["model_fingerprint"] == report["final_model_fingerprint"]
    assert events[2]["threshold"] == pytest.approx(
        report["threshold_update_attempts"][0]["new_threshold"]
    )
    assert events[2]["expected_profile"] == pytest.approx(
        {"tcp": 1.1 / 10.2, "udp": 2.05 / 10.2,
         "ssdp": 3.05 / 10.2, "arp": 4 / 10.2}
    )
    assert result["model_refit_count"] == 1
    assert report["initial_model_fingerprint"] != report["final_model_fingerprint"]
    assert report["model_refit_attempts"][0]["promoted"] is True
    assert report["model_refit_attempts"][1]["promoted"] is False
    assert all(
        np.array_equal(matrix[:4], np.eye(4, dtype=np.int64))
        for matrix in fit_matrices
    )


def test_periodic_refit_rejects_nonconverged_candidate(
    tmp_path,
    config_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = config_factory(
        ingest={"processed_root": tmp_path},
        estimator={"log_likelihood_backend": "scipy"},
        calibration={"minimum_samples": 2, "quantile": 0.25},
        adaptation={
            "mode": "periodic_refit",
            "buffer_size": 2,
            "update_every": 2,
            "minimum_refit_windows": 2,
            "safe_margin": 0.0,
        },
        outputs={
            "model_path": tmp_path / "model.json",
            "threshold_path": tmp_path / "threshold.json",
            "events_path": tmp_path / "events.json",
            "reports_dir": tmp_path / "reports",
        },
    )
    write_model(config)
    write_threshold(config, "raw", threshold_value=-1e9)
    write_processed_captures(config, "calibration")
    write_processed_captures(config)

    class NonconvergingEstimator:
        def fit(self, counts):
            alpha = np.asarray([2.0, 3.0, 4.0, 5.0])
            return DirichletFit(
                alpha.copy(), alpha, 14.0, 1.0 / 14.0, 1, False, -10.0, 1e9
            )

    monkeypatch.setattr(
        scoring_stage,
        "create_estimator",
        lambda *_args: NonconvergingEstimator(),
    )

    result = adapt_windows(config)
    events = json.loads(config.outputs.events_path.read_text(encoding="utf-8"))
    report = json.loads(
        (config.outputs.reports_dir / "adaptation.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["model_refit_count"] == 0
    assert report["initial_model_fingerprint"] == report["final_model_fingerprint"]
    assert all(
        event["model_fingerprint"] == report["initial_model_fingerprint"]
        for event in events
    )
    assert all(
        attempt["reason"] == "candidate did not converge"
        for attempt in report["model_refit_attempts"]
    )
    assert result["threshold_update_count"] == 0
    assert report["initial_threshold"] == report["final_threshold"]
    assert report["initial_score_iqr"] == report["final_score_iqr"]
    assert all(event["threshold"] == -1e9 for event in events)
    assert all(not update["promoted"] for update in report["threshold_update_attempts"])


@pytest.fixture
def adaptive_config(tmp_path, config_factory):
    config = config_factory(
        ingest={"raw_root": tmp_path / "no-pcaps", "processed_root": tmp_path},
        estimator={"log_likelihood_backend": "scipy"},
        calibration={"minimum_samples": 2, "quantile": 0.25},
        adaptation={
            "mode": "periodic_refit", "buffer_size": 2, "update_every": 2,
            "minimum_refit_windows": 2, "safe_margin": 0.0,
            "volume_quantile": 1.0, "max_threshold_shift_iqr": 10.0,
        },
        outputs={
            "model_path": tmp_path / "model.json",
            "threshold_path": tmp_path / "threshold.json",
            "events_path": tmp_path / "events.json",
            "reports_dir": tmp_path / "reports",
        },
    )
    write_model(config)
    write_threshold(config, "raw", threshold_value=-10.0)
    write_processed_captures(config, "calibration")
    write_processed_captures(config)
    return config


def read_adaptation_outputs(config):
    events = json.loads(config.outputs.events_path.read_text())
    report = json.loads((config.outputs.reports_dir / "adaptation.json").read_text())
    return events, report


def test_legacy_threshold_uses_all_recent_scores_without_calibration(adaptive_config):
    config = adaptive_config
    config.adaptation.policy = "legacy"
    config.adaptation.mode = "adaptive_threshold"
    config.adaptation.max_threshold_shift_iqr = 0.0
    write_threshold(config, "raw", threshold_value=-1.0)
    for capture_id in config.ingest.partitions.calibration:
        (config.ingest.processed_root / config.ingest.dataset_folder / f"{capture_id}.json").unlink()
    initial_model = config.outputs.model_path.read_bytes()
    initial_threshold = config.outputs.threshold_path.read_bytes()

    result = adapt_windows(config)
    events, report = read_adaptation_outputs(config)

    assert events[0]["is_anomaly"] is True
    expected_threshold = np.quantile([events[0]["score"], events[1]["score"]], 0.25)
    assert [event["threshold"] for event in events[:2]] == [-1.0, -1.0]
    assert events[2]["threshold"] == pytest.approx(expected_threshold)
    assert result["adaptation_policy"] == report["policy"] == "legacy"
    assert result["threshold_update_count"] == 2
    assert result["model_refit_count"] == 0
    assert report["packet_count_limit"] is None
    assert report["max_threshold_shift_iqr"] is None
    assert all(update["source"] == "recent_scores" for update in report["threshold_update_attempts"])
    assert config.outputs.model_path.read_bytes() == initial_model
    assert config.outputs.threshold_path.read_bytes() == initial_threshold


def test_legacy_refit_uses_baseline_gate_and_recalibrates_on_all_recent_counts(
    adaptive_config, monkeypatch,
):
    config = adaptive_config
    config.adaptation.policy = "legacy"
    config.adaptation.update_every = 3
    config.adaptation.buffer_size = 3
    config.adaptation.max_concentration_ratio = 1.0
    config.adaptation.max_category_shift = 0.0
    config.adaptation.max_threshold_shift_iqr = 0.0
    write_processed_captures(config, counts=((10, 0, 0, 0), (1, 2, 3, 4), (0, 0, 0, 0)))
    matrices = []
    candidate_alpha = np.asarray([2.0, 4.0, 6.0, 8.0])

    class SuccessfulEstimator:
        def fit(self, counts):
            matrices.append(counts.copy())
            likelihood = ScipyLogLikelihood().calculate(counts, candidate_alpha)
            return DirichletFit(
                candidate_alpha.copy(), candidate_alpha, 20.0, 0.05,
                1, True, likelihood, likelihood,
            )

    monkeypatch.setattr(scoring_stage, "create_estimator", lambda *_: SuccessfulEstimator())
    result = adapt_windows(config)
    events, report = read_adaptation_outputs(config)

    assert events[0]["is_anomaly"] is True
    assert result["model_refit_count"] == 2
    assert all(tuple(row) != (10, 0, 0, 0) for matrix in matrices for row in matrix[4:])
    scores = [
        scoring_stage.anomaly_score(np.asarray(counts), candidate_alpha, ScipyLogLikelihood(), "raw")
        for counts in ((10, 0, 0, 0), (1, 2, 3, 4), (0, 0, 0, 0))
    ]
    assert report["threshold_update_attempts"][0]["new_threshold"] == pytest.approx(np.quantile(scores, 0.25))
    assert events[2]["model_fingerprint"] == report["initial_model_fingerprint"]
    assert events[3]["model_fingerprint"] == report["final_model_fingerprint"]
    assert events[3]["threshold"] == pytest.approx(np.quantile(scores, 0.25))


@pytest.mark.parametrize("failure", ["nonconverged", "fit_error", "invalid_alpha", "nonfinite_likelihood"])
def test_legacy_failed_refit_retains_model_and_uses_threshold_fallback(
    adaptive_config, monkeypatch, failure,
):
    config = adaptive_config
    config.adaptation.policy = "legacy"

    class RejectedEstimator:
        def fit(self, counts):
            if failure == "fit_error":
                raise DataValidationError("fitting failed")
            alpha = np.asarray([1.0, 2.0, 3.0, 4.0])
            likelihood = ScipyLogLikelihood().calculate(counts, alpha)
            if failure == "invalid_alpha":
                alpha[0] = 0.0
            if failure == "nonfinite_likelihood":
                likelihood = float("nan")
            return DirichletFit(
                alpha.copy(), alpha, 10.0, 0.1, 1,
                failure != "nonconverged", likelihood, likelihood,
            )

    monkeypatch.setattr(scoring_stage, "create_estimator", lambda *_: RejectedEstimator())
    result = adapt_windows(config)
    events, report = read_adaptation_outputs(config)

    assert result["model_refit_count"] == 0
    assert result["threshold_update_count"] == 2
    assert report["initial_alpha"] == report["final_alpha"]
    assert report["initial_model_fingerprint"] == report["final_model_fingerprint"]
    assert events[2]["threshold"] != events[1]["threshold"]
    assert all(attempt["reason"] for attempt in report["model_refit_attempts"])
    assert all(update["source"] == "recent_scores" for update in report["threshold_update_attempts"])


@pytest.mark.parametrize("mode", ["adaptive_threshold", "periodic_refit"])
def test_only_confident_windows_with_bounded_volume_influence_updates(
    adaptive_config, monkeypatch, mode,
):
    config = adaptive_config
    config.adaptation.mode = mode
    config.adaptation.safe_margin = 1.0
    config.adaptation.buffer_size = 4
    config.adaptation.update_every = 5
    write_threshold(config, "raw", threshold_value=-5.0)
    write_processed_captures(
        config, counts=((10, 0, 0, 0), (1, 2, 3, 4), (100, 200, 300, 400),
                        (1, 0, 0, 0), (0, 0, 0, 0), None),
    )
    fit_matrices = []

    class UnchangedEstimator:
        def fit(self, counts):
            fit_matrices.append(counts.copy())
            alpha = np.asarray([1.0, 2.0, 3.0, 4.0])
            likelihood = ScipyLogLikelihood().calculate(counts, alpha)
            return DirichletFit(
                alpha.copy(), alpha, 10.0, 0.1, 1, True, likelihood, likelihood,
            )

    monkeypatch.setattr(scoring_stage, "create_estimator", lambda *_: UnchangedEstimator())
    adapt_windows(config)
    events, report = read_adaptation_outputs(config)

    assert events[0]["is_anomaly"] is True
    assert events[1]["is_anomaly"] is False  # Normal, but too close to the threshold.
    assert report["packet_count_limit"] == 10.0
    assert report["rejected_learning_windows"]["excessive_volume"] == 2
    assert report["rejected_learning_windows"]["low_confidence"] >= 4
    expected_scores = [-np.log(10.0), 0.0]
    first_update = report["threshold_update_attempts"][0]
    assert first_update["promoted"] is True
    assert first_update["new_threshold"] == pytest.approx(np.quantile(expected_scores, 0.25))
    assert first_update["buffer_window_count"] == 2
    assert report["safe_window_count"] <= config.adaptation.buffer_size
    if mode == "periodic_refit":
        assert fit_matrices
        assert all(
            tuple(row) in {(1, 0, 0, 0), (0, 0, 0, 0)}
            for matrix in fit_matrices for row in matrix[4:]
        )


def test_volume_guard_rejects_a_flood_that_has_a_normal_normalized_score(adaptive_config):
    config = adaptive_config
    config.adaptation.mode = "adaptive_threshold"
    config.calibration.score_type = "normalized"
    write_threshold(config, "normalized", threshold_value=-5.0)
    write_processed_captures(
        config, counts=((1, 2, 3, 4), (0, 0, 0, 0), (100, 200, 300, 400)),
    )

    adapt_windows(config)
    events, report = read_adaptation_outputs(config)

    assert events[2]["is_anomaly"] is False
    assert report["rejected_learning_windows"]["excessive_volume"] == 2
    expected = np.quantile([-0.4477098690890937, 0.0], 0.25)
    assert report["threshold_update_attempts"][0]["new_threshold"] == pytest.approx(expected)


def test_default_margin_and_threshold_limit_allow_a_conservative_update(adaptive_config):
    config = adaptive_config
    config.adaptation.mode = "adaptive_threshold"
    config.adaptation.safe_margin = AdaptationConfig().safe_margin
    config.adaptation.max_threshold_shift_iqr = AdaptationConfig().max_threshold_shift_iqr
    write_threshold(config, "raw", threshold_value=-3.4)
    write_processed_captures(config, counts=((1, 0, 0, 0), (0, 0, 0, 0)))

    result = adapt_windows(config)
    _, report = read_adaptation_outputs(config)

    assert result["threshold_update_count"] == 1
    update = report["threshold_update_attempts"][0]
    assert update["promoted"] is True
    assert 0 < update["new_threshold"] - update["old_threshold"] < 2.0


def test_refitting_uses_the_loaded_models_backend_and_precision(adaptive_config, monkeypatch):
    config = adaptive_config
    write_model(config, log_likelihood_backend="lm", precision_digits=7)
    write_threshold(config, "raw", threshold_value=-10.0)
    original_factory = scoring_stage.create_estimator
    calls = []

    def create_estimator(settings, precision_digits):
        calls.append((settings.log_likelihood_backend, precision_digits))
        return original_factory(settings, precision_digits)

    monkeypatch.setattr(scoring_stage, "create_estimator", create_estimator)
    adapt_windows(config)

    assert calls == [("lm", 7)]
    assert config.estimator.log_likelihood_backend == "scipy"


@pytest.mark.parametrize(
    ("candidate_alpha", "likelihood", "reason"),
    [
        ([2.0, 4.0, 6.0, 8.0], -9.0, "concentration change"),
        ([0.5, 1.0, 1.5, 2.0], -9.0, "concentration change"),
        ([2.0, 1.0, 3.0, 4.0], -9.0, "category proportion change"),
        ([0.0, 2.0, 3.0, 4.0], -9.0, "candidate alpha"),
        ([-1.0, 2.0, 3.0, 4.0], -9.0, "candidate alpha"),
        ([float("nan"), 2.0, 3.0, 4.0], -9.0, "candidate alpha"),
        ([float("inf"), 2.0, 3.0, 4.0], -9.0, "candidate alpha"),
        ([1.0, 2.0, 3.0], -9.0, "candidate alpha"),
        ([1.0, 2.0, 3.0, 4.0], float("nan"), "likelihood must be finite"),
        ([1.0, 2.0, 3.0, 4.0], -1e9, "likelihood is worse"),
    ],
)
def test_rejected_refit_keeps_all_active_state(
    adaptive_config, monkeypatch, candidate_alpha, likelihood, reason,
):
    config = adaptive_config
    original_model = config.outputs.model_path.read_bytes()
    original_threshold = config.outputs.threshold_path.read_bytes()

    class RejectedEstimator:
        def fit(self, _counts):
            alpha = np.asarray(candidate_alpha)
            return DirichletFit(
                alpha.copy(), alpha, 10.0, 0.1, 1, True, -10.0, likelihood,
            )

    monkeypatch.setattr(scoring_stage, "create_estimator", lambda *_: RejectedEstimator())
    result = adapt_windows(config)
    events, report = read_adaptation_outputs(config)

    assert result["model_refit_count"] == result["threshold_update_count"] == 0
    assert all(reason in attempt["reason"] for attempt in report["model_refit_attempts"])
    assert report["initial_alpha"] == report["final_alpha"]
    assert report["initial_threshold"] == report["final_threshold"]
    assert report["initial_score_iqr"] == report["final_score_iqr"]
    assert all(event["model_fingerprint"] == report["initial_model_fingerprint"] for event in events)
    assert all(event["threshold"] == -10.0 for event in events)
    assert config.outputs.model_path.read_bytes() == original_model
    assert config.outputs.threshold_path.read_bytes() == original_threshold


@pytest.mark.parametrize(
    ("mode", "failure"),
    [("periodic_refit", "fit_error"),
     ("periodic_refit", "threshold_jump"), ("adaptive_threshold", "threshold_jump"),
     ("periodic_refit", "zero_iqr"), ("adaptive_threshold", "zero_iqr"),
     ("periodic_refit", "nonfinite_score"), ("adaptive_threshold", "nonfinite_score")],
)
def test_failed_update_has_no_threshold_fallback(adaptive_config, monkeypatch, mode, failure):
    config = adaptive_config
    config.adaptation.mode = mode

    class Estimator:
        def fit(self, counts):
            if failure == "fit_error":
                raise DataValidationError("fitting failed")
            alpha = np.asarray([1.1, 2.05, 3.05, 4.0])
            return DirichletFit(
                alpha.copy(), alpha, 10.2, 1 / 10.2, 1, True, -10.0, -9.0,
            )

    monkeypatch.setattr(scoring_stage, "create_estimator", lambda *_: Estimator())
    if failure == "threshold_jump":
        config.adaptation.max_threshold_shift_iqr = 0.1
    elif failure == "zero_iqr":
        write_processed_captures(config, counts=((1, 2, 3, 4), (1, 2, 3, 4)))
    elif failure == "nonfinite_score":
        original_score = scoring_stage.anomaly_score
        score_calls = 0

        def score_with_invalid_candidate(*args):
            nonlocal score_calls
            candidate_score = score_calls % 4 >= 2
            score_calls += 1
            return float("nan") if candidate_score else original_score(*args)

        monkeypatch.setattr(scoring_stage, "anomaly_score", score_with_invalid_candidate)

    result = adapt_windows(config)
    events, report = read_adaptation_outputs(config)

    assert result["model_refit_count"] == result["threshold_update_count"] == 0
    assert report["initial_alpha"] == report["final_alpha"]
    assert report["initial_threshold"] == report["final_threshold"]
    assert report["initial_score_iqr"] == report["final_score_iqr"]
    assert all(event["threshold"] == -10.0 for event in events)
    assert all(attempt.get("reason") for attempt in report["threshold_update_attempts"])


@pytest.mark.parametrize("scores", [[float("nan"), 0], [float("inf"), 0], [1, 1], []])
def test_invalid_threshold_evidence_is_rejected(scores):
    with pytest.raises(DataValidationError):
        scoring_stage._candidate_threshold(scores, 0.25)


def test_missing_calibration_counts_fails_before_overwriting_events(adaptive_config):
    config = adaptive_config
    first_capture = config.ingest.partitions.calibration[0]
    path = config.ingest.processed_root / config.ingest.dataset_folder / f"{first_capture}.json"
    path.unlink()
    config.outputs.events_path.write_text("previous results")

    with pytest.raises(DataValidationError, match="processed dataset not found"):
        adapt_windows(config)

    assert config.outputs.events_path.read_text() == "previous results"


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["adaptive_threshold", "periodic_refit"])
@pytest.mark.parametrize("policy", ["conservative", "legacy"])
def test_adapt_cli_runs_without_pcaps(adaptive_config, mode, policy):
    config = adaptive_config
    config.adaptation.mode = mode
    config.adaptation.policy = policy
    config_path = config.outputs.reports_dir.parent / "config.json"
    config_path.write_text(config.model_dump_json())
    environment = dict(os.environ, LM_IDNET_LOG_PATH=os.devnull)

    result = subprocess.run(
        [sys.executable, "-m", "lm_idnet.cli", "adapt", "--config", str(config_path)],
        capture_output=True, text=True, env=environment, check=False,
    )

    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    assert status["status"] == "completed"
    assert status["adaptation_mode"] == mode
    assert status["adaptation_policy"] == policy
    events, report = read_adaptation_outputs(config)
    assert status["window_count"] == len(events) == 4
    assert report["packet_count_limit"] == (10.0 if policy == "conservative" else None)
    assert not config.ingest.raw_root.exists()


@pytest.fixture
def persistent_detector(adaptive_config, monkeypatch):
    class Estimator:
        def fit(self, counts):
            alpha = np.asarray([1.1, 2.05, 3.05, 4.0])
            likelihood = ScipyLogLikelihood().calculate(counts, alpha)
            return DirichletFit(
                alpha.copy(), alpha, float(alpha.sum()), 1 / float(alpha.sum()),
                1, True, likelihood, likelihood,
            )

    monkeypatch.setattr(scoring_stage, "create_estimator", lambda *_: Estimator())
    return adaptive_config


@pytest.mark.parametrize("policy", ["conservative", "legacy"])
@pytest.mark.parametrize("mode", ["adaptive_threshold", "periodic_refit"])
def test_persistent_updates_restore_accepted_pair(persistent_detector, policy, mode):
    config = persistent_detector
    config.adaptation.policy = policy
    config.adaptation.mode = mode
    state_dir = config.outputs.reports_dir.parent / "state"
    original_model = config.outputs.model_path.read_bytes()
    original_threshold = config.outputs.threshold_path.read_bytes()

    result = adapt_windows(config, state_dir=state_dir)
    _, report = read_adaptation_outputs(config)
    saved = load_detector_state(state_dir, config)

    assert result["threshold_update_count"] >= 1
    assert result["final_state_version"] == saved.version_id
    assert result["starting_state_version"] != saved.version_id
    assert saved.alpha == tuple(report["final_alpha"])
    assert saved.threshold.threshold == report["final_threshold"]
    assert saved.threshold.score_iqr == report["final_score_iqr"]
    assert saved.baseline_threshold.threshold == -10.
    assert saved.baseline_model.alpha == (1., 2., 3., 4.)
    assert config.outputs.model_path.read_bytes() == original_model
    assert config.outputs.threshold_path.read_bytes() == original_threshold

    config.outputs.model_path.unlink()
    config.outputs.threshold_path.unlink()
    for capture_id in config.ingest.partitions.calibration:
        (config.ingest.processed_root / config.ingest.dataset_folder / f"{capture_id}.json").unlink()
    restarted = adapt_windows(config, state_dir=state_dir)
    events, _ = read_adaptation_outputs(config)
    assert restarted["starting_state_version"] == saved.version_id
    assert events[0]["threshold"] == saved.threshold.threshold
    assert events[0]["model_fingerprint"] == saved.threshold.model_fingerprint


@pytest.mark.parametrize("policy", ["conservative", "legacy"])
@pytest.mark.parametrize("mode", ["adaptive_threshold", "periodic_refit"])
def test_failed_persistence_rejects_whole_update(persistent_detector, monkeypatch, policy, mode):
    config = persistent_detector
    config.adaptation.policy = policy
    config.adaptation.mode = mode
    state_dir = config.outputs.reports_dir.parent / "state"
    initial = load_detector_state(state_dir, config)
    pointer = (state_dir / "current.json").read_bytes()

    def fail(*_args):
        raise DataValidationError("storage unavailable")

    monkeypatch.setattr(scoring_stage, "save_detector_state", fail)
    result = adapt_windows(config, state_dir=state_dir)
    events, report = read_adaptation_outputs(config)

    assert result["threshold_update_count"] == result["model_refit_count"] == 0
    assert (state_dir / "current.json").read_bytes() == pointer
    assert load_detector_state(state_dir, config) == initial
    assert all(event["threshold"] == initial.threshold.threshold for event in events)
    assert all(event["model_fingerprint"] == initial.threshold.model_fingerprint for event in events)
    assert all(update["reason"] == "storage unavailable" for update in report["threshold_update_attempts"])


def test_legacy_fitting_failure_persists_threshold_fallback(persistent_detector, monkeypatch):
    config = persistent_detector
    config.adaptation.policy = "legacy"

    class RejectedEstimator:
        def fit(self, _counts):
            alpha = np.asarray([1., 2., 3., 4.])
            return DirichletFit(alpha.copy(), alpha, 10., .1, 1, False, -10., -9.)

    monkeypatch.setattr(scoring_stage, "create_estimator", lambda *_: RejectedEstimator())
    state_dir = config.outputs.reports_dir.parent / "state"
    result = adapt_windows(config, state_dir=state_dir)
    saved = load_detector_state(state_dir, config)
    assert result["threshold_update_count"] > 0
    assert result["model_refit_count"] == 0
    assert saved.source == "adaptive_threshold"
    assert saved.alpha == saved.baseline_model.alpha
    assert saved.threshold.threshold != saved.baseline_threshold.threshold


def test_post_switch_failure_stops_before_writing_success(persistent_detector, monkeypatch):
    config = persistent_detector
    state_dir = config.outputs.reports_dir.parent / "state"
    original_save = scoring_stage.save_detector_state

    def committed_but_unsynced(directory, state):
        original_save(directory, state)
        raise DetectorStateCommitError("pointer changed; sync failed")

    monkeypatch.setattr(scoring_stage, "save_detector_state", committed_but_unsynced)
    config.outputs.events_path.write_text("previous events")
    with pytest.raises(DetectorStateCommitError):
        adapt_windows(config, state_dir=state_dir)
    assert config.outputs.events_path.read_text() == "previous events"
    assert not (config.outputs.reports_dir / "adaptation.json").exists()
    assert load_detector_state(state_dir, config).source == "periodic_refit"


@pytest.mark.integration
@pytest.mark.parametrize("policy", ["conservative", "legacy"])
def test_cli_persists_restarts_and_rolls_back(adaptive_config, policy):
    config = adaptive_config
    config.adaptation.mode = "adaptive_threshold"
    config.adaptation.policy = policy
    config.adaptation.update_every = 3
    state_dir = config.outputs.reports_dir.parent / "state"
    config_path = state_dir.parent / "config.json"
    config_path.write_text(config.model_dump_json())
    environment = dict(os.environ, LM_IDNET_LOG_PATH=os.devnull)

    def run(command):
        result = subprocess.run([
            sys.executable, "-m", "lm_idnet.cli", command, "--config", str(config_path),
            "--state-dir", str(state_dir),
        ], capture_output=True, text=True, env=environment, check=False)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    first = run("adapt")
    assert first["threshold_update_count"] == 1
    config.outputs.model_path.unlink()
    config.outputs.threshold_path.unlink()
    for capture_id in config.ingest.partitions.calibration:
        (config.ingest.processed_root / config.ingest.dataset_folder / f"{capture_id}.json").unlink()
    second = run("adapt")
    assert second["starting_state_version"] == first["final_state_version"]
    index = json.loads((state_dir / "current.json").read_text())
    previous = load_detector_version(state_dir, index["previous_version"])
    restored = run("rollback")
    assert restored["active_version"] == previous.version_id
    assert load_detector_state(state_dir, config) == previous


@pytest.mark.integration
@pytest.mark.parametrize("command", ["adapt", "rollback"])
def test_cli_busy_state_fails_without_success(adaptive_config, command):
    config = adaptive_config
    state_dir = config.outputs.reports_dir.parent / "state"
    config_path = state_dir.parent / "config.json"
    config_path.write_text(config.model_dump_json())
    with detector_state_lock(state_dir):
        result = subprocess.run([
            sys.executable, "-m", "lm_idnet.cli", "--error-format", "json",
            command, "--config", str(config_path), "--state-dir", str(state_dir),
        ], capture_output=True, text=True,
            env=dict(os.environ, LM_IDNET_LOG_PATH=os.devnull), check=False)
    assert result.returncode == 8
    assert not result.stdout
    assert json.loads(result.stderr)["error"]["code"] == "policy_rejection_error"
    assert not (state_dir / "current.json").exists()


@pytest.mark.integration
@pytest.mark.parametrize("command", ["adapt", "rollback"])
def test_persistent_cli_dry_run_does_not_create_state(adaptive_config, command):
    config = adaptive_config
    state_dir = config.outputs.reports_dir.parent / "state"
    config_path = state_dir.parent / "config.json"
    config_path.write_text(config.model_dump_json())
    result = subprocess.run([
        sys.executable, "-m", "lm_idnet.cli", command, "--config", str(config_path),
        "--state-dir", str(state_dir), "--dry-run",
    ], capture_output=True, text=True,
        env=dict(os.environ, LM_IDNET_LOG_PATH=os.devnull), check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "ready"
    assert not state_dir.exists()
