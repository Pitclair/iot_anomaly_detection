"""Tests for the LM-versus-SciPy benchmark report."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from lm_idnet.evaluation import benchmark_stage
from lm_idnet.models.schemas import AnomalyEventArtifact

pytestmark = pytest.mark.unit


def _event(score: float, is_anomaly: bool) -> AnomalyEventArtifact:
    return AnomalyEventArtifact(
        device_id="camera",
        capture_id="capture-1",
        window_start_utc="2020-01-01T00:00:00Z",
        window_end_utc="2020-01-01T00:10:00Z",
        counts={"tcp": 2, "udp": 1, "ssdp": 0, "arp": 0},
        score_type="raw",
        score=score,
        threshold=-2.0,
        is_anomaly=is_anomaly,
        severity=max(0.0, -2.0 - score),
        expected_profile={"tcp": 0.1, "udp": 0.2, "ssdp": 0.3, "arp": 0.4},
        category_residuals={
            "tcp": 2 / 3 - 0.1,
            "udp": 1 / 3 - 0.2,
            "ssdp": -0.3,
            "arp": -0.4,
        },
        model_fingerprint="0" * 64,
    )


def test_benchmark_compares_backends_without_using_configured_artifacts(
    tmp_path,
    config_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = config_factory(
        outputs={
            "model_path": tmp_path / "accepted-model.json",
            "threshold_path": tmp_path / "accepted-threshold.json",
            "events_path": tmp_path / "accepted-events.json",
            "reports_dir": tmp_path / "reports",
        }
    )
    counts = np.array([[2, 1, 0, 0], [4, 0, 3, 1]], dtype=np.int64)
    monkeypatch.setattr(
        benchmark_stage,
        "load_training_matrix",
        lambda _config: SimpleNamespace(counts=counts),
    )

    def train(run_config):
        is_lm = run_config.estimator.log_likelihood_backend == "lm"
        return {
            "alpha": [1.000001 if is_lm else 1.0, 2.0, 3.0, 4.0],
            "duration_seconds": 0.2 if is_lm else 0.1,
            "iterations": 11 if is_lm else 10,
            "initial_log_likelihood": -11.0,
            "final_log_likelihood": -10.0,
        }

    monkeypatch.setattr(benchmark_stage, "train_model", train)
    monkeypatch.setattr(
        benchmark_stage,
        "calibrate_threshold",
        lambda run_config: {
            "threshold": -2.000001
            if run_config.estimator.log_likelihood_backend == "lm"
            else -2.0
        },
    )
    monkeypatch.setattr(
        benchmark_stage,
        "score_windows",
        lambda _config: {"anomaly_count": 0},
    )
    monkeypatch.setattr(
        benchmark_stage,
        "_load_events",
        lambda path: [
            _event(-1.000001 if path.parent.name == "lm" else -1.0, False)
        ],
    )
    monkeypatch.setattr(benchmark_stage, "_MEASUREMENT_BATCHES", 1)
    monkeypatch.setattr(benchmark_stage, "_MODEL_FIT_MEASUREMENTS", 1)
    monkeypatch.setattr(benchmark_stage, "_LIKELIHOOD_CALLS", 1)
    monkeypatch.setattr(benchmark_stage, "_SCORE_CALLS", 1)

    result = benchmark_stage.benchmark_backends(config)

    assert result["workload"] == {
        "training_matrix_shape": [2, 4],
        "representative_window_counts": [2, 1, 0, 0],
        "scored_window_count": 1,
    }
    assert result["backends"]["lm"]["model_fit_median_seconds"] == 0.2
    assert result["backends"]["scipy"]["model_fit_median_seconds"] == 0.1
    assert result["comparisons"]["decision_mismatch_count"] == 0
    assert result["comparisons"]["lm_over_scipy_runtime_ratio"][
        "model_fit_median_seconds"
    ] == 2.0
    assert result["report_path"] == str(
        config.outputs.reports_dir / "lm_backend_benchmark.json"
    )
    report = json.loads(
        (config.outputs.reports_dir / "lm_backend_benchmark.json").read_text()
    )
    assert report["report_type"] == "lm_backend_benchmark"
    assert not config.outputs.model_path.exists()
    assert not config.outputs.threshold_path.exists()
    assert not config.outputs.events_path.exists()
