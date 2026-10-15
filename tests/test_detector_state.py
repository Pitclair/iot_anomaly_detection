"""Correctness checks for saved detector state using small temporary files."""

import json

import pytest

from lm_idnet.artifacts import artifact_fingerprint, validate_artifact
from lm_idnet.exceptions import ArtifactCompatibilityError
from lm_idnet.model_fingerprint import model_parameter_fingerprint
from lm_idnet.models.schemas import DetectorStateArtifact, ModelArtifact

pytestmark = pytest.mark.unit


@pytest.fixture
def detector_config(tmp_path, config_factory):
    return config_factory(
        ingest={"processed_root": tmp_path / "processed"},
        estimator={"log_likelihood_backend": "scipy"},
        adaptation={"mode": "periodic_refit"},
        outputs={
            "model_path": tmp_path / "model.json",
            "threshold_path": tmp_path / "threshold.json",
        },
    )


@pytest.fixture
def state_factory(detector_config):
    config = detector_config
    model = ModelArtifact(
        dataset=config.ingest.dataset_folder, device_id=config.ingest.device_id,
        categories=config.ingest.categories, alpha=(1., 2., 3., 4.),
        concentration=10., mean_probabilities=(.1, .2, .3, .4), psi=.1,
        log_likelihood_backend="scipy", precision_digits=config.precision_digits,
        training_capture_ids=config.ingest.partitions.fit, training_window_count=288,
        training_start_utc="2020-10-08T00:00:00Z",
        training_end_utc="2020-10-14T00:00:00Z",
        fit_diagnostics={
            "initial_alpha": (2.5, 2.5, 2.5, 2.5), "iterations": 2,
            "converged": True, "tolerance": 1e-6, "max_iterations": 10,
            "initial_log_likelihood": -10., "final_log_likelihood": -9.,
            "duration_seconds": .1,
        },
    )
    baseline_threshold = {
        "model_fingerprint": artifact_fingerprint(model), "score_type": "raw",
        "quantile": config.calibration.quantile, "threshold": -10.,
        "calibration_capture_ids": config.ingest.partitions.calibration,
        "calibration_window_count": 288,
        "calibration_start_utc": "2020-10-14T00:00:00Z",
        "calibration_end_utc": "2020-10-16T00:00:00Z",
        "quantile_method": "linear", "score_minimum": -12.,
        "score_median": -5., "score_maximum": -1., "score_iqr": 2.,
    }

    def create(*, alpha=(1.1, 2.1, 3.1, 4.1), threshold=-8., **updates):
        raw = {
            "source": "periodic_refit", "baseline_model": model.model_dump(),
            "baseline_threshold": dict(baseline_threshold), "alpha": alpha,
            "threshold": {
                "model_fingerprint": model_parameter_fingerprint(
                    alpha=alpha, categories=model.categories,
                    log_likelihood_backend=model.log_likelihood_backend,
                    precision_digits=model.precision_digits,
                ),
                "threshold": threshold, "score_iqr": 1.5,
            },
            "window_minutes": config.ingest.window_minutes,
            "device_mac": config.ingest.device_mac,
            "calibration": config.calibration.model_dump(),
            "adaptation": config.adaptation.model_dump(),
            "estimator": config.estimator.model_dump(), "packet_count_limit": 100.,
        }
        raw.update(updates)
        return DetectorStateArtifact.model_validate(raw)

    return create


def test_detector_state_round_trip_preserves_active_pair_and_safety_references(state_factory):
    state = state_factory()
    restored = validate_artifact(json.loads(state.model_dump_json()), expected_type="detector_state")

    assert restored == state
    assert restored.alpha != restored.baseline_model.alpha
    assert restored.threshold.threshold == -8.
    assert restored.baseline_threshold.threshold == -10.
    assert restored.packet_count_limit == 100.


@pytest.mark.parametrize(("field", "value"), [
    ("alpha", [0., 2., 3., 4.]), ("alpha", [float("nan"), 2., 3., 4.]),
    ("alpha", [1., 2.]), ("alpha", [1e308] * 4),
    ("schema_version", 2), ("window_minutes", True), ("window_minutes", 0),
    ("device_mac", "invalid"), ("packet_count_limit", None),
    ("packet_count_limit", float("inf")), ("version_id", "../escape"),
])
def test_detector_state_rejects_invalid_values(state_factory, field, value):
    raw = state_factory().model_dump(mode="json")
    raw[field] = value
    with pytest.raises(ArtifactCompatibilityError):
        validate_artifact(raw, expected_type="detector_state")


@pytest.mark.parametrize(("section", "field", "value"), [
    ("threshold", "model_fingerprint", "0" * 64),
    ("baseline_threshold", "model_fingerprint", "0" * 64),
    ("threshold", "threshold", float("nan")),
    ("threshold", "score_iqr", 0.),
    ("baseline_threshold", "threshold", float("inf")),
    ("calibration", "score_type", "normalized"),
    ("calibration", "quantile", .1),
    ("adaptation", "buffer_size", 1),
])
def test_detector_state_rejects_incompatible_pairs(state_factory, section, field, value):
    raw = state_factory().model_dump(mode="json")
    raw[section][field] = value
    with pytest.raises(ArtifactCompatibilityError):
        validate_artifact(raw, expected_type="detector_state")


def test_threshold_only_changes_get_distinct_versions(state_factory):
    first = state_factory()
    second = state_factory(threshold=-7.)
    assert first.threshold.model_fingerprint == second.threshold.model_fingerprint
    assert first.version_id != second.version_id
