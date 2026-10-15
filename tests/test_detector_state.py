"""Correctness checks for saved detector state using small temporary files."""

import json
from pathlib import Path

import pytest

from lm_idnet import artifacts
from lm_idnet.artifacts import (
    artifact_fingerprint,
    load_detector_version,
    save_detector_version,
    save_detector_state,
    validate_artifact,
)
from lm_idnet.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
    DetectorStateCommitError,
)
from lm_idnet.model_fingerprint import model_parameter_fingerprint
from lm_idnet.models.schemas import DetectorStateArtifact, DetectorStateIndex, ModelArtifact

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


def test_detector_version_stores_complete_pair_without_activating_it(tmp_path, state_factory):
    state = state_factory()
    path = save_detector_version(tmp_path, state)
    assert path == tmp_path / "versions" / f"{state.version_id}.json"
    assert load_detector_version(tmp_path, state.version_id) == state
    assert not (tmp_path / "current.json").exists()


def test_detector_versions_are_immutable(tmp_path, state_factory):
    state = state_factory()
    path = save_detector_version(tmp_path, state)
    original = path.read_bytes()
    replacement = state_factory(version_id=state.version_id, threshold=-7.)
    with pytest.raises(DataValidationError, match="cannot save detector"):
        save_detector_version(tmp_path, replacement)
    assert path.read_bytes() == original
    assert load_detector_version(tmp_path, state.version_id) == state


def test_detector_version_validates_even_modified_schema_instances(tmp_path, state_factory):
    state = state_factory()
    state.alpha = (0., 0., 0., 0.)
    with pytest.raises(ArtifactCompatibilityError):
        save_detector_version(tmp_path, state)
    assert not (tmp_path / "versions").exists()


@pytest.mark.parametrize("version_id", ["../escape", "/tmp/escape", "", "A" * 32])
def test_loading_detector_version_rejects_unsafe_identifiers(tmp_path, version_id):
    with pytest.raises(ArtifactCompatibilityError):
        load_detector_version(tmp_path, version_id)


def test_detector_version_must_match_filename(tmp_path, state_factory):
    state = state_factory()
    path = save_detector_version(tmp_path, state)
    wrong_id = "0" * 32
    path.rename(path.with_name(f"{wrong_id}.json"))
    with pytest.raises(ArtifactCompatibilityError, match="filename"):
        load_detector_version(tmp_path, wrong_id)


def test_detector_index_round_trip_and_validation(state_factory):
    first, second = state_factory(), state_factory()
    index = DetectorStateIndex(active_version=second.version_id, previous_version=first.version_id)
    assert validate_artifact(json.loads(index.model_dump_json()), expected_type="detector_state_index") == index
    raw = index.model_dump(mode="json")
    for invalid_previous in (second.version_id, "../escape"):
        raw["previous_version"] = invalid_previous
        with pytest.raises(ArtifactCompatibilityError):
            validate_artifact(raw, expected_type="detector_state_index")


def test_committing_detector_state_preserves_previous_complete_pair(tmp_path, state_factory):
    first, second = state_factory(), state_factory(threshold=-7.)
    initial = save_detector_state(tmp_path, first)
    assert initial.previous_version is None
    current = save_detector_state(tmp_path, second)
    assert current.active_version == second.version_id
    assert current.previous_version == first.version_id
    assert load_detector_version(tmp_path, current.active_version) == second
    assert load_detector_version(tmp_path, current.previous_version) == first
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("failure", ["snapshot_validation", "index_validation", "snapshot_publish", "pointer_replace", "file_sync"])
def test_failed_save_keeps_previous_pair_active(tmp_path, state_factory, monkeypatch, failure):
    first, second = state_factory(), state_factory(threshold=-7.)
    save_detector_state(tmp_path, first)
    pointer = tmp_path / "current.json"
    original = pointer.read_bytes()

    def fail(*_args, **_kwargs):
        raise OSError("simulated storage failure")

    if failure.endswith("validation"):
        original_load = artifacts.load_artifact

        def invalid_temporary(path, *, expected_type):
            target_type = "detector_state" if failure == "snapshot_validation" else "detector_state_index"
            if str(path).endswith(".tmp") and expected_type == target_type:
                raise ArtifactCompatibilityError("simulated validation failure")
            return original_load(path, expected_type=expected_type)

        monkeypatch.setattr(artifacts, "load_artifact", invalid_temporary)
    elif failure == "snapshot_publish":
        monkeypatch.setattr(artifacts.os, "link", fail)
    elif failure == "pointer_replace":
        monkeypatch.setattr(artifacts.os, "replace", fail)
    else:
        monkeypatch.setattr(artifacts.os, "fsync", fail)

    with pytest.raises((DataValidationError, ArtifactCompatibilityError)):
        save_detector_state(tmp_path, second)
    assert pointer.read_bytes() == original
    assert load_detector_version(tmp_path, first.version_id) == first
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("after_switch", [False, True])
def test_interrupted_switch_exposes_one_complete_pair(tmp_path, state_factory, monkeypatch, after_switch):
    first, second = state_factory(), state_factory(threshold=-7.)
    save_detector_state(tmp_path, first)
    original_replace = artifacts.os.replace

    def interrupt(source, destination):
        if after_switch:
            original_replace(source, destination)
        raise KeyboardInterrupt

    monkeypatch.setattr(artifacts.os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        save_detector_state(tmp_path, second)
    index = json.loads((tmp_path / "current.json").read_text())
    expected = second if after_switch else first
    assert load_detector_version(tmp_path, index["active_version"]) == expected


def test_directory_sync_failure_after_switch_reports_committed_state(tmp_path, state_factory, monkeypatch):
    first, second = state_factory(), state_factory(threshold=-7.)
    save_detector_state(tmp_path, first)
    original_sync = artifacts._sync_directory

    def fail_after_switch(directory):
        if directory == tmp_path:
            raise OSError("simulated directory sync failure")
        original_sync(directory)

    monkeypatch.setattr(artifacts, "_sync_directory", fail_after_switch)
    with pytest.raises(DetectorStateCommitError) as error:
        save_detector_state(tmp_path, second)
    assert error.value.committed
    index = json.loads((tmp_path / "current.json").read_text())
    assert load_detector_version(tmp_path, index["active_version"]) == second
    assert load_detector_version(tmp_path, index["previous_version"]) == first


def test_failed_temporary_cleanup_does_not_undo_commit(tmp_path, state_factory, monkeypatch):
    state = state_factory()
    original_unlink = Path.unlink

    def fail_cleanup(path, *args, **kwargs):
        if path.suffix == ".tmp":
            raise OSError("simulated cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    index = save_detector_state(tmp_path, state)
    assert load_detector_version(tmp_path, index.active_version) == state


def test_saving_rejects_changed_safety_reference(tmp_path, state_factory):
    first = state_factory()
    save_detector_state(tmp_path, first)
    pointer = (tmp_path / "current.json").read_bytes()
    with pytest.raises(ArtifactCompatibilityError, match="packet_count_limit changed"):
        save_detector_state(tmp_path, state_factory(packet_count_limit=200.))
    assert (tmp_path / "current.json").read_bytes() == pointer
