"""Correctness checks for saved detector state using small temporary files."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lm_idnet import artifacts
from lm_idnet.artifacts import (
    artifact_fingerprint,
    detector_state_lock,
    load_detector_version,
    load_detector_state,
    save_artifact,
    save_detector_version,
    save_detector_state,
    record_detector_update,
    rollback_detector_state,
    validate_artifact,
)
from lm_idnet.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
    DetectorStateCommitError,
    PolicyRejectionError,
)
from lm_idnet.model_fingerprint import model_parameter_fingerprint
from lm_idnet.models.schemas import DetectorStateArtifact, DetectorStateIndex, ModelArtifact
from lm_idnet.algorithms.log_likelihood import ScipyLogLikelihood
from lm_idnet.models.scoring_stage import score_window
from lm_idnet.processing.schemas import Metadata, ProcessedDataset, WindowRecord
from lm_idnet.processing.storage import save_processed_dataset

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


def write_initial_artifacts(config, state):
    save_artifact(config.outputs.model_path, state.baseline_model)
    save_artifact(config.outputs.threshold_path, state.baseline_threshold)


def write_calibration_counts(config):
    for capture_id in config.ingest.partitions.calibration:
        start = datetime(2020, 10, 14, tzinfo=timezone.utc)
        windows = tuple(WindowRecord(
            start_utc=start + timedelta(minutes=config.ingest.window_minutes * offset),
            end_utc=start + timedelta(minutes=config.ingest.window_minutes * (offset + 1)),
            categories=config.ingest.categories, counts=(10 + offset, 2, 3, 4), state="observed",
        ) for offset in range(20))
        save_processed_dataset(ProcessedDataset(
            metadata=Metadata(
                device_id=config.ingest.device_id, capture_id=capture_id,
                partition="calibration", date=capture_id, file_source="fixture.pcap",
            ),
            windows=windows,
        ), config.ingest.processed_root / config.ingest.dataset_folder / f"{capture_id}.json")


@pytest.mark.parametrize(("mode", "policy"), [
    ("periodic_refit", "conservative"), ("adaptive_threshold", "conservative"),
    ("periodic_refit", "legacy"), ("adaptive_threshold", "legacy"),
    ("static", "conservative"),
])
def test_empty_state_directory_initializes_matching_pair(tmp_path, detector_config, state_factory, mode, policy):
    config = detector_config
    config.adaptation.mode = mode
    config.adaptation.policy = policy
    state = state_factory(packet_count_limit=100. if mode != "static" and policy == "conservative" else None)
    write_initial_artifacts(config, state)
    if mode != "static" and policy == "conservative":
        write_calibration_counts(config)
    directory = tmp_path / "detector"
    if mode == "static":
        directory.mkdir()

    initial = load_detector_state(directory, config)

    assert initial.source == "initial"
    assert initial.alpha == state.baseline_model.alpha
    assert initial.threshold.threshold == state.baseline_threshold.threshold
    assert initial.threshold.score_iqr == state.baseline_threshold.score_iqr
    assert initial.packet_count_limit == (38. if mode != "static" and policy == "conservative" else None)
    assert load_detector_state(directory, config) == initial


def test_restart_restores_decisions_and_original_guards_without_source_files(
    tmp_path, detector_config, state_factory, window_factory, monkeypatch,
):
    directory = tmp_path / "detector"
    state = state_factory()
    save_detector_state(directory, state)

    def unexpected_initialization(*_args, **_kwargs):
        pytest.fail("a restart must not reload initial artifacts or calibration counts")

    monkeypatch.setattr(artifacts, "load_model", unexpected_initialization)
    restored = load_detector_state(directory, detector_config)
    assert restored == state
    assert restored.baseline_model.alpha != restored.alpha
    assert restored.baseline_threshold.threshold == -10.
    assert restored.packet_count_limit == 100.

    def decision(snapshot):
        alpha = np.asarray(snapshot.alpha)
        return score_window(
            window_factory(), device_id=snapshot.baseline_model.device_id,
            capture_id="fixture", model_categories=snapshot.baseline_model.categories,
            expected_profile=tuple(alpha / alpha.sum()),
            model_fingerprint=snapshot.threshold.model_fingerprint, alpha=alpha,
            log_likelihood=ScipyLogLikelihood(), threshold=snapshot.threshold.threshold,
            score_iqr=snapshot.threshold.score_iqr, score_type=snapshot.calibration.score_type,
        )

    assert decision(restored) == decision(state)
    assert not detector_config.outputs.model_path.exists()
    assert not detector_config.ingest.processed_root.exists()


@pytest.mark.parametrize(("section", "field", "value"), [
    ("ingest", "device_id", "other-device"),
    ("ingest", "dataset_folder", "other-dataset"),
    ("ingest", "device_mac", "00:11:22:33:44:55"),
    ("ingest", "categories", ("arp", "ssdp", "udp", "tcp")),
    ("ingest", "window_minutes", 5),
    ("calibration", "score_type", "normalized"),
    ("calibration", "quantile", .1),
    ("calibration", "minimum_samples", 31),
    ("adaptation", "policy", "legacy"),
    ("adaptation", "max_threshold_shift_iqr", 3.),
    ("estimator", "tolerance_delta", .1),
    ("estimator", "log_likelihood_backend", "lm"),
])
def test_restart_rejects_incompatible_configuration(tmp_path, detector_config, state_factory, section, field, value):
    state = state_factory()
    save_detector_state(tmp_path, state)
    changed_section = getattr(detector_config, section).model_copy(update={field: value})
    changed = detector_config.model_copy(update={section: changed_section})
    pointer = (tmp_path / "current.json").read_bytes()
    with pytest.raises(ArtifactCompatibilityError, match="does not match configuration"):
        load_detector_state(tmp_path, changed)
    assert (tmp_path / "current.json").read_bytes() == pointer


def test_restart_rejects_changed_precision(tmp_path, detector_config, state_factory):
    save_detector_state(tmp_path, state_factory())
    changed = detector_config.model_copy(update={"precision_digits": 8})
    with pytest.raises(ArtifactCompatibilityError, match="precision"):
        load_detector_state(tmp_path, changed)


def test_restart_allows_relocated_input_and_output_paths(tmp_path, detector_config, state_factory):
    state = state_factory()
    save_detector_state(tmp_path, state)
    relocated = detector_config.model_copy(update={
        "outputs": detector_config.outputs.model_copy(update={"model_path": tmp_path / "absent.json"}),
        "ingest": detector_config.ingest.model_copy(update={"processed_root": tmp_path / "absent"}),
    })
    assert load_detector_state(tmp_path, relocated) == state


@pytest.mark.parametrize("corruption", ["index", "snapshot", "missing_snapshot", "missing_index"])
def test_corrupt_existing_state_never_reinitializes(tmp_path, detector_config, state_factory, corruption):
    state = state_factory()
    write_initial_artifacts(detector_config, state)
    write_calibration_counts(detector_config)
    directory = tmp_path / "detector"
    save_detector_state(directory, state)
    pointer = directory / "current.json"
    snapshot = directory / "versions" / f"{state.version_id}.json"
    if corruption == "index":
        pointer.write_text("{")
    elif corruption == "snapshot":
        snapshot.write_text("{")
    elif corruption == "missing_snapshot":
        snapshot.unlink()
    else:
        pointer.unlink()
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    with pytest.raises(ArtifactCompatibilityError):
        load_detector_state(directory, detector_config)
    assert {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()} == before


@pytest.mark.integration
def test_new_process_restores_accepted_snapshot(tmp_path, detector_config, state_factory):
    directory = tmp_path / "detector"
    state = state_factory()
    save_detector_state(directory, state)
    config_path = tmp_path / "config.json"
    config_path.write_text(detector_config.model_dump_json())
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; from lm_idnet.artifacts import load_detector_state; "
        "from lm_idnet.config import load_config; "
        "print(load_detector_state(sys.argv[1], load_config(sys.argv[2])).model_dump_json())",
        str(directory), str(config_path),
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert DetectorStateArtifact.model_validate_json(result.stdout) == state


def test_initialization_rejects_mismatched_pair_before_creating_state(tmp_path, detector_config, state_factory):
    state = state_factory()
    write_initial_artifacts(detector_config, state)
    write_calibration_counts(detector_config)
    threshold = json.loads(detector_config.outputs.threshold_path.read_text())
    threshold["model_fingerprint"] = "0" * 64
    detector_config.outputs.threshold_path.write_text(json.dumps(threshold))
    directory = tmp_path / "detector"
    with pytest.raises(ArtifactCompatibilityError, match="fingerprint"):
        load_detector_state(directory, detector_config)
    assert not directory.exists()


def test_rollback_restores_previous_pair_without_initial_artifacts(tmp_path, detector_config, state_factory):
    first, second = state_factory(), state_factory(threshold=-7.)
    save_detector_state(tmp_path, first)
    save_detector_state(tmp_path, second)
    index = rollback_detector_state(tmp_path, detector_config)
    assert index.active_version == first.version_id
    assert index.previous_version == second.version_id
    assert load_detector_state(tmp_path, detector_config) == first
    rollback_detector_state(tmp_path, detector_config)
    assert load_detector_state(tmp_path, detector_config) == second


@pytest.mark.parametrize("has_initial_version", [False, True])
def test_rollback_without_previous_version_does_not_initialize(tmp_path, detector_config, state_factory, has_initial_version):
    directory = tmp_path / "detector"
    if has_initial_version:
        save_detector_state(directory, state_factory())
    pointer = directory / "current.json"
    before = pointer.read_bytes() if pointer.exists() else None
    with pytest.raises(PolicyRejectionError, match="no previous"):
        rollback_detector_state(directory, detector_config)
    assert (pointer.read_bytes() if pointer.exists() else None) == before


@pytest.mark.parametrize("failure", ["corrupt_previous", "wrong_configuration", "pointer_write"])
def test_failed_rollback_preserves_active_pointer(tmp_path, detector_config, state_factory, monkeypatch, failure):
    first, second = state_factory(), state_factory(threshold=-7.)
    save_detector_state(tmp_path, first)
    save_detector_state(tmp_path, second)
    pointer = tmp_path / "current.json"
    before = pointer.read_bytes()
    config = detector_config
    if failure == "corrupt_previous":
        (tmp_path / "versions" / f"{first.version_id}.json").write_text("{")
    elif failure == "wrong_configuration":
        config = config.model_copy(update={"ingest": config.ingest.model_copy(update={"window_minutes": 5})})
    else:
        def fail(*_args):
            raise OSError("simulated rollback write failure")
        monkeypatch.setattr(artifacts.os, "replace", fail)
    with pytest.raises((ArtifactCompatibilityError, DataValidationError)):
        rollback_detector_state(tmp_path, config)
    assert pointer.read_bytes() == before
    assert load_detector_version(tmp_path, second.version_id) == second


def test_state_lock_blocks_aliases_and_releases_after_failure(tmp_path):
    directory = tmp_path / "state"
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    with pytest.raises(RuntimeError):
        with detector_state_lock(directory):
            with pytest.raises(PolicyRejectionError, match="in use"):
                with detector_state_lock(alias):
                    pytest.fail("another writer acquired the same directory")
            with detector_state_lock(tmp_path / "other"):
                pass
            raise RuntimeError("simulated interrupted writer")
    with detector_state_lock(alias):
        pass


def test_version_retention_keeps_active_and_previous_and_cleans_orphans(tmp_path, detector_config, state_factory):
    for offset in range(12):
        current = state_factory(threshold=-8. + offset / 100)
        index = save_detector_state(tmp_path, current)
        expected = {index.active_version}
        if index.previous_version is not None:
            expected.add(index.previous_version)
        assert {path.stem for path in (tmp_path / "versions").glob("*.json")} == expected
    orphan = state_factory()
    save_detector_version(tmp_path, orphan)
    (tmp_path / "versions" / f"{'0' * 32}.json").write_text("interrupted write")
    (tmp_path / "versions" / f".{'0' * 32}.json.abandoned.tmp").write_text("partial")
    (tmp_path / ".current.json.abandoned.tmp").write_text("partial")
    (tmp_path / ".history.json.abandoned.tmp").write_text("partial")
    unrelated = tmp_path / "versions" / "notes.txt"
    unrelated.write_text("keep this")

    assert load_detector_state(tmp_path, detector_config) == current
    assert {path.stem for path in (tmp_path / "versions").glob("*.json")} == expected
    assert not list(tmp_path.rglob("*.tmp"))
    assert unrelated.read_text() == "keep this"


def test_cleanup_failure_does_not_reject_committed_version(tmp_path, detector_config, state_factory, monkeypatch, caplog):
    first, second, third = state_factory(), state_factory(threshold=-7.), state_factory(threshold=-6.)
    save_detector_state(tmp_path, first)
    save_detector_state(tmp_path, second)
    obsolete = tmp_path / "versions" / f"{first.version_id}.json"
    original_unlink = Path.unlink

    def fail_one_file(path, *args, **kwargs):
        if path == obsolete:
            raise PermissionError("simulated cleanup failure")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_one_file)
        index = save_detector_state(tmp_path, third)
        assert index.active_version == third.version_id
        assert load_detector_state(tmp_path, detector_config) == third
    assert "Cannot clean detector state" in caplog.text
    assert load_detector_state(tmp_path, detector_config) == third
    assert not obsolete.exists()


def test_update_history_bounds_record_count_and_reason_size(tmp_path, state_factory):
    state = state_factory()
    save_detector_state(tmp_path, state)
    for offset in range(105):
        record_detector_update(
            tmp_path, source="periodic_refit", old_version=state.version_id,
            promoted=False, reason=f"rejected {offset}: " + "x" * 2000,
            after_window=offset + 1,
        )
    path = tmp_path / "history.json"
    history = artifacts.load_artifact(path, expected_type="detector_state_history")
    assert len(history.records) == 100
    assert history.records[0].after_window == 6
    assert history.records[-1].after_window == 105
    assert all(len(record.reason) == 1024 for record in history.records)
    assert path.stat().st_size < 160 * 1024


def test_update_history_rejects_invalid_outcomes_and_excess_records(state_factory):
    version = state_factory().version_id
    record = {"source": "periodic_refit", "old_version": version, "promoted": False}
    raw = {"artifact_type": "detector_state_history", "records": [record]}
    with pytest.raises(ArtifactCompatibilityError, match="reason"):
        validate_artifact(raw, expected_type="detector_state_history")
    record["reason"] = "candidate rejected"
    raw["records"] = [record] * 101
    with pytest.raises(ArtifactCompatibilityError):
        validate_artifact(raw, expected_type="detector_state_history")


def test_rollback_records_version_change(tmp_path, detector_config, state_factory):
    first, second = state_factory(), state_factory(threshold=-7.)
    save_detector_state(tmp_path, first)
    save_detector_state(tmp_path, second)
    rollback_detector_state(tmp_path, detector_config)
    history = artifacts.load_artifact(tmp_path / "history.json", expected_type="detector_state_history")
    assert len(history.records) == 1
    change = history.records[0]
    assert change.source == "rollback"
    assert change.old_version == second.version_id
    assert change.new_version == first.version_id
    assert change.promoted is True


def test_audit_failure_does_not_reject_rollback(tmp_path, detector_config, state_factory, monkeypatch, caplog):
    first, second = state_factory(), state_factory(threshold=-7.)
    save_detector_state(tmp_path, first)
    save_detector_state(tmp_path, second)
    original_write = artifacts._write_detector_artifact

    def fail_history(path, artifact, **kwargs):
        if path.name == "history.json":
            raise DataValidationError("simulated audit storage failure")
        return original_write(path, artifact, **kwargs)

    monkeypatch.setattr(artifacts, "_write_detector_artifact", fail_history)
    result = rollback_detector_state(tmp_path, detector_config)
    assert result.active_version == first.version_id
    assert load_detector_state(tmp_path, detector_config) == first
    assert "Cannot save detector update history" in caplog.text
