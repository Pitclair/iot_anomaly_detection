"""Load and save validated JSON artifacts."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

from pydantic import BaseModel, ValidationError

from lm_idnet.config import AppConfig, DetectorInputConfig, EdgeConfig
from lm_idnet.evaluation.schemas import ExperimentManifestArtifact
from lm_idnet.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
    DetectorStateCommitError,
    PolicyRejectionError,
)
from lm_idnet.model_fingerprint import model_parameter_fingerprint
from lm_idnet.models.schemas import (
    DETECTOR_HISTORY_LIMIT,
    AnomalyEventArtifact,
    DetectorStateArtifact,
    DetectorStateIndex,
    DetectorStateChange,
    DetectorStateHistory,
    ModelArtifact,
    ThresholdArtifact,
)

logger = logging.getLogger(__name__)

Artifact = (
    ModelArtifact
    | ThresholdArtifact
    | AnomalyEventArtifact
    | ExperimentManifestArtifact
    | DetectorStateArtifact
    | DetectorStateIndex
    | DetectorStateHistory
)

SCHEMA_BY_TYPE: dict[str, type[BaseModel]] = {
    "model": ModelArtifact,
    "threshold": ThresholdArtifact,
    "anomaly_event": AnomalyEventArtifact,
    "experiment_manifest": ExperimentManifestArtifact,
    "detector_state": DetectorStateArtifact,
    "detector_state_index": DetectorStateIndex,
    "detector_state_history": DetectorStateHistory,
}


def artifact_fingerprint(artifact: ModelArtifact) -> str:
    """Fingerprint the model inputs that can change scoring."""
    return model_parameter_fingerprint(
        alpha=artifact.alpha,
        categories=artifact.categories,
        log_likelihood_backend=artifact.log_likelihood_backend,
        precision_digits=artifact.precision_digits,
        **artifact.input_contract.model_dump(exclude={"categories"}),
    )


def validate_artifact(raw: dict, *, expected_type: str) -> Artifact:
    """Route an artifact to the schema owned by its application layer."""
    schema = SCHEMA_BY_TYPE.get(expected_type)
    if schema is None:
        raise ArtifactCompatibilityError(f"unknown artifact type: {expected_type}")
    if raw.get("artifact_type") != expected_type:
        raise ArtifactCompatibilityError(
            f"expected {expected_type}, found {raw.get('artifact_type')!r}"
        )
    if expected_type == "model" and "input_contract" not in raw:
        raise ArtifactCompatibilityError(
            "model input contract is missing; retrain and recalibrate before deployment"
        )
    try:
        return schema.model_validate(raw)
    except ValidationError as error:
        raise ArtifactCompatibilityError(
            f"{expected_type} does not match its schema: {error}"
        ) from error


def load_artifact(path: str | Path, *, expected_type: str) -> Artifact:
    """Read a JSON artifact and validate its type and contents."""
    artifact_path = Path(path)
    try:
        raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ArtifactCompatibilityError(
            f"{expected_type} not found: {artifact_path}"
        ) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactCompatibilityError(
            f"{expected_type} cannot be read as JSON: {artifact_path}"
        ) from error

    if not isinstance(raw, dict):
        raise ArtifactCompatibilityError(
            f"{expected_type} must be a JSON object: {artifact_path}"
        )
    return validate_artifact(raw, expected_type=expected_type)


def load_model(config: AppConfig | EdgeConfig) -> ModelArtifact:
    """Load a model and reject provenance from another configured dataset."""
    model = cast(
        ModelArtifact,
        load_artifact(config.outputs.model_path, expected_type="model"),
    )
    _validate_model_configuration(model, config)
    return model


def _validate_model_configuration(model: ModelArtifact, config: AppConfig | EdgeConfig) -> None:
    for field in DetectorInputConfig.model_fields:
        if getattr(model.input_contract, field) != getattr(config.ingest, field):
            raise ArtifactCompatibilityError(
                f"model {field.replace('_', ' ')} does not match configuration"
            )
    if isinstance(config, EdgeConfig):
        expected_fields = {
            "precision_digits": config.precision_digits,
            "log_likelihood_backend": config.estimator.log_likelihood_backend,
        }
    else:
        expected_fields = {
            "dataset": config.ingest.dataset_folder,
            "training_capture_ids": config.ingest.partitions.fit,
        }
    for field, expected in expected_fields.items():
        actual = getattr(model, field)
        if actual != expected:
            raise ArtifactCompatibilityError(
                f"model {field.replace('_', ' ')} does not match configuration: "
                f"expected {expected!r}, found {actual!r}"
            )


def load_model_threshold(config: AppConfig | EdgeConfig) -> tuple[ModelArtifact, ThresholdArtifact]:
    """Load the same compatible pair for batch and one-window inference."""
    model = load_model(config)
    threshold = cast(ThresholdArtifact, load_artifact(
        config.outputs.threshold_path, expected_type="threshold",
    ))
    if threshold.model_fingerprint != artifact_fingerprint(model):
        raise ArtifactCompatibilityError("threshold model fingerprint does not match loaded model")
    if threshold.score_type != config.calibration.score_type:
        raise ArtifactCompatibilityError(
            f"threshold score type must match configured {config.calibration.score_type!r}; "
            f"found {threshold.score_type!r}"
        )
    return model, threshold


def save_artifact(path: str | Path, artifact: Artifact) -> None:
    """Write an already validated artifact as readable JSON."""
    artifact_path = Path(path)
    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            artifact.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise DataValidationError(
            f"cannot save {artifact.artifact_type} artifact: {artifact_path}"
        ) from error


def load_detector_version(directory: str | Path, version_id: str) -> DetectorStateArtifact:
    """Read one complete snapshot using a validated, path-safe identifier."""
    index = cast(DetectorStateIndex, validate_artifact(
        {"artifact_type": "detector_state_index", "active_version": version_id},
        expected_type="detector_state_index",
    ))
    path = Path(directory) / "versions" / f"{index.active_version}.json"
    state = cast(DetectorStateArtifact, load_artifact(path, expected_type="detector_state"))
    if state.version_id != version_id:
        raise ArtifactCompatibilityError("snapshot version does not match its filename")
    return state


def save_detector_version(directory: str | Path, state: DetectorStateArtifact) -> Path:
    """Store and reread a new snapshot without overwriting an existing version."""
    validated = cast(DetectorStateArtifact, validate_artifact(
        state.model_dump(mode="python"), expected_type="detector_state",
    ))
    path = Path(directory) / "versions" / f"{validated.version_id}.json"
    _write_detector_artifact(path, validated, replace=False)
    return path


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_detector_artifact(path: Path, artifact: Artifact, *, replace: bool) -> None:
    temporary_path = None
    published = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(artifact.model_dump_json(indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        if load_artifact(temporary_path, expected_type=artifact.artifact_type) != artifact:
            raise ArtifactCompatibilityError("written detector state differs from supplied state")
        if replace:
            os.replace(temporary_path, path)
        else:
            # Publish an immutable version without replacing a colliding identifier.
            os.link(temporary_path, path)
        published = True
        _sync_directory(path.parent)
    except OSError as error:
        if replace and published and path.name == "current.json":
            raise DetectorStateCommitError(
                f"detector state pointer changed but directory sync failed: {path}"
            ) from error
        raise DataValidationError(f"cannot save detector state file: {path}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Cannot remove detector temporary file: %s", temporary_path)


def _read_detector_index(directory: Path) -> DetectorStateIndex | None:
    path = directory / "current.json"
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ArtifactCompatibilityError(f"cannot inspect detector state index: {path}") from error
    return cast(DetectorStateIndex, load_artifact(path, expected_type="detector_state_index"))


@contextmanager
def detector_state_lock(directory: str | Path) -> Iterator[Path]:
    """Hold one nonblocking Linux writer lock for a state directory."""
    import fcntl

    directory = Path(directory).resolve()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        lock = (directory / ".lock").open("a")
    except OSError as error:
        raise DataValidationError(f"cannot open detector state lock: {directory}") from error
    with lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PolicyRejectionError(f"detector state is in use: {directory}") from error
        except OSError as error:
            raise DataValidationError(f"cannot lock detector state: {directory}") from error
        yield directory


def save_detector_state(directory: str | Path, state: DetectorStateArtifact) -> DetectorStateIndex:
    """Commit a complete version, retaining the previous active snapshot."""
    directory = Path(directory)
    validated = cast(DetectorStateArtifact, validate_artifact(
        state.model_dump(mode="python"), expected_type="detector_state",
    ))
    previous = _read_detector_index(directory)
    if previous is not None:
        active = load_detector_version(directory, previous.active_version)
        for field in (
            "baseline_model", "baseline_threshold", "window_minutes", "device_mac",
            "calibration", "adaptation", "estimator", "packet_count_limit",
        ):
            if getattr(active, field) != getattr(validated, field):
                raise ArtifactCompatibilityError(f"detector state {field} changed")
    save_detector_version(directory, validated)
    try:
        _sync_directory(directory.parent)
    except OSError as error:
        raise DataValidationError(f"cannot sync detector state directory: {directory}") from error
    index = DetectorStateIndex(
        active_version=validated.version_id,
        previous_version=previous.active_version if previous is not None else None,
    )
    _write_detector_artifact(directory / "current.json", index, replace=True)
    _prune_detector_state(directory, index)
    return index


def _prune_detector_state(directory: Path, index: DetectorStateIndex) -> None:
    """Remove unreferenced versions and abandoned files after validating references."""
    try:
        retained = {index.active_version}
        if index.previous_version is not None:
            retained.add(index.previous_version)
        for version_id in retained:
            load_detector_version(directory, version_id)
        versions = directory / "versions"
        changed_directories = set()
        for path in versions.iterdir():
            obsolete = re.fullmatch(r"[0-9a-f]{32}\.json", path.name) and path.stem not in retained
            abandoned = re.fullmatch(r"\.[0-9a-f]{32}\.json\..+\.tmp", path.name)
            if obsolete or abandoned:
                path.unlink()
                changed_directories.add(versions)
        for prefix in (".current.json.", ".history.json."):
            for path in directory.glob(f"{prefix}*.tmp"):
                path.unlink()
                changed_directories.add(directory)
        for changed in changed_directories:
            _sync_directory(changed)
    except (ArtifactCompatibilityError, OSError) as error:
        logger.warning("Cannot clean detector state directory %s: %s", directory, error)


def record_detector_update(
    directory: str | Path,
    *,
    source: str,
    old_version: str,
    promoted: bool,
    new_version: str | None = None,
    reason: str | None = None,
    after_window: int | None = None,
) -> None:
    """Keep recent decisions without letting audit failures undo committed state."""
    path = Path(directory) / "history.json"
    try:
        history = cast(DetectorStateHistory, load_artifact(
            path, expected_type="detector_state_history",
        )) if path.exists() else DetectorStateHistory()
        change = DetectorStateChange(
            source=source, old_version=old_version, new_version=new_version,
            promoted=promoted, after_window=after_window,
            reason=reason[:1024] if reason else (None if promoted else "update rejected"),
        )
        history = DetectorStateHistory(records=(
            *history.records[-(DETECTOR_HISTORY_LIMIT - 1):], change,
        ))
        _write_detector_artifact(path, history, replace=True)
    except (ArtifactCompatibilityError, DataValidationError, OSError, ValueError) as error:
        logger.warning("Cannot save detector update history %s: %s", path, error)


def _validate_detector_configuration(state: DetectorStateArtifact, config: AppConfig | EdgeConfig) -> None:
    _validate_model_configuration(state.baseline_model, config)
    expected_fields = {
        "window duration": (state.window_minutes, config.ingest.window_minutes),
        "device MAC": (state.device_mac, config.ingest.device_mac),
        "calibration settings": (state.calibration, config.calibration),
        "adaptation settings": (state.adaptation, config.adaptation),
        "estimator settings": (state.estimator, config.estimator),
        "precision": (state.baseline_model.precision_digits, config.precision_digits),
        "likelihood backend": (
            state.baseline_model.log_likelihood_backend,
            config.estimator.log_likelihood_backend,
        ),
    }
    if isinstance(config, AppConfig):
        expected_fields["calibration captures"] = (
            state.baseline_threshold.calibration_capture_ids, config.ingest.partitions.calibration,
        )
    for field, (actual, expected) in expected_fields.items():
        if actual != expected:
            raise ArtifactCompatibilityError(f"detector {field} does not match configuration")
    if state.baseline_threshold.calibration_window_count < config.calibration.minimum_samples:
        raise ArtifactCompatibilityError("baseline threshold has insufficient calibration windows")


def load_detector_state(directory: str | Path, config: AppConfig | EdgeConfig) -> DetectorStateArtifact:
    """Restore accepted state, or initialize and commit an empty state directory."""
    directory = Path(directory)
    index = _read_detector_index(directory)
    if index is not None:
        state = load_detector_version(directory, index.active_version)
        _validate_detector_configuration(state, config)
        _prune_detector_state(directory, index)
        return state
    try:
        if any(path.name != ".lock" for path in directory.iterdir()):
            raise ArtifactCompatibilityError("detector state directory has no active index")
    except FileNotFoundError:
        pass
    except OSError as error:
        raise ArtifactCompatibilityError(f"cannot inspect detector state directory: {directory}") from error
    if isinstance(config, EdgeConfig):
        raise ArtifactCompatibilityError(
            "edge detector state is not prepared; initialise it with the research configuration "
            "on the development machine before copying it to the gateway"
        )

    # Calibration imports artifact helpers; defer this import to initial setup.
    from lm_idnet.models.calibration_stage import load_calibration_packet_count_limit

    model, threshold = load_model_threshold(config)
    packet_count_limit = None
    if config.adaptation.mode != "static" and config.adaptation.policy == "conservative":
        packet_count_limit = load_calibration_packet_count_limit(config, model.categories)
    state = cast(DetectorStateArtifact, validate_artifact({
        "artifact_type": "detector_state", "source": "initial",
        "baseline_model": model.model_dump(), "baseline_threshold": threshold.model_dump(),
        "alpha": model.alpha,
        "threshold": {
            "model_fingerprint": threshold.model_fingerprint,
            "threshold": threshold.threshold, "score_iqr": threshold.score_iqr,
        },
        "window_minutes": config.ingest.window_minutes, "device_mac": config.ingest.device_mac,
        "calibration": config.calibration.model_dump(),
        "adaptation": config.adaptation.model_dump(), "estimator": config.estimator.model_dump(),
        "packet_count_limit": packet_count_limit,
    }, expected_type="detector_state"))
    _validate_detector_configuration(state, config)
    save_detector_state(directory, state)
    return state


def rollback_detector_state(directory: str | Path, config: AppConfig | EdgeConfig) -> DetectorStateIndex:
    """Validate the previous snapshot and atomically make it active again."""
    with detector_state_lock(directory) as directory:
        index = _read_detector_index(directory)
        if index is None or index.previous_version is None:
            raise PolicyRejectionError("no previous detector version is available")
        previous = load_detector_version(directory, index.previous_version)
        _validate_detector_configuration(previous, config)
        restored = DetectorStateIndex(
            active_version=index.previous_version, previous_version=index.active_version,
        )
        _write_detector_artifact(directory / "current.json", restored, replace=True)
        _prune_detector_state(directory, restored)
        record_detector_update(
            directory, source="rollback", old_version=index.active_version,
            new_version=restored.active_version, promoted=True,
        )
        return restored
