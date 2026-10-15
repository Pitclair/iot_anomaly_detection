"""Load and save validated JSON artifacts."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ValidationError

from lm_idnet.evaluation.schemas import ExperimentManifestArtifact
from lm_idnet.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
    DetectorStateCommitError,
)
from lm_idnet.model_fingerprint import model_parameter_fingerprint
from lm_idnet.models.schemas import (
    AnomalyEventArtifact,
    DetectorStateArtifact,
    DetectorStateIndex,
    ModelArtifact,
    ThresholdArtifact,
)

if TYPE_CHECKING:
    from lm_idnet.config import AppConfig

logger = logging.getLogger(__name__)

Artifact = (
    ModelArtifact
    | ThresholdArtifact
    | AnomalyEventArtifact
    | ExperimentManifestArtifact
    | DetectorStateArtifact
    | DetectorStateIndex
)

SCHEMA_BY_TYPE: dict[str, type[BaseModel]] = {
    "model": ModelArtifact,
    "threshold": ThresholdArtifact,
    "anomaly_event": AnomalyEventArtifact,
    "experiment_manifest": ExperimentManifestArtifact,
    "detector_state": DetectorStateArtifact,
    "detector_state_index": DetectorStateIndex,
}


def artifact_fingerprint(artifact: ModelArtifact) -> str:
    """Fingerprint the model inputs that can change scoring."""
    return model_parameter_fingerprint(
        alpha=artifact.alpha,
        categories=artifact.categories,
        log_likelihood_backend=artifact.log_likelihood_backend,
        precision_digits=artifact.precision_digits,
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


def load_model(config: "AppConfig") -> ModelArtifact:
    """Load a model and reject provenance from another configured dataset."""
    model = cast(
        ModelArtifact,
        load_artifact(config.outputs.model_path, expected_type="model"),
    )
    expected_fields = {
        "dataset": config.ingest.dataset_folder,
        "device_id": config.ingest.device_id,
        "categories": config.ingest.categories,
        "training_capture_ids": config.ingest.partitions.fit,
    }
    for field, expected in expected_fields.items():
        actual = getattr(model, field)
        if actual != expected:
            raise ArtifactCompatibilityError(
                f"model {field.replace('_', ' ')} does not match configuration: "
                f"expected {expected!r}, found {actual!r}"
            )
    return model


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
        if replace and published:
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
    return index
