"""Load and save validated JSON artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from lm_idnet.evaluation.schemas import ExperimentManifestArtifact
from lm_idnet.exceptions import ArtifactCompatibilityError, DataValidationError
from lm_idnet.models.schemas import (
    AnomalyEventArtifact,
    ModelArtifact,
    ThresholdArtifact,
)

Artifact = (
    ModelArtifact
    | ThresholdArtifact
    | AnomalyEventArtifact
    | ExperimentManifestArtifact
)

SCHEMA_BY_TYPE: dict[str, type[BaseModel]] = {
    "model": ModelArtifact,
    "threshold": ThresholdArtifact,
    "anomaly_event": AnomalyEventArtifact,
    "experiment_manifest": ExperimentManifestArtifact,
}


def artifact_fingerprint(artifact: ModelArtifact) -> str:
    """Fingerprint the model inputs that can change scoring."""
    canonical = json.dumps(
        {
            "alpha": artifact.alpha,
            "categories": artifact.categories,
            "log_likelihood_backend": artifact.log_likelihood_backend,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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
