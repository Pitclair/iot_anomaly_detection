"""Load and save validated JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from lm_idnet.artifact_schemas import Artifact, validate_artifact
from lm_idnet.exceptions import ArtifactCompatibilityError, DataValidationError


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
