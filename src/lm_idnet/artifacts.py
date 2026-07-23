"""Integrity verification for artifacts consumed by trusted stages."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

from lm_idnet.exceptions import (
    ArtifactCompatibilityError,
    ArtifactIntegrityError,
)


def artifact_checksum(artifact: dict[str, Any]) -> str:
    """Return the SHA-256 digest of an artifact excluding its checksum field."""
    unsigned = {key: value for key, value in artifact.items() if key != "checksum"}
    canonical = json.dumps(
        unsigned,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_verified_artifact(
    path: str | Path,
    *,
    artifact_type: str = "artifact",
) -> dict[str, Any]:
    """Load an artifact only when it is readable, structured, and unmodified."""
    artifact_path = Path(path)
    try:
        raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ArtifactCompatibilityError(
            f"{artifact_type} not found: {artifact_path}"
        ) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactCompatibilityError(
            f"{artifact_type} cannot be read as JSON: {artifact_path}"
        ) from error

    if not isinstance(raw, dict):
        raise ArtifactCompatibilityError(
            f"{artifact_type} must be a JSON object: {artifact_path}"
        )

    supplied = raw.get("checksum")
    if not isinstance(supplied, str) or not supplied:
        raise ArtifactIntegrityError(
            f"{artifact_type} has no valid checksum: {artifact_path}"
        )

    try:
        expected = artifact_checksum(raw)
    except (TypeError, ValueError) as error:
        raise ArtifactCompatibilityError(
            f"{artifact_type} cannot be canonicalized: {artifact_path}"
        ) from error

    if not hmac.compare_digest(supplied, expected):
        raise ArtifactIntegrityError(
            f"{artifact_type} checksum mismatch: {artifact_path}"
        )
    return raw


def load_model_for_scoring(path: str | Path) -> dict[str, Any]:
    """Fail-closed model loading boundary for current and future scorers."""
    return load_verified_artifact(path, artifact_type="model")
