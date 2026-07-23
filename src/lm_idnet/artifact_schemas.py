"""Versioned schemas and explicit migrations for persisted artifacts."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lm_idnet.exceptions import ArtifactCompatibilityError
from lm_idnet.processing.schemas import Metadata, WindowCount

CURRENT_SCHEMA_VERSION = "1.1.0"
PREVIOUS_SCHEMA_VERSION = "1.0.0"
_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


class VersionedArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.1.0"] = CURRENT_SCHEMA_VERSION
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProcessedDatasetArtifact(VersionedArtifact):
    artifact_type: Literal["processed_dataset"] = "processed_dataset"
    metadata: Metadata
    windows: tuple[WindowCount, ...]


class ModelArtifact(VersionedArtifact):
    artifact_type: Literal["model"] = "model"
    model_version: str = Field(min_length=1)
    categories: tuple[str, ...] = Field(min_length=2)
    alpha: tuple[float, ...] = Field(min_length=2)


class ThresholdArtifact(VersionedArtifact):
    artifact_type: Literal["threshold"] = "threshold"
    model_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_type: str = Field(min_length=1)
    quantile: float = Field(gt=0, lt=1)
    threshold: float


class AnomalyEventArtifact(VersionedArtifact):
    artifact_type: Literal["anomaly_event"] = "anomaly_event"
    event_id: str = Field(min_length=1)
    window_id: str = Field(min_length=1)
    score: float
    threshold: float
    decision: bool
    model_version: str = Field(min_length=1)


class ExperimentManifestArtifact(VersionedArtifact):
    artifact_type: Literal["experiment_manifest"] = "experiment_manifest"
    run_id: str = Field(min_length=1)
    command: str = Field(min_length=1)
    code_commit: str = Field(min_length=1)
    configuration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int = Field(ge=0)


Artifact = (
    ProcessedDatasetArtifact
    | ModelArtifact
    | ThresholdArtifact
    | AnomalyEventArtifact
    | ExperimentManifestArtifact
)

SCHEMA_BY_TYPE: dict[str, type[VersionedArtifact]] = {
    "processed_dataset": ProcessedDatasetArtifact,
    "model": ModelArtifact,
    "threshold": ThresholdArtifact,
    "anomaly_event": AnomalyEventArtifact,
    "experiment_manifest": ExperimentManifestArtifact,
}


def parse_semantic_version(value: object) -> tuple[int, int, int]:
    """Parse a strict MAJOR.MINOR.PATCH schema version."""
    if not isinstance(value, str) or not (match := _SEMVER.fullmatch(value)):
        raise ArtifactCompatibilityError(
            f"invalid semantic schema version: {value!r}"
        )
    return tuple(int(part) for part in match.groups())


def migrate_artifact(
    raw: dict[str, Any],
    *,
    expected_type: str,
) -> dict[str, Any]:
    """Migrate a supported previous schema to the current representation."""
    migrated = deepcopy(raw)
    version = migrated.get("schema_version")
    parsed = parse_semantic_version(version)
    current = parse_semantic_version(CURRENT_SCHEMA_VERSION)

    if parsed[0] != current[0]:
        raise ArtifactCompatibilityError(
            f"unsupported schema major version {parsed[0]}; "
            f"supported major is {current[0]}"
        )
    if parsed > current:
        raise ArtifactCompatibilityError(
            f"schema version {version} is newer than supported "
            f"{CURRENT_SCHEMA_VERSION}"
        )
    if version == PREVIOUS_SCHEMA_VERSION:
        migrated["artifact_type"] = expected_type
        migrated["schema_version"] = CURRENT_SCHEMA_VERSION
        # The source checksum has already served to verify the old bytes. The
        # migrated in-memory representation receives a checksum for its new
        # canonical content.
        from lm_idnet.artifacts import artifact_checksum

        migrated["checksum"] = artifact_checksum(migrated)
        return migrated
    if version != CURRENT_SCHEMA_VERSION:
        raise ArtifactCompatibilityError(
            f"no migration from schema version {version}"
        )
    return migrated


def validate_versioned_artifact(
    raw: dict[str, Any],
    *,
    expected_type: str,
) -> Artifact:
    """Migrate and validate an artifact against its registered current schema."""
    schema = SCHEMA_BY_TYPE.get(expected_type)
    if schema is None:
        raise ArtifactCompatibilityError(
            f"unknown artifact type: {expected_type}"
        )
    migrated = migrate_artifact(raw, expected_type=expected_type)
    if migrated.get("artifact_type") != expected_type:
        raise ArtifactCompatibilityError(
            f"expected {expected_type}, found {migrated.get('artifact_type')!r}"
        )
    try:
        return schema.model_validate(migrated)
    except ValidationError as error:
        raise ArtifactCompatibilityError(
            f"{expected_type} does not match schema "
            f"{CURRENT_SCHEMA_VERSION}: {error}"
        ) from error
