"""Versioned schemas and explicit migrations for persisted artifacts."""

from __future__ import annotations

import re
from copy import deepcopy
from math import isclose, isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from lm_idnet.exceptions import ArtifactCompatibilityError

CURRENT_SCHEMA_VERSION = "1.2.0"
PREVIOUS_SCHEMA_VERSION = "1.1.0"
LEGACY_SCHEMA_VERSION = "1.0.0"
_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


class VersionedArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.2.0"] = CURRENT_SCHEMA_VERSION
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class FitDiagnostics(BaseModel):
    """Convergence details required to reproduce and assess a model fit."""

    model_config = ConfigDict(extra="forbid")

    initial_alpha: tuple[float, ...] = Field(min_length=2)
    iterations: int = Field(gt=0)
    converged: bool
    tolerance: float = Field(gt=0, allow_inf_nan=False)
    max_iterations: int = Field(gt=0)
    initial_log_likelihood: float = Field(allow_inf_nan=False)
    final_log_likelihood: float = Field(allow_inf_nan=False)
    duration_seconds: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def values_must_be_consistent(self) -> "FitDiagnostics":
        if any(not isfinite(value) or value <= 0 for value in self.initial_alpha):
            raise ValueError("initial alpha values must be finite and positive")
        if self.iterations > self.max_iterations:
            raise ValueError("iterations cannot exceed maximum iterations")
        return self


class ModelArtifact(VersionedArtifact):
    artifact_type: Literal["model"] = "model"
    model_version: str = Field(min_length=1)
    categories: tuple[str, ...] = Field(min_length=2)
    alpha: tuple[float, ...] = Field(min_length=2)
    concentration: float = Field(gt=0, allow_inf_nan=False)
    mean_probabilities: tuple[float, ...] = Field(min_length=2)
    psi: float = Field(gt=0, allow_inf_nan=False)
    training_capture_ids: tuple[str, ...]
    log_likelihood_backend: Literal["scipy", "lm"] | None
    fit_diagnostics: FitDiagnostics | None

    @model_validator(mode="after")
    def parameters_must_be_consistent(self) -> "ModelArtifact":
        category_count = len(self.categories)
        if len(self.alpha) != category_count:
            raise ValueError("alpha length must match categories length")
        if len(self.mean_probabilities) != category_count:
            raise ValueError(
                "mean probabilities length must match categories length"
            )
        if any(not isfinite(value) or value <= 0 for value in self.alpha):
            raise ValueError("alpha values must be finite and positive")

        expected_concentration = sum(self.alpha)
        if not isclose(
            self.concentration,
            expected_concentration,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("concentration must equal the sum of alpha")
        if not isclose(
            self.psi,
            1.0 / self.concentration,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("psi must be the reciprocal of concentration")

        expected_probabilities = tuple(
            value / self.concentration for value in self.alpha
        )
        if any(
            not isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)
            for actual, expected in zip(
                self.mean_probabilities,
                expected_probabilities,
            )
        ):
            raise ValueError("mean probabilities must equal alpha / concentration")

        if len(self.training_capture_ids) != len(set(self.training_capture_ids)):
            raise ValueError("training capture IDs must be unique")
        if any(not capture_id.strip() for capture_id in self.training_capture_ids):
            raise ValueError("training capture IDs must not be blank")

        metadata_fields_present = (
            bool(self.training_capture_ids),
            self.log_likelihood_backend is not None,
            self.fit_diagnostics is not None,
        )
        if any(metadata_fields_present) and not all(metadata_fields_present):
            raise ValueError(
                "training captures, likelihood backend, and fit diagnostics "
                "must be provided together"
            )
        if (
            self.fit_diagnostics is not None
            and len(self.fit_diagnostics.initial_alpha) != category_count
        ):
            raise ValueError("initial alpha length must match categories length")
        return self


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
    ModelArtifact
    | ThresholdArtifact
    | AnomalyEventArtifact
    | ExperimentManifestArtifact
)

SCHEMA_BY_TYPE: dict[str, type[VersionedArtifact]] = {
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


def _add_legacy_model_fields(model: dict[str, Any]) -> None:
    """Add derivable fields while marking unavailable fit metadata explicitly."""
    alpha = model.get("alpha")
    if not isinstance(alpha, (list, tuple)):
        return
    try:
        alpha_values = tuple(float(value) for value in alpha)
    except (TypeError, ValueError):
        return
    if not alpha_values or any(
        not isfinite(value) or value <= 0 for value in alpha_values
    ):
        return

    concentration = sum(alpha_values)
    model.setdefault("concentration", concentration)
    model.setdefault(
        "mean_probabilities",
        [value / concentration for value in alpha_values],
    )
    model.setdefault("psi", 1.0 / concentration)
    # Versions before 1.2.0 never persisted these values. Empty/null makes the
    # missing provenance visible instead of guessing it during migration.
    model.setdefault("training_capture_ids", [])
    model.setdefault("log_likelihood_backend", None)
    model.setdefault("fit_diagnostics", None)


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
    if version == LEGACY_SCHEMA_VERSION:
        migrated["artifact_type"] = expected_type
        version = PREVIOUS_SCHEMA_VERSION

    if version == PREVIOUS_SCHEMA_VERSION:
        if expected_type == "model":
            _add_legacy_model_fields(migrated)
        migrated["schema_version"] = CURRENT_SCHEMA_VERSION

        # The original checksum was verified before migration. Recalculate it
        # for the new in-memory representation after adding current fields.
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
