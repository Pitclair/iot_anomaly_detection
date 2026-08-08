"""Schemas for persisted model and result artifacts."""

from __future__ import annotations

from datetime import datetime
from math import isclose, isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from lm_idnet.exceptions import ArtifactCompatibilityError


class ArtifactBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


class ModelArtifact(ArtifactBase):
    artifact_type: Literal["model"] = "model"
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

        provenance_fields_present = (
            bool(self.training_capture_ids),
            self.log_likelihood_backend is not None,
            self.fit_diagnostics is not None,
        )
        if any(provenance_fields_present) and not all(provenance_fields_present):
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


class ThresholdArtifact(ArtifactBase):
    artifact_type: Literal["threshold"] = "threshold"
    score_type: str = Field(min_length=1)
    quantile: float = Field(gt=0, lt=1)
    threshold: float
    calibration_capture_ids: tuple[str, ...]
    calibration_window_count: int
    calibration_start_utc: datetime
    calibration_end_utc: datetime
    quantile_method: Literal["linear"]
    score_minimum: float
    score_median: float
    score_maximum: float


class AnomalyEventArtifact(ArtifactBase):
    artifact_type: Literal["anomaly_event"] = "anomaly_event"
    event_id: str = Field(min_length=1)
    window_id: str = Field(min_length=1)
    score: float
    threshold: float
    decision: bool


class ExperimentManifestArtifact(ArtifactBase):
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

SCHEMA_BY_TYPE: dict[str, type[ArtifactBase]] = {
    "model": ModelArtifact,
    "threshold": ThresholdArtifact,
    "anomaly_event": AnomalyEventArtifact,
    "experiment_manifest": ExperimentManifestArtifact,
}


def validate_artifact(raw: dict, *, expected_type: str) -> Artifact:
    """Validate an artifact against the requested schema."""
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
