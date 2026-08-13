"""Schemas for training, calibration, and scoring artifacts."""

from datetime import datetime
from math import isclose, isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator


class ModelSchema(BaseModel):
    """Reject fields that are not part of a model-layer artifact."""

    model_config = ConfigDict(extra="forbid")


class FitDiagnostics(ModelSchema):
    """Convergence details required to reproduce and assess a model fit."""

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


class ModelArtifact(ModelSchema):
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


class ThresholdArtifact(ModelSchema):
    artifact_type: Literal["threshold"] = "threshold"
    model_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_type: Literal["raw", "normalized"]
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
    score_iqr: float = Field(gt=0, allow_inf_nan=False)


class AnomalyEventArtifact(ModelSchema):
    artifact_type: Literal["anomaly_event"] = "anomaly_event"
    device_id: str = Field(min_length=1)
    capture_id: str = Field(min_length=1)
    window_start_utc: datetime
    window_end_utc: datetime
    counts: dict[str, NonNegativeInt] = Field(min_length=2)
    score_type: Literal["raw", "normalized"]
    score: float = Field(allow_inf_nan=False)
    threshold: float = Field(allow_inf_nan=False)
    is_anomaly: bool
    severity: float = Field(ge=0, allow_inf_nan=False)
    expected_profile: dict[str, float] = Field(min_length=2)
    category_residuals: dict[str, float] = Field(min_length=2)
    model_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def values_must_be_consistent(self) -> "AnomalyEventArtifact":
        categories = self.counts.keys()
        if (
            self.expected_profile.keys() != categories
            or self.category_residuals.keys() != categories
        ):
            raise ValueError("event category fields must have matching keys")
        if self.window_end_utc <= self.window_start_utc:
            raise ValueError("window end must be after window start")
        if any(value < 0 for value in self.expected_profile.values()) or not isclose(
            sum(self.expected_profile.values()), 1.0
        ):
            raise ValueError("expected profile must sum to one")
        total = sum(self.counts.values())
        if any(
            not isclose(
                self.category_residuals[category],
                (self.counts[category] / total if total else 0.0) - expected,
            )
            for category, expected in self.expected_profile.items()
        ):
            raise ValueError("category residuals must equal observed minus expected")
        if self.is_anomaly != (self.score < self.threshold):
            raise ValueError("anomaly decision must match score and threshold")
        return self
