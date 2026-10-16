"""Schemas for training, calibration, and scoring artifacts."""

from datetime import datetime, timezone
from math import isclose, isfinite
from typing import Literal
from uuid import uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    StrictInt,
    model_validator,
)

from lm_idnet.config import (
    AdaptationConfig, CalibrationConfig, DetectorInputConfig, EstimatorConfig,
)
from lm_idnet.model_fingerprint import model_parameter_fingerprint


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
    schema_version: Literal[2] = 2
    dataset: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    categories: tuple[str, ...] = Field(min_length=2)
    input_contract: DetectorInputConfig
    alpha: tuple[float, ...] = Field(min_length=2)
    concentration: float = Field(gt=0, allow_inf_nan=False)
    mean_probabilities: tuple[float, ...] = Field(min_length=2)
    psi: float = Field(gt=0, allow_inf_nan=False)
    log_likelihood_backend: Literal["scipy", "lm"]
    precision_digits: StrictInt = Field(gt=0)
    training_capture_ids: tuple[str, ...] = Field(min_length=1)
    training_window_count: StrictInt = Field(gt=0)
    training_start_utc: AwareDatetime
    training_end_utc: AwareDatetime
    fit_diagnostics: FitDiagnostics

    @model_validator(mode="after")
    def parameters_must_be_consistent(self) -> "ModelArtifact":
        if (
            self.device_id != self.input_contract.device_id
            or self.categories != self.input_contract.categories
        ):
            raise ValueError("model device/categories must match its input contract")
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

        if not self.dataset.strip() or not self.device_id.strip():
            raise ValueError("dataset and device ID must not be blank")
        if len(self.training_capture_ids) != len(set(self.training_capture_ids)):
            raise ValueError("training capture IDs must be unique")
        if any(not capture_id.strip() for capture_id in self.training_capture_ids):
            raise ValueError("training capture IDs must not be blank")
        if self.training_end_utc <= self.training_start_utc:
            raise ValueError("training end must be after training start")
        if len(self.fit_diagnostics.initial_alpha) != category_count:
            raise ValueError("initial alpha length must match categories length")
        return self


class CountTailThreshold(ModelSchema):
    """An upper tail of log1p packet or ARP counts from normal windows."""

    threshold: float = Field(ge=0, allow_inf_nan=False)
    score_iqr: float = Field(gt=0, allow_inf_nan=False)


class ThresholdArtifact(ModelSchema):
    artifact_type: Literal["threshold"] = "threshold"
    model_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_type: Literal["raw", "normalized"]
    quantile: float = Field(gt=0, lt=1)
    threshold: float = Field(allow_inf_nan=False)
    calibration_capture_ids: tuple[str, ...]
    calibration_window_count: int
    calibration_start_utc: datetime
    calibration_end_utc: datetime
    quantile_method: Literal["linear"]
    score_minimum: float = Field(allow_inf_nan=False)
    score_median: float = Field(allow_inf_nan=False)
    score_maximum: float = Field(allow_inf_nan=False)
    score_iqr: float = Field(gt=0, allow_inf_nan=False)
    count_tails: dict[str, CountTailThreshold] = Field(default_factory=dict)

    @model_validator(mode="after")
    def count_features_must_be_complete(self) -> "ThresholdArtifact":
        if self.count_tails and self.count_tails.keys() != {"total", "arp"}:
            raise ValueError("count tails must cover total packets and ARP")
        return self


class DetectorThresholdState(ModelSchema):
    """An adaptive threshold bound to the parameters that produced its scores."""

    model_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    threshold: float = Field(allow_inf_nan=False)
    score_iqr: float = Field(gt=0, allow_inf_nan=False)


class DetectorStateArtifact(ModelSchema):
    """Current scoring state and the immutable references used for learning."""

    artifact_type: Literal["detector_state"] = "detector_state"
    schema_version: Literal[2] = 2
    version_id: str = Field(default_factory=lambda: uuid4().hex, pattern=r"^[0-9a-f]{32}$")
    created_utc: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: Literal["initial", "adaptive_threshold", "periodic_refit"]
    baseline_model: ModelArtifact
    baseline_threshold: ThresholdArtifact
    alpha: tuple[float, ...] = Field(min_length=2)
    threshold: DetectorThresholdState
    window_minutes: StrictInt = Field(gt=0)
    device_mac: str = Field(pattern=r"^(?:[0-9a-f]{2}:){5}[0-9a-f]{2}$")
    calibration: CalibrationConfig
    adaptation: AdaptationConfig
    estimator: EstimatorConfig
    packet_count_limit: float | None = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def state_must_be_consistent(self) -> "DetectorStateArtifact":
        model = self.baseline_model
        if (
            self.window_minutes != model.input_contract.window_minutes
            or self.device_mac != model.input_contract.device_mac
        ):
            raise ValueError("detector state must match the model input contract")
        if len(self.alpha) != len(model.categories) or any(
            not isfinite(value) or value <= 0 for value in self.alpha
        ) or not isfinite(sum(self.alpha)):
            raise ValueError("active alpha must be finite, positive, and match categories")
        for alpha, threshold in (
            (model.alpha, self.baseline_threshold),
            (self.alpha, self.threshold),
        ):
            fingerprint = model_parameter_fingerprint(
                alpha=alpha,
                categories=model.categories,
                log_likelihood_backend=model.log_likelihood_backend,
                precision_digits=model.precision_digits,
                **model.input_contract.model_dump(exclude={"categories"}),
            )
            if threshold.model_fingerprint != fingerprint:
                raise ValueError("threshold model fingerprint does not match parameters")
        if any(not isfinite(value) for value in (
            self.baseline_threshold.threshold,
            self.baseline_threshold.score_minimum,
            self.baseline_threshold.score_median,
            self.baseline_threshold.score_maximum,
        )):
            raise ValueError("baseline threshold statistics must be finite")
        if (
            self.baseline_threshold.score_type != self.calibration.score_type
            or self.baseline_threshold.quantile != self.calibration.quantile
        ):
            raise ValueError("baseline threshold does not match calibration settings")
        needs_volume_limit = (
            self.adaptation.mode != "static" and self.adaptation.policy == "conservative"
        )
        if needs_volume_limit != (self.packet_count_limit is not None):
            raise ValueError("packet count limit must match adaptation policy")
        if self.adaptation.mode != "static" and (
            self.adaptation.buffer_size < self.calibration.minimum_samples
            or (self.adaptation.mode == "periodic_refit"
                and self.adaptation.minimum_refit_windows > self.adaptation.buffer_size)
        ):
            raise ValueError("adaptation buffer is too small")
        return self


class DetectorStateIndex(ModelSchema):
    """The active snapshot and its previous accepted version."""

    artifact_type: Literal["detector_state_index"] = "detector_state_index"
    schema_version: Literal[1] = 1
    active_version: str = Field(pattern=r"^[0-9a-f]{32}$")
    previous_version: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")

    @model_validator(mode="after")
    def versions_must_differ(self) -> "DetectorStateIndex":
        if self.active_version == self.previous_version:
            raise ValueError("active and previous versions must differ")
        return self


DETECTOR_HISTORY_LIMIT = 100


class DetectorStateChange(ModelSchema):
    """A bounded record of an adaptive decision or operator rollback."""

    created_utc: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: Literal["adaptive_threshold", "periodic_refit", "rollback"]
    old_version: str = Field(pattern=r"^[0-9a-f]{32}$")
    new_version: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    promoted: bool
    reason: str | None = Field(default=None, min_length=1, max_length=1024)
    after_window: StrictInt | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def outcome_must_be_consistent(self) -> "DetectorStateChange":
        if self.promoted != (self.new_version is not None):
            raise ValueError("accepted changes must identify the new version")
        if not self.promoted and not self.reason:
            raise ValueError("rejected changes must record a reason")
        return self


class DetectorStateHistory(ModelSchema):
    artifact_type: Literal["detector_state_history"] = "detector_state_history"
    schema_version: Literal[1] = 1
    records: tuple[DetectorStateChange, ...] = Field(default=(), max_length=DETECTOR_HISTORY_LIMIT)


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
    composition_score: float | None = Field(default=None, allow_inf_nan=False)
    feature_margins: dict[str, float] = Field(default_factory=dict)

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
        if self.feature_margins:
            if self.feature_margins.keys() != {"composition", "total", "arp"}:
                raise ValueError("feature margins must cover composition, total packets, and ARP")
            if self.composition_score is None or any(
                not isfinite(value) for value in self.feature_margins.values()
            ):
                raise ValueError("feature scores must be finite")
            if self.threshold != 0.0 or self.score != min(self.feature_margins.values()):
                raise ValueError("combined score must equal the minimum feature margin")
        return self
