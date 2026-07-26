"""Strict, typed configuration for LM-IDNet."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from lm_idnet.exceptions import ConfigurationError


class StrictModel(BaseModel):
    """Base model that rejects misspelled and unsupported settings."""

    model_config = ConfigDict(extra="forbid")


class DuplicateCaptureGroup(StrictModel):
    capture_ids: tuple[str, ...] = Field(min_length=2)
    reason: str = Field(min_length=1)

    @field_validator("capture_ids")
    @classmethod
    def capture_ids_must_be_unique(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("duplicate explanation contains repeated capture IDs")
        return value

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank(cls, value: str) -> str:
        reason = value.strip()
        if not reason:
            raise ValueError("duplicate explanation reason must not be blank")
        return reason


class IngestConfig(StrictModel):
    raw_root: Path
    processed_root: Path
    dataset_folder: str = Field(min_length=1)
    time_col: str = Field(default="timestamp", min_length=1)
    protocol_col: str = Field(default="protocol", min_length=1)
    window_minutes: int = Field(gt=0)
    categories: tuple[str, ...] = Field(min_length=2)
    training_dates: tuple[str, ...] = Field(min_length=1)
    testing_dates: tuple[str, ...] = Field(min_length=1)
    allowed_duplicate_captures: tuple[DuplicateCaptureGroup, ...] = ()

    @field_validator("categories", mode="before")
    @classmethod
    def normalize_categories(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        normalized = tuple(str(category).strip().lower() for category in value)
        if any(not category for category in normalized):
            raise ValueError("categories must not contain empty names")
        if len(set(normalized)) != len(normalized):
            raise ValueError("categories must be unique after normalization")
        return normalized

    @field_validator("training_dates", "testing_dates")
    @classmethod
    def dates_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("date identifiers must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("date identifiers must be unique within a partition")
        return value

    @model_validator(mode="after")
    def partitions_must_be_disjoint(self) -> "IngestConfig":
        overlap = set(self.training_dates).intersection(self.testing_dates)
        if overlap:
            raise ValueError(
                "training_dates and testing_dates must be disjoint; overlap: "
                + ", ".join(sorted(overlap))
            )
        configured_ids = set(self.training_dates + self.testing_dates)
        explained_ids: set[str] = set()
        for group in self.allowed_duplicate_captures:
            unknown_ids = set(group.capture_ids) - configured_ids
            if unknown_ids:
                raise ValueError(
                    "duplicate explanation references unconfigured captures: "
                    + ", ".join(sorted(unknown_ids))
                )
            repeated_ids = explained_ids.intersection(group.capture_ids)
            if repeated_ids:
                raise ValueError(
                    "captures appear in more than one duplicate explanation: "
                    + ", ".join(sorted(repeated_ids))
                )
            explained_ids.update(group.capture_ids)
        return self


class SeedConfig(StrictModel):
    simulation: int = Field(ge=0)
    model_fitting: int = Field(ge=0)
    bootstrap: int = Field(ge=0)
    randomized_algorithms: int = Field(ge=0)


class EstimatorConfig(StrictModel):
    categories_k: int = Field(gt=1)
    tolerance_delta: float = Field(gt=0)
    max_iterations: int = Field(default=1000, gt=0)


class CalibrationConfig(StrictModel):
    quantile: float = Field(gt=0, lt=1)
    minimum_samples: int = Field(default=30, gt=0)


class AdaptationConfig(StrictModel):
    enabled: bool = False
    safe_margin: float = Field(default=0.0, ge=0)
    buffer_size: int = Field(default=1000, gt=0)


class OutputConfig(StrictModel):
    model_path: Path
    threshold_path: Path
    events_path: Path
    reports_dir: Path


class ForecastConfig(StrictModel):
    horizon_hours: int = Field(default=24, gt=0)


class AppConfig(StrictModel):
    """Complete application configuration with cross-section validation."""

    precision_digits: int = Field(default=6, gt=0)
    ingest: IngestConfig
    seeds: SeedConfig
    estimator: EstimatorConfig
    calibration: CalibrationConfig
    adaptation: AdaptationConfig
    outputs: OutputConfig
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)

    @model_validator(mode="after")
    def category_count_must_match(self) -> "AppConfig":
        actual = len(self.ingest.categories)
        if self.estimator.categories_k != actual:
            raise ValueError(
                f"estimator.categories_k ({self.estimator.categories_k}) "
                f"must equal the number of ingest.categories ({actual})"
            )
        return self


def load_config(path: str | Path) -> AppConfig:
    """Read and validate a JSON configuration file."""
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        return AppConfig.model_validate(raw)
    except FileNotFoundError as error:
        raise ConfigurationError(f"config not found: {config_path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"config is not readable JSON: {config_path}") from error
    except ValueError as error:
        raise ConfigurationError(f"config validation failed: {error}") from error
