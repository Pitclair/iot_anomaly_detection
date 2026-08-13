"""Strict, typed configuration for LM-IDNet."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from lm_idnet.exceptions import ConfigurationError
from lm_idnet.processing.categories import validate_categories
from lm_idnet.processing.window_policy import validate_window_minutes

_CAPTURE_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
_MAC_ADDRESS_PATTERN = re.compile(r"^(?:[0-9a-f]{2}:){5}[0-9a-f]{2}$")


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


def _date_from_capture_id(capture_id: str) -> date:
    match = _CAPTURE_DATE_PATTERN.search(capture_id)
    if match is None:
        raise ValueError(
            f"capture identifier has no YYYY-MM-DD date: {capture_id}"
        )
    try:
        return date.fromisoformat(match.group(1))
    except ValueError as error:
        raise ValueError(f"capture identifier has an invalid date: {capture_id}") from error


class TemporalPartitions(StrictModel):
    """Chronological capture-level partitions for the research protocol."""

    fit: tuple[str, ...] = Field(min_length=1)
    calibration: tuple[str, ...] = Field(min_length=1)
    development_test: tuple[str, ...] = Field(min_length=1)
    final_test: tuple[str, ...] = Field(min_length=1)

    @field_validator("fit", "calibration", "development_test", "final_test")
    @classmethod
    def partition_must_be_unique_and_chronological(
        cls,
        capture_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(set(capture_ids)) != len(capture_ids):
            raise ValueError("capture identifiers must be unique within a partition")
        capture_dates = [_date_from_capture_id(capture_id) for capture_id in capture_ids]
        if capture_dates != sorted(capture_dates):
            raise ValueError("capture identifiers must be in chronological order")
        return capture_ids

    @model_validator(mode="after")
    def partitions_must_be_disjoint_and_ordered(self) -> "TemporalPartitions":
        named_partitions = (
            ("fit", self.fit),
            ("calibration", self.calibration),
            ("development_test", self.development_test),
            ("final_test", self.final_test),
        )
        owner_by_capture: dict[str, str] = {}
        for partition_name, capture_ids in named_partitions:
            for capture_id in capture_ids:
                previous_owner = owner_by_capture.get(capture_id)
                if previous_owner is not None:
                    raise ValueError(
                        f"capture {capture_id} appears in both "
                        f"{previous_owner} and {partition_name}"
                    )
                owner_by_capture[capture_id] = partition_name

        # Entire partitions, not individual windows, move forward in time.
        for (earlier_name, earlier_ids), (later_name, later_ids) in zip(
            named_partitions,
            named_partitions[1:],
        ):
            if _date_from_capture_id(earlier_ids[-1]) >= _date_from_capture_id(
                later_ids[0]
            ):
                raise ValueError(
                    f"{earlier_name} must end before {later_name} begins"
                )
        return self

    def all_capture_ids(self) -> tuple[str, ...]:
        return self.fit + self.calibration + self.development_test + self.final_test


class IngestConfig(StrictModel):
    raw_root: Path
    processed_root: Path
    dataset_folder: str = Field(min_length=1)
    device_name: str = Field(min_length=1)
    device_mac: str
    time_col: str = Field(default="timestamp", min_length=1)
    protocol_col: str = Field(default="protocol", min_length=1)
    window_minutes: int
    categories: tuple[str, ...] = Field(min_length=2)
    partitions: TemporalPartitions
    allowed_duplicate_captures: tuple[DuplicateCaptureGroup, ...] = ()

    @field_validator("window_minutes", mode="before")
    @classmethod
    def window_minutes_must_be_positive(cls, value: object) -> int:
        return validate_window_minutes(value)

    @field_validator("categories", mode="before")
    @classmethod
    def normalize_categories(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        return validate_categories(value)

    @field_validator("device_name")
    @classmethod
    def device_name_must_not_be_blank(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("device_name must not be blank")
        return name

    @field_validator("device_mac")
    @classmethod
    def normalize_device_mac(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _MAC_ADDRESS_PATTERN.fullmatch(normalized):
            raise ValueError("device_mac must be a colon-separated MAC address")
        return normalized

    @model_validator(mode="after")
    def duplicate_explanations_must_reference_configured_captures(
        self,
    ) -> "IngestConfig":
        configured_ids = set(self.partitions.all_capture_ids())
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
    initial_alpha_concentration: float = Field(gt=0)
    tolerance_delta: float = Field(gt=0)
    max_iterations: int = Field(default=1000, gt=0)
    log_likelihood_backend: Literal["scipy", "lm"] = "scipy"


class CalibrationConfig(StrictModel):
    quantile: float = Field(gt=0, lt=1)
    minimum_samples: int = Field(default=30, gt=0)
    score_type: Literal["raw", "normalized"] = "raw"


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
