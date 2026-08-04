"""Pydantic schemas for processed packet windows and dataset metadata."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    field_validator,
    model_validator,
)

from .categories import validate_categories


class WindowRecord(BaseModel):
    """A timestamped packet-count vector with explicit category semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    device_id: str = Field(min_length=1)
    capture_id: str | None = Field(default=None, min_length=1)
    start_utc: datetime
    end_utc: datetime
    categories: tuple[str, ...]
    counts: tuple[NonNegativeInt, ...] | None
    state: Literal["observed", "observed-silent", "missing"]

    @field_validator("device_id", "capture_id")
    @classmethod
    def identifiers_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        identifier = value.strip()
        if not identifier:
            raise ValueError("identifier must not be blank")
        return identifier

    @field_validator("categories", mode="before")
    @classmethod
    def categories_must_use_supported_taxonomy(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        return validate_categories(value)

    @field_validator("start_utc", "end_utc")
    @classmethod
    def timestamps_must_be_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("window timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_window(self) -> "WindowRecord":
        if self.end_utc <= self.start_utc:
            raise ValueError("window end must be after window start")
        if self.state == "missing":
            if self.counts is not None:
                raise ValueError("missing windows cannot have counts")
            return self

        if self.counts is None:
            raise ValueError("non-missing windows require counts")
        if len(self.counts) != len(self.categories):
            raise ValueError("counts length must match categories length")

        if self.state == "observed" and self.total_count == 0:
            raise ValueError("observed windows require packet counts")
        if self.state == "observed-silent" and self.total_count != 0:
            raise ValueError("silent windows require zero counts")
        return self

    @property
    def total_count(self) -> int | None:
        """Return the packet total, or no value when coverage is missing."""
        return None if self.counts is None else sum(self.counts)


class Metadata(BaseModel):
    date: str
    file_source: str


class ProcessedDataset(BaseModel):
    metadata: Metadata
    windows: list[WindowRecord]
