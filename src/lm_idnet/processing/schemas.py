"""Pydantic schemas for processed packet windows and dataset metadata."""

from datetime import datetime, timezone
from typing import Any, Literal

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

    model_config = ConfigDict(extra="forbid")

    device_id: str = Field(min_length=1)
    start_utc: datetime
    end_utc: datetime
    categories: tuple[str, ...]
    counts: tuple[NonNegativeInt, ...] | None
    total_count: NonNegativeInt | None
    state: Literal["observed", "observed-silent", "missing"]
    metadata: dict[str, Any] = Field(default_factory=dict)

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
            if self.counts is not None or self.total_count is not None:
                raise ValueError("missing windows must not contain observed counts")
            return self

        if self.counts is None or self.total_count is None:
            raise ValueError("observed windows must contain counts and a total_count")
        if len(self.counts) != len(self.categories):
            raise ValueError("counts length must match categories length")
        if self.total_count != sum(self.counts):
            raise ValueError("total_count must equal the sum of counts")

        if self.state == "observed" and self.total_count == 0:
            raise ValueError("observed windows must contain at least one packet")
        if self.state == "observed-silent" and self.total_count != 0:
            raise ValueError("observed-silent windows must have zero counts")
        return self


class Metadata(BaseModel):
    date: str
    file_source: str


class ProcessedDataset(BaseModel):
    metadata: Metadata
    windows: list[WindowRecord]
