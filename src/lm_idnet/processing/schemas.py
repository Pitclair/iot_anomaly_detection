"""Pydantic schemas for processed packet windows and dataset metadata."""

from datetime import datetime, timezone
import re
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from .categories import SSDP_IS_EXCLUSIVE_OF_UDP, SSDP_PORT, validate_categories
from .window_policy import WINDOW_CLOSED, WINDOW_LABEL, WINDOW_ORIGIN

_MAC_ADDRESS_PATTERN = re.compile(r"^(?:[0-9a-f]{2}:){5}[0-9a-f]{2}$")


def normalize_device_mac(value: str) -> str:
    """Normalize and validate the device filter at configuration boundaries."""
    normalized = value.strip().lower()
    if not _MAC_ADDRESS_PATTERN.fullmatch(normalized):
        raise ValueError("device_mac must be a colon-separated MAC address")
    return normalized


class InputPolicy(BaseModel):
    """The device filter and counting rules used to produce model inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    device_mac: str | None
    window_minutes: StrictInt = Field(gt=0)
    ssdp_port: StrictInt = Field(gt=0)
    ssdp_is_exclusive_of_udp: StrictBool
    window_origin: str
    window_closed: str
    window_label: str

    @field_validator("device_mac")
    @classmethod
    def device_mac_must_be_valid(cls, value: str | None) -> str | None:
        return None if value is None else normalize_device_mac(value)

    @classmethod
    def current(cls, device_mac: str | None, window_minutes: int) -> "InputPolicy":
        """Describe the configured filter and the processing code's rules."""
        return cls(
            device_mac=device_mac,
            window_minutes=window_minutes,
            ssdp_port=SSDP_PORT,
            ssdp_is_exclusive_of_udp=SSDP_IS_EXCLUSIVE_OF_UDP,
            window_origin=WINDOW_ORIGIN,
            window_closed=WINDOW_CLOSED,
            window_label=WINDOW_LABEL,
        )


class WindowRecord(BaseModel):
    """A timestamped packet-count vector with explicit category semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_utc: datetime
    end_utc: datetime
    categories: tuple[str, ...]
    counts: tuple[Annotated[StrictInt, Field(ge=0)], ...] | None
    state: Literal["observed", "observed-silent", "missing"]

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
    model_config = ConfigDict(extra="forbid", frozen=True)

    device_id: str = Field(min_length=1)
    input_policy: InputPolicy
    capture_id: str = Field(min_length=1)
    partition: Literal["fit", "calibration", "development_test", "final_test"]
    date: str
    file_source: str

    @field_validator("device_id", "capture_id")
    @classmethod
    def identifiers_must_not_be_blank(cls, value: str) -> str:
        identifier = value.strip()
        if not identifier:
            raise ValueError("identifier must not be blank")
        return identifier


class ProcessedDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: Metadata
    windows: tuple[WindowRecord, ...]
