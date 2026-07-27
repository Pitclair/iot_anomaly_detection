"""Normalize capture timestamps without losing packet-capture precision.

PCAP timestamps are Unix-epoch offsets and therefore represent UTC instants;
the capture format does not carry an original local time zone. Naive datetime
values supplied by other ingestion paths are also assumed to already be UTC.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from numbers import Real

import pandas as pd

CAPTURE_TIME_ZONE_ASSUMPTION = (
    "Numeric PCAP times are Unix-epoch UTC; naive datetime values are assumed UTC."
)
NANOSECONDS_PER_SECOND = Decimal("1000000000")


def normalize_utc_timestamp(value: object) -> pd.Timestamp:
    """Return a finite, timezone-aware UTC timestamp at nanosecond precision."""
    if isinstance(value, bool):
        raise ValueError("boolean values are not valid timestamps")

    if isinstance(value, (Real, Decimal)):
        return _normalize_epoch_seconds(value)

    return _normalize_datetime_like(value)


def _normalize_epoch_seconds(value: Real | Decimal) -> pd.Timestamp:
    """Convert numeric Unix seconds to an exact UTC pandas timestamp."""
    try:
        seconds = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"invalid epoch timestamp: {value!r}") from error

    if not seconds.is_finite():
        raise ValueError(f"timestamp must be finite: {value!r}")

    nanoseconds = seconds * NANOSECONDS_PER_SECOND
    integral_nanoseconds = nanoseconds.to_integral_value()
    if nanoseconds != integral_nanoseconds:
        raise ValueError(
            f"timestamp exceeds supported nanosecond precision: {value!r}"
        )

    try:
        return pd.Timestamp(int(integral_nanoseconds), unit="ns", tz="UTC")
    except (OverflowError, ValueError) as error:
        raise ValueError(f"timestamp is outside the supported range: {value!r}") from error


def _normalize_datetime_like(value: object) -> pd.Timestamp:
    """Normalize datetime-like input, applying the documented naive assumption."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid timestamp: {value!r}") from error

    if pd.isna(timestamp):
        raise ValueError(f"timestamp must be finite: {value!r}")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")
