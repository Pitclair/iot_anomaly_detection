"""Transform classified packet records into fixed-window count data."""

from collections.abc import Iterable, Sequence
from datetime import datetime

import numpy as np
import pandas as pd

from .categories import validate_categories
from .schemas import WindowRecord
from .timestamps import normalize_utc_timestamp
from .window_policy import (
    WINDOW_CLOSED,
    WINDOW_LABEL,
    WINDOW_ORIGIN,
    validate_window_minutes,
)


class PacketTransformer:
    """Build UTC time series and epoch-aligned half-open count windows."""

    def __init__(
        self,
        categories: Sequence[str],
        device_id: str,
        window_minutes: int = 10,
    ) -> None:
        self.categories = validate_categories(categories)
        self.device_id = device_id.strip()
        if not self.device_id:
            raise ValueError("device_id must not be empty")
        self.window_minutes = validate_window_minutes(window_minutes)

    def build_time_series(
        self,
        records: Iterable[tuple[object, str]],
    ) -> pd.DataFrame:
        """Build a timestamp-sorted UTC time series from packet records."""
        rows: list[dict[str, object]] = []
        for record_number, record in enumerate(records, start=1):
            timestamp_value, protocol = record
            try:
                timestamp = normalize_utc_timestamp(timestamp_value)
            except ValueError as error:
                raise ValueError(
                    f"invalid timestamp in record {record_number}"
                ) from error
            rows.append({"timestamp": timestamp, "protocol": protocol})

        if not rows:
            empty_index = pd.DatetimeIndex([], name="timestamp", tz="UTC")
            return pd.DataFrame({"protocol": []}, index=empty_index)

        time_series = pd.DataFrame(rows).set_index("timestamp")
        return time_series.sort_index(kind="stable")

    def to_windows(
        self,
        time_series: pd.DataFrame,
        metadata: dict[str, object] | None = None,
        capture_discontinuities: Iterable[tuple[datetime | str, datetime | str]] = (),
    ) -> list[WindowRecord]:
        """Count packets while preserving declared gaps as missing coverage."""
        if time_series.empty:
            return []
        if "protocol" not in time_series.columns:
            raise ValueError("time series must contain a 'protocol' column")
        if not isinstance(time_series.index, pd.DatetimeIndex):
            raise ValueError("time series must use a DatetimeIndex")

        sorted_series = time_series.sort_index(kind="stable")
        frequency = f"{self.window_minutes}min"
        grouped = (
            sorted_series.groupby("protocol")
            .resample(
                frequency,
                origin=WINDOW_ORIGIN,
                closed=WINDOW_CLOSED,
                label=WINDOW_LABEL,
            )
            .size()
            .unstack(level=0)
            .fillna(0)
        )
        grouped = self._include_silent_windows(grouped, frequency)
        grouped = grouped.reindex(columns=self.categories, fill_value=0).astype(int)

        window_metadata = dict(metadata or {})
        duration = pd.Timedelta(minutes=self.window_minutes)
        missing_starts = self._missing_window_starts(
            capture_discontinuities,
            duration,
            grouped.index,
        )
        for missing_start in missing_starts:
            if int(grouped.loc[missing_start].sum()) > 0:
                raise ValueError(
                    "capture discontinuity overlaps a window containing packets"
                )
        return [
            self._build_window_record(
                start,
                row,
                duration,
                window_metadata,
                is_missing=start in missing_starts,
            )
            for start, row in grouped.iterrows()
        ]

    def to_numpy_matrix(self, windows: Sequence[WindowRecord]) -> np.ndarray:
        """Convert window counts to a matrix in canonical category order."""
        if not windows:
            return np.empty((0, len(self.categories)), dtype=int)

        for window in windows:
            if window.categories != self.categories:
                raise ValueError(
                    "window categories do not match the configured category order"
                )

        observed_counts = [
            window.counts for window in windows if window.state != "missing"
        ]
        if not observed_counts:
            return np.empty((0, len(self.categories)), dtype=int)
        return np.array(observed_counts, dtype=int)

    @staticmethod
    def _missing_window_starts(
        discontinuities: Iterable[tuple[datetime | str, datetime | str]],
        duration: pd.Timedelta,
        window_index: pd.DatetimeIndex,
    ) -> set[pd.Timestamp]:
        """Validate gaps and return the covered window starts they remove."""
        missing_starts: set[pd.Timestamp] = set()
        for gap_start_value, gap_end_value in discontinuities:
            gap_start = pd.Timestamp(normalize_utc_timestamp(gap_start_value))
            gap_end = pd.Timestamp(normalize_utc_timestamp(gap_end_value))
            if gap_end <= gap_start:
                raise ValueError("capture discontinuity end must be after its start")

            for window_start in window_index:
                window_end = window_start + duration
                if gap_start < window_end and gap_end > window_start:
                    missing_starts.add(window_start)
        return missing_starts

    @staticmethod
    def _include_silent_windows(
        grouped: pd.DataFrame,
        frequency: str,
    ) -> pd.DataFrame:
        """Insert candidate rows; declared discontinuities are marked later."""
        full_index = pd.date_range(
            start=grouped.index.min(),
            end=grouped.index.max(),
            freq=frequency,
        )
        return grouped.reindex(full_index, fill_value=0)

    def _build_window_record(
        self,
        start: pd.Timestamp,
        row: pd.Series,
        duration: pd.Timedelta,
        metadata: dict[str, object],
        is_missing: bool,
    ) -> WindowRecord:
        if is_missing:
            return WindowRecord(
                device_id=self.device_id,
                start_utc=start,
                end_utc=start + duration,
                categories=self.categories,
                counts=None,
                total_count=None,
                state="missing",
                metadata=metadata,
            )
        counts = tuple(int(row[category]) for category in self.categories)
        total_count = sum(counts)
        state = "observed" if total_count else "observed-silent"
        return WindowRecord(
            device_id=self.device_id,
            start_utc=start,
            end_utc=start + duration,
            categories=self.categories,
            counts=counts,
            total_count=total_count,
            state=state,
            metadata=metadata,
        )
