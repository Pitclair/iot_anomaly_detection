"""Transform classified packet records into fixed-window count data."""

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .categories import validate_category_order
from .schemas import WindowCount
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
        window_minutes: int = 10,
    ) -> None:
        self.categories = validate_category_order(list(categories))
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

    def to_windows(self, time_series: pd.DataFrame) -> list[WindowCount]:
        """Count classified packets in continuous fixed-duration windows."""
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

        return [
            WindowCount(**row.to_dict())
            for _, row in grouped.iterrows()
        ]

    def to_numpy_matrix(self, windows: Sequence[WindowCount]) -> np.ndarray:
        """Convert window counts to a matrix in canonical category order."""
        if not windows:
            return np.empty((0, len(self.categories)), dtype=int)

        return np.array(
            [
                [getattr(window, category) for category in self.categories]
                for window in windows
            ],
            dtype=int,
        )

    @staticmethod
    def _include_silent_windows(
        grouped: pd.DataFrame,
        frequency: str,
    ) -> pd.DataFrame:
        """Insert zero rows for unobserved intervals between packet windows."""
        full_index = pd.date_range(
            start=grouped.index.min(),
            end=grouped.index.max(),
            freq=frequency,
        )
        return grouped.reindex(full_index, fill_value=0)
