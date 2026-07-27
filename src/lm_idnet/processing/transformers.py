"""Transform raw packet records into ten-minute count windows."""
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from .schemas import WindowCount
from .categories import CATEGORIES, validate_category_order
from .timestamps import normalize_utc_timestamp


def build_time_series(records: Iterable[Tuple[object, str]]) -> pd.DataFrame:
    """Build a timestamp-sorted UTC time series from packet records."""
    rows: list[dict[str, object]] = []
    for record_number, (timestamp_value, protocol) in enumerate(records, start=1):
        try:
            timestamp = normalize_utc_timestamp(timestamp_value)
        except ValueError as error:
            raise ValueError(f"invalid timestamp in record {record_number}") from error
        rows.append({"timestamp": timestamp, "protocol": protocol})

    if not rows:
        empty_index = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        return pd.DataFrame({"protocol": []}, index=empty_index)

    time_series = pd.DataFrame(rows).set_index("timestamp")
    return time_series.sort_index(kind="stable")


def to_10min_windows(df: pd.DataFrame, categories: List[str]) -> List[WindowCount]:
    """Resample df into 10-minute windows and return list of validated WindowCount objects.

    Handles silent windows by filling zeros for missing intervals.
    """
    category_order = validate_category_order(categories)

    if df.empty:
        # no packets at all -> return empty list
        return []

    # Sorting here also protects callers that construct a frame directly.
    df = df.sort_index()

    window = '10min'  # Corretto da '10T' a '10min' per evitare errori di frequenza
    # Resample counting per protocol
    grouped = df.groupby('protocol').resample(window).size().unstack(level=0).fillna(0)

    # reindex to continuous time range from first to last with freq=10min to include silent windows
    start = grouped.index.min()
    end = grouped.index.max()
    full_index = pd.date_range(start=start, end=end, freq=window)
    grouped = grouped.reindex(full_index, fill_value=0)

    # Ensure columns for all categories
    for c in category_order:
        if c not in grouped.columns:
            grouped[c] = 0

    grouped = grouped[list(category_order)].astype(int)

    # Convert rows to Pydantic WindowCount objects dynamically
    windows = []
    for _, row in grouped.iterrows():
        wc_data = {protocol.lower(): int(row[protocol]) for protocol in grouped.columns}
        wc = WindowCount(**wc_data)
        windows.append(wc)

    return windows


def to_numpy_matrix(windows: List[WindowCount]) -> np.ndarray:
    """Convert list of WindowCount into numpy array shape (R, K).
    Order is defined by the canonical category policy.
    """
    if not windows:
        return np.empty((0, len(CATEGORIES)), dtype=int)

    mat = np.array(
        [[getattr(window, category) for category in CATEGORIES] for window in windows],
        dtype=int,
    )
    return mat
