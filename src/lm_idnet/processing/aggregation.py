"""Aggregate packet traces into fixed windows and count vectors."""
import pandas as pd

from .categories import CATEGORIES, validate_category_order
from .window_policy import (
    WINDOW_CLOSED,
    WINDOW_LABEL,
    WINDOW_ORIGIN,
    validate_window_minutes,
)


def aggregate_packet_traces(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    protocol_col: str = "protocol",
    window_minutes: int = 10,
    categories: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Aggregate raw packet rows into time windows and count occurrences per protocol category.

    Inputs:
    - df: DataFrame with at least time_col and protocol_col
    - window_minutes: window size in minutes (default 10)
    - categories: list of protocol categories to use (order preserved). Rows with protocols not in categories are ignored.

    Output:
    - DataFrame indexed by window start time with columns for each category containing counts.
    """
    category_order = CATEGORIES if categories is None else validate_category_order(categories)
    validate_window_minutes(window_minutes)
    if time_col not in df or protocol_col not in df:
        raise ValueError(f"input must contain {time_col!r} and {protocol_col!r} columns")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)

    # resample into windows and count occurrences per protocol
    window = f"{window_minutes}min"
    grouped = (
        df.groupby(protocol_col)
        .resample(
            window,
            origin=WINDOW_ORIGIN,
            closed=WINDOW_CLOSED,
            label=WINDOW_LABEL,
        )
        .size()
        .unstack(level=0)
        .fillna(0)
    )

    # ensure all categories present
    for c in category_order:
        if c not in grouped.columns:
            grouped[c] = 0

    # keep only categories and sort columns by given order
    grouped = grouped[list(category_order)]

    # coerce to integer counts
    grouped = grouped.astype(int)
    return grouped
