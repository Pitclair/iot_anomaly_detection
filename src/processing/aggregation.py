"""
Utilities to aggregate packet traces into fixed windows and produce count vectors.
"""
from typing import List
import pandas as pd
import numpy as np


def aggregate_packet_traces(df: pd.DataFrame, time_col: str = 'timestamp', protocol_col: str = 'protocol', window_minutes: int = 10, categories: List[str] = None) -> pd.DataFrame:
    """
    Aggregate raw packet rows into time windows and count occurrences per protocol category.

    Inputs:
    - df: DataFrame with at least time_col and protocol_col
    - window_minutes: window size in minutes (default 10)
    - categories: list of protocol categories to use (order preserved). Rows with protocols not in categories are ignored.

    Output:
    - DataFrame indexed by window start time with columns for each category containing counts.
    """
    if categories is None:
        categories = ['TCP', 'UDP', 'SSDP', 'ARP']

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)

    # resample into windows and count occurrences per protocol
    window = f'{window_minutes}T'
    grouped = df.groupby(protocol_col).resample(window).size().unstack(level=0).fillna(0)

    # ensure all categories present
    for c in categories:
        if c not in grouped.columns:
            grouped[c] = 0

    # keep only categories and sort columns by given order
    grouped = grouped[categories]

    # coerce to integer counts
    grouped = grouped.astype(int)
    return grouped

