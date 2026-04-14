"""
Transform raw (timestamp, protocol) pairs into 10-minute count windows.
"""
from typing import Iterable, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from .schemas import WindowCount


def build_time_series(records: Iterable[Tuple[float, str]]) -> pd.DataFrame:
    """Convert an iterable of (timestamp, protocol) into a DataFrame with datetime index and protocol column.

    Returns DataFrame with columns ['protocol'] indexed by datetime.
    """
    rows = []
    for ts, proto in records:
        # convert ts to pandas Timestamp
        try:
            t = pd.to_datetime(ts, unit='s')
        except Exception:
            continue
        rows.append({'timestamp': t, 'protocol': proto})

    if not rows:
        return pd.DataFrame(columns=['protocol']).set_index(pd.DatetimeIndex([]))

    df = pd.DataFrame(rows)
    df.set_index('timestamp', inplace=True)
    return df


def to_10min_windows(df: pd.DataFrame, categories: List[str]) -> List[WindowCount]:
    """Resample df into 10-minute windows and return list of validated WindowCount objects.

    Handles silent windows by filling zeros for missing intervals.
    """
    if not categories:
        raise ValueError("The 'categories' argument must not be empty.")

    if df.empty:
        # no packets at all -> return empty list
        return []

    # Ensure timezone-naive timestamps and sort
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
    for c in categories:
        if c not in grouped.columns:
            grouped[c] = 0

    grouped = grouped[categories].astype(int)

    # Convert rows to Pydantic WindowCount objects dynamically
    windows = []
    for _, row in grouped.iterrows():
        wc_data = {protocol.lower(): int(row[protocol]) for protocol in grouped.columns}
        wc = WindowCount(**wc_data)
        windows.append(wc)

    return windows


def to_numpy_matrix(windows: List[WindowCount]) -> np.ndarray:
    """Convert list of WindowCount into numpy array shape (R, K).
    Order: TCP, UDP, SSDP, ARP
    """
    if not windows:
        return np.empty((0, 4), dtype=int)

    mat = np.array([[w.tcp, w.udp, w.ssdp, w.arp] for w in windows], dtype=int)
    return mat
