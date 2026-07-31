"""Shared deterministic window-boundary policy."""

from numbers import Integral

WINDOW_ORIGIN = "epoch"
WINDOW_CLOSED = "left"
WINDOW_LABEL = "left"


def validate_window_minutes(window_minutes: object) -> int:
    """Return a positive whole-minute window length."""
    if isinstance(window_minutes, bool) or not isinstance(window_minutes, Integral):
        raise ValueError("window_minutes must be an integer")
    if window_minutes <= 0:
        raise ValueError("window_minutes must be positive")
    return int(window_minutes)
