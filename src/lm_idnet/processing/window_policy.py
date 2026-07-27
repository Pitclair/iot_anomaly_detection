"""Shared deterministic window-boundary policy."""

SUPPORTED_WINDOW_MINUTES = (1, 5, 10, 30)
WINDOW_ORIGIN = "epoch"
WINDOW_CLOSED = "left"
WINDOW_LABEL = "left"


def validate_window_minutes(window_minutes: int) -> int:
    """Return a supported window length or raise a clear configuration error."""
    if window_minutes not in SUPPORTED_WINDOW_MINUTES:
        raise ValueError(
            f"window_minutes must be one of {SUPPORTED_WINDOW_MINUTES}"
        )
    return window_minutes
