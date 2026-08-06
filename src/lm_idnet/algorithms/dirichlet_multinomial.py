"""Distribution-level Dirichlet-multinomial calculations."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

from lm_idnet.exceptions import DataValidationError


def log_multinomial_coefficient(counts: NDArray[np.int64]) -> float:
    """Return the log multinomial coefficient for one count vector."""
    if counts.ndim != 1 or counts.size == 0:
        raise DataValidationError("counts must be a non-empty vector")
    if (counts < 0).any():
        raise DataValidationError("counts must be non-negative")

    total_count = counts.sum()
    coefficient = gammaln(total_count + 1) - gammaln(counts + 1).sum()
    if not np.isfinite(coefficient):
        raise DataValidationError("log multinomial coefficient is not finite")
    return float(coefficient)
