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


def log_probability(counts, alpha, log_likelihood) -> float:
    coefficient = log_multinomial_coefficient(counts)
    kernel = log_likelihood.calculate(counts[np.newaxis, :], alpha)
    return coefficient + kernel


def anomaly_score(counts, alpha, log_likelihood, score_type: str) -> float:
    """Return a raw or per-packet normalized anomaly score."""
    score = log_probability(counts, alpha, log_likelihood)
    if score_type == "raw":
        return score
    if score_type == "normalized":
        return score / max(int(counts.sum()), 1)
    raise DataValidationError(f"unknown anomaly score type: {score_type}")
