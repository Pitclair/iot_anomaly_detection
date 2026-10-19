"""Distribution-level Dirichlet-multinomial calculations."""

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

from lm_idnet.exceptions import DataValidationError, NumericalPrecisionError

logger = logging.getLogger(__name__)


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
    try:
        score = log_probability(counts, alpha, log_likelihood)
    except NumericalPrecisionError:
        from mpmath import mp

        # Recompute the full probability so large factorial terms cancel accurately.
        context = mp.clone()
        largest = max(float(max(counts)), float(max(alpha)), 1.0)
        context.dps = max(50, int(np.log10(largest)) + 40)
        total = sum(int(count) for count in counts)
        parameters = [context.mpf(float(value)) for value in alpha]
        concentration = context.fsum(parameters)
        score = float(context.fsum([
            context.loggamma(total + 1),
            context.loggamma(concentration),
            -context.loggamma(concentration + total),
            *(context.loggamma(int(count) + parameter)
              - context.loggamma(parameter) - context.loggamma(int(count) + 1)
              for count, parameter in zip(counts, parameters)),
        ]))
        logger.warning("Native likelihood precision limit reached; used high-precision DM scoring")
    if score_type == "raw":
        return score
    if score_type == "normalized":
        return score / max(int(counts.sum()), 1)
    raise DataValidationError(f"unknown anomaly score type: {score_type}")
