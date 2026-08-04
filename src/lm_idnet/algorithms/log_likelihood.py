"""Dirichlet-multinomial log-likelihood implementations."""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from scipy.special import gammaln

from lm_idnet.exceptions import CommandUnavailableError, DataValidationError

LogLikelihoodBackend = Literal["scipy", "lm"]


class LogLikelihood(ABC):
    """Common interface for interchangeable log-likelihood implementations."""

    @abstractmethod
    def calculate(self, counts: np.ndarray, alpha: np.ndarray) -> float:
        """Return the log-likelihood of the count matrix under alpha."""


class ScipyLogLikelihood(LogLikelihood):
    """Standard DM log-likelihood calculated with SciPy's gammaln."""

    def calculate(self, counts: np.ndarray, alpha: np.ndarray) -> float:
        counts = np.asarray(counts)
        alpha = np.asarray(alpha, dtype=np.float64)
        self._validate_inputs(counts, alpha)

        row_totals = counts.sum(axis=1)
        concentration = alpha.sum()
        multinomial_terms = (
            gammaln(row_totals + 1) - gammaln(counts + 1).sum(axis=1)
        )
        concentration_terms = gammaln(concentration) - gammaln(
            concentration + row_totals
        )
        category_terms = (
            gammaln(counts + alpha) - gammaln(alpha)
        ).sum(axis=1)
        value = float(
            np.sum(multinomial_terms + concentration_terms + category_terms)
        )
        if not np.isfinite(value):
            raise DataValidationError(
                "Dirichlet-multinomial log-likelihood is not finite"
            )
        return value

    @staticmethod
    def _validate_inputs(counts: np.ndarray, alpha: np.ndarray) -> None:
        if counts.ndim != 2 or counts.shape[0] == 0:
            raise DataValidationError("count matrix must have at least one row")
        if alpha.ndim != 1 or alpha.shape[0] != counts.shape[1]:
            raise DataValidationError(
                "alpha length must match the count matrix columns"
            )
        if not np.isfinite(counts).all() or (counts < 0).any():
            raise DataValidationError("counts must be finite and non-negative")
        if not np.equal(counts, np.floor(counts)).all():
            raise DataValidationError("counts must contain whole numbers")
        if not np.isfinite(alpha).all() or (alpha <= 0).any():
            raise DataValidationError("alpha values must be finite and positive")


class LmLogLikelihood(LogLikelihood):
    """Placeholder for the future Languasco-Migliardi implementation."""

    def calculate(self, counts: np.ndarray, alpha: np.ndarray) -> float:
        raise CommandUnavailableError(
            "the LM log-likelihood implementation is not available yet"
        )


_IMPLEMENTATIONS: dict[str, type[LogLikelihood]] = {
    "scipy": ScipyLogLikelihood,
    "lm": LmLogLikelihood,
}


def initialize_log_likelihood(backend: LogLikelihoodBackend) -> LogLikelihood:
    """Instantiate the configured log-likelihood implementation."""
    implementation = _IMPLEMENTATIONS.get(backend)
    if implementation is None:
        raise DataValidationError(f"unknown log-likelihood backend: {backend}")
    return implementation()
