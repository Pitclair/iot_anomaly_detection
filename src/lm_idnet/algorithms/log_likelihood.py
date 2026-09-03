"""Parameter-dependent Dirichlet-multinomial likelihood kernels."""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from scipy.special import gammaln

from lm_idnet.algorithms.lm_native import NativeLmKernel
from lm_idnet.exceptions import DataValidationError

LogLikelihoodBackend = Literal["scipy", "lm"]


class LogLikelihood(ABC):
    """Common interface for interchangeable likelihood-kernel implementations.

    The kernel omits the count-only multinomial coefficient because parameter
    estimation compares alpha values for the same observed counts.
    """

    @abstractmethod
    def calculate(self, counts: np.ndarray, alpha: np.ndarray) -> float:
        """Return the alpha-dependent log-likelihood kernel for the matrix."""


class ScipyLogLikelihood(LogLikelihood):
    """DM likelihood kernel calculated with SciPy's gammaln."""

    def calculate(self, counts: np.ndarray, alpha: np.ndarray) -> float:
        counts = np.asarray(counts)
        alpha = np.asarray(alpha, dtype=np.float64)
        self._validate_inputs(counts, alpha)

        row_totals = counts.sum(axis=1)
        concentration = alpha.sum()
        concentration_terms = gammaln(concentration) - gammaln(
            concentration + row_totals
        )
        category_terms = (
            gammaln(counts + alpha) - gammaln(alpha)
        ).sum(axis=1)
        kernel_value = float(np.sum(concentration_terms + category_terms))
        if not np.isfinite(kernel_value):
            raise DataValidationError(
                "Dirichlet-multinomial likelihood kernel is not finite"
            )
        return kernel_value

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
    """DM likelihood kernel calculated with the bundled LM implementation."""

    def __init__(self, precision_digits: int = 6) -> None:
        self.kernel = NativeLmKernel(precision_digits)

    def calculate(self, counts: np.ndarray, alpha: np.ndarray) -> float:
        counts = np.asarray(counts)
        alpha = np.asarray(alpha, dtype=np.float64)
        ScipyLogLikelihood._validate_inputs(counts, alpha)
        return self.kernel.calculate(counts, alpha)


_IMPLEMENTATIONS: dict[str, type[LogLikelihood]] = {
    "scipy": ScipyLogLikelihood,
    "lm": LmLogLikelihood,
}


def initialize_log_likelihood(
    backend: LogLikelihoodBackend,
    precision_digits: int = 6,
) -> LogLikelihood:
    """Instantiate the configured likelihood-kernel implementation."""
    implementation = _IMPLEMENTATIONS.get(backend)
    if implementation is None:
        raise DataValidationError(f"unknown log-likelihood backend: {backend}")
    return (
        implementation(precision_digits)
        if implementation is LmLogLikelihood
        else implementation()
    )
