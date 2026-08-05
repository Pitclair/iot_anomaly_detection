"""Python implementations for Dirichlet parameter estimation."""

from dataclasses import dataclass

import numpy as np
from scipy.special import digamma

from lm_idnet.exceptions import DataValidationError

from .log_likelihood import LogLikelihood


@dataclass(frozen=True)
class DirichletFit:
    """Parameters and convergence details from Dirichlet fitting."""

    initial_alpha: np.ndarray
    alpha: np.ndarray
    concentration: float
    psi: float
    iterations: int
    converged: bool
    initial_log_likelihood: float
    final_log_likelihood: float


class DirichletMultinomialEstimator:
    """Fit Dirichlet-multinomial parameters using configured settings."""

    def __init__(
        self,
        log_likelihood: LogLikelihood,
        initial_concentration: float,
        tolerance: float,
        max_iterations: int,
    ) -> None:
        self.log_likelihood = log_likelihood
        self.initial_concentration = initial_concentration
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def fit(self, counts: np.ndarray) -> DirichletFit:
        """Initialize alpha once and fit it to the supplied count matrix."""
        initial_alpha = self._create_initial_alpha(
            counts,
            self.initial_concentration,
        )
        return self._fixed_point_dirichlet(
            counts,
            self.log_likelihood,
            initial_alpha,
            self.tolerance,
            self.max_iterations,
        )

    def _create_initial_alpha(
            self,
            counts: np.ndarray
    ) -> np.ndarray:
        """Initialize alpha from observed category proportions."""
        counts = np.asarray(counts)
        if counts.ndim != 2 or counts.shape[0] == 0 or counts.shape[1] < 2:
            raise DataValidationError(
                "initial alpha requires a non-empty matrix with multiple categories"
            )
        if not np.isfinite(self.initial_concentration) or self.initial_concentration <= 0:
            raise DataValidationError(
                "initial alpha concentration must be finite and positive"
            )

        category_totals = counts.sum(axis=0, dtype=np.float64)
        if (category_totals <= 0).any():
            raise DataValidationError(
                "every category needs observed counts to initialize alpha"
            )
        category_proportions = category_totals / category_totals.sum()
        return category_proportions * self.initial_concentration

    def _fixed_point_dirichlet(
            self,
            counts: np.ndarray,
            alpha_init: np.ndarray,
    ) -> DirichletFit:
        """Run Minka's fixed-point iteration from an existing alpha vector."""
        counts = np.asarray(counts, dtype=np.float64)
        initial_alpha = np.asarray(alpha_init, dtype=np.float64).copy()
        alpha = initial_alpha.copy()
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if self.max_iterations <= 0:
            raise ValueError("maximum iterations must be positive")
        initial_log_likelihood = self.log_likelihood.calculate(counts, alpha)
        previous_log_likelihood = initial_log_likelihood
        final_log_likelihood = initial_log_likelihood
        converged = False
        iterations = 0

        for iteration in range(1, self.max_iterations + 1):
            category_update = np.sum(
                digamma(counts + alpha) - digamma(alpha),
                axis=0,
            )
            concentration = alpha.sum()
            concentration_update = np.sum(
                digamma(counts.sum(axis=1) + concentration)
                - digamma(concentration)
            )
            if not np.isfinite(concentration_update) or concentration_update <= 0:
                raise ValueError("fixed-point denominator must be finite and positive")

            alpha = alpha * (category_update / concentration_update)
            if not np.isfinite(alpha).all() or (alpha <= 0).any():
                raise ValueError("fixed-point update produced invalid alpha values")

            final_log_likelihood = self.log_likelihood.calculate(counts, alpha)
            iterations = iteration
            if abs(final_log_likelihood - previous_log_likelihood) < self.tolerance:
                converged = True
                break
            previous_log_likelihood = final_log_likelihood

        concentration = float(alpha.sum())
        return DirichletFit(
            initial_alpha=initial_alpha,
            alpha=alpha,
            concentration=concentration,
            psi=1.0 / concentration,
            iterations=iterations,
            converged=converged,
            initial_log_likelihood=initial_log_likelihood,
            final_log_likelihood=final_log_likelihood,
        )

