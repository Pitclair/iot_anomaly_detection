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


def create_initial_alpha(
    counts: np.ndarray,
    concentration: float,
) -> np.ndarray:
    """Initialize alpha from observed category proportions."""
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[0] == 0 or counts.shape[1] < 2:
        raise DataValidationError(
            "initial alpha requires a non-empty matrix with multiple categories"
        )
    if not np.isfinite(concentration) or concentration <= 0:
        raise DataValidationError(
            "initial alpha concentration must be finite and positive"
        )

    category_totals = counts.sum(axis=0, dtype=np.float64)
    if (category_totals <= 0).any():
        raise DataValidationError(
            "every category needs observed counts to initialize alpha"
        )
    category_proportions = category_totals / category_totals.sum()
    return category_proportions * concentration


def fixed_point_dirichlet(
    counts: np.ndarray,
    log_likelihood: LogLikelihood,
    alpha_init: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> DirichletFit:
    """Estimate Dirichlet parameters using Minka's fixed-point iteration."""
    counts = np.asarray(counts, dtype=np.float64)
    initial_alpha = np.asarray(alpha_init, dtype=np.float64).copy()
    alpha = initial_alpha.copy()
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if max_iterations <= 0:
        raise ValueError("maximum iterations must be positive")
    initial_log_likelihood = log_likelihood.calculate(counts, alpha)
    previous_log_likelihood = initial_log_likelihood
    final_log_likelihood = initial_log_likelihood
    converged = False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
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

        final_log_likelihood = log_likelihood.calculate(counts, alpha)
        iterations = iteration
        if abs(final_log_likelihood - previous_log_likelihood) < tolerance:
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
