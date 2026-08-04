"""Python implementations for Dirichlet parameter estimation."""

import numpy as np
from scipy.special import digamma, gammaln

from .log_likelihood import LogLikelihood


def fixed_point_dirichlet(
    counts: np.ndarray,
    log_likelihood: LogLikelihood,
    alpha_init: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, dict]:
    """
    Minka's fixed-point iteration to estimate Dirichlet alpha from count data.

    Args:
      counts: (n_samples, k) integer counts per category per sample
      log_likelihood: configured likelihood implementation used for convergence
      alpha_init: configured initial alpha vector (k,)
      tolerance: configured tolerance for log-likelihood improvement
      max_iterations: configured maximum number of iterations

    Returns:
      alpha: estimated alpha vector
      info: dict with diagnostics (iterations, converged, ll_history)
    """
    counts = np.asarray(counts, dtype=np.float64)
    alpha = np.asarray(alpha_init, dtype=np.float64)
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations <= 0:
        raise ValueError("maximum iterations must be positive")
    ll_prev = log_likelihood.calculate(counts, alpha)
    ll_history = [ll_prev]

    for it in range(1, max_iterations + 1):
        # compute sufficient statistics
        dig1 = digamma(counts + alpha)
        dig2 = digamma(alpha)
        numer = np.sum(dig1 - dig2, axis=0)  # shape (k,)

        denom = np.sum(digamma(counts.sum(axis=1) + alpha.sum()) - digamma(alpha.sum()))
        if not np.isfinite(denom) or denom <= 0:
            raise ValueError("fixed-point denominator must be finite and positive")
        alpha = alpha * (numer / denom)
        if not np.isfinite(alpha).all() or (alpha <= 0).any():
            raise ValueError("fixed-point update produced invalid alpha values")

        ll = log_likelihood.calculate(counts, alpha)
        ll_history.append(ll)
        if abs(ll - ll_prev) < tolerance:
            return alpha, {'iterations': it, 'converged': True, 'll_history': ll_history}
        ll_prev = ll

    return alpha, {'iterations': max_iterations, 'converged': False, 'll_history': ll_history}
