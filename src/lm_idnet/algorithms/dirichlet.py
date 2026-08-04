"""Python implementations for Dirichlet parameter estimation."""

import numpy as np
from scipy.special import digamma, gammaln

from .log_likelihood import LogLikelihood


def fixed_point_dirichlet(
    counts: np.ndarray,
    log_likelihood: LogLikelihood,
    alpha_init: np.ndarray | None = None,
    tol: float = 1e-9,
    max_iter: int = 1000,
) -> tuple[np.ndarray, dict]:
    """
    Minka's fixed-point iteration to estimate Dirichlet alpha from count data.

    Args:
      counts: (n_samples, k) integer counts per category per sample
      log_likelihood: configured likelihood implementation used for convergence
      alpha_init: optional initial alpha vector (k,)
      tol: tolerance for log-likelihood improvement
      max_iter: max iterations

    Returns:
      alpha: estimated alpha vector
      info: dict with diagnostics (iterations, converged, ll_history)
    """
    counts = np.asarray(counts, dtype=np.float64)
    if alpha_init is None:
        # method of moments initialization (simple)
        mean = counts.mean(axis=0)
        mean /= mean.sum()
        alpha_init = np.maximum(mean * 10.0, 1e-2)

    alpha = alpha_init.astype(np.float64)
    ll_prev = log_likelihood.calculate(counts, alpha)
    ll_history = [ll_prev]

    for it in range(1, max_iter + 1):
        # compute sufficient statistics
        dig1 = digamma(counts + alpha)
        dig2 = digamma(alpha)
        numer = np.sum(dig1 - dig2, axis=0)  # shape (k,)

        denom = np.sum(digamma(counts.sum(axis=1) + alpha.sum()) - digamma(alpha.sum()))
        # fixed-point update (element-wise)
        # avoid division by zero
        denom = np.maximum(denom, 1e-16)
        alpha = alpha * (numer / denom)
        # ensure positivity
        alpha = np.maximum(alpha, 1e-12)

        ll = log_likelihood.calculate(counts, alpha)
        ll_history.append(ll)
        if abs(ll - ll_prev) < tol:
            return alpha, {'iterations': it, 'converged': True, 'll_history': ll_history}
        ll_prev = ll

    return alpha, {'iterations': max_iter, 'converged': False, 'll_history': ll_history}
