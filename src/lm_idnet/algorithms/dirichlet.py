"""Python-only implementations for Dirichlet parameter estimation.
Includes a Fixed-Point Iteration (Minka) implementation and placeholders for LM, YS, and lgam-std helpers.
"""
from typing import Optional, Tuple
import numpy as np
import math
from scipy.special import digamma, gammaln


def dirichlet_multinomial_log_likelihood(counts: np.ndarray, alpha: np.ndarray) -> float:
    """Compute log-likelihood of Dirichlet-Multinomial for count matrix (n_samples, k).
    Uses gammaln for stability.
    """
    if counts.ndim == 1:
        counts = counts.reshape(1, -1)
    N = counts.sum(axis=1)
    K = alpha.size
    A = alpha.sum()
    ll = np.sum(gammaln(N + 1)) - np.sum(gammaln(counts + 1))
    ll += len(counts) * (gammaln(A) - np.sum(gammaln(alpha)))
    ll += np.sum(gammaln(counts + alpha) - gammaln(A + N)[:, None])
    return float(ll)


def fixed_point_dirichlet(counts: np.ndarray, alpha_init: Optional[np.ndarray] = None, tol: float = 1e-9, max_iter: int = 1000) -> Tuple[np.ndarray, dict]:
    """
    Minka's fixed-point iteration to estimate Dirichlet alpha from count data.

    Args:
      counts: (n_samples, k) integer counts per category per sample
      alpha_init: optional initial alpha vector (k,)
      tol: tolerance for log-likelihood improvement
      max_iter: max iterations

    Returns:
      alpha: estimated alpha vector
      info: dict with diagnostics (iterations, converged, ll_history)
    """
    counts = np.asarray(counts, dtype=np.float64)
    n, k = counts.shape
    if alpha_init is None:
        # method of moments initialization (simple)
        mean = counts.mean(axis=0)
        mean /= mean.sum()
        alpha_init = np.maximum(mean * 10.0, 1e-2)

    alpha = alpha_init.astype(np.float64)
    ll_prev = dirichlet_multinomial_log_likelihood(counts, alpha)
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

        ll = dirichlet_multinomial_log_likelihood(counts, alpha)
        ll_history.append(ll)
        if abs(ll - ll_prev) < tol:
            return alpha, {'iterations': it, 'converged': True, 'll_history': ll_history}
        ll_prev = ll

    return alpha, {'iterations': max_iter, 'converged': False, 'll_history': ll_history}


# Placeholder implementations for LM, YS, lgam-std approaches

def lm_estimate(*args, **kwargs):
    """Languasco-Migliardi estimator placeholder.
    For now, delegate to fixed_point_dirichlet.
    """
    return fixed_point_dirichlet(*args, **kwargs)


def ys_estimate(*args, **kwargs):
    """Yu-Shaw estimator placeholder.
    """
    return fixed_point_dirichlet(*args, **kwargs)


def lgam_std(x: float) -> float:
    """Use math.lgamma or scipy.special.gammaln for gamma-log computations.
    """
    return float(gammaln(x))
