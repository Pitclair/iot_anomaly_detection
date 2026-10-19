"""Distance metrics and statistical tests used for evaluation.
Includes TVD, KL divergence and MSE.
"""
import numpy as np
from scipy.special import rel_entr


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between distributions p and q (per-row average if matrix)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * np.sum(np.abs(p - q)) / (p.shape[0] if p.ndim == 2 else 1)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # use elementwise relative entropy then sum
    return float(np.sum(rel_entr(p, q)))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean((a - b) ** 2))
