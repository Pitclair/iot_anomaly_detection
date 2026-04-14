"""
Simple statistics utilities to check mean/variance per protocol column.
"""
from typing import Sequence, Tuple
import numpy as np


def mean_variance_per_column(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (means, variances) for each column in matrix.

    matrix shape: (R, K)
    """
    if matrix.size == 0:
        return np.array([]), np.array([])

    means = np.mean(matrix, axis=0)
    variances = np.var(matrix, axis=0, ddof=1)  # sample variance
    return means, variances


def is_overdispersed(matrix: np.ndarray) -> np.ndarray:
    """Return boolean array of length K indicating variance>mean per column."""
    means, variances = mean_variance_per_column(matrix)
    if means.size == 0:
        return np.array([])
    return variances > means

