"""Tests for selectable log-likelihood implementations."""

import numpy as np
import pytest

from lm_idnet.algorithms.log_likelihood import (
    LmLogLikelihood,
    ScipyLogLikelihood,
    initialize_log_likelihood,
)
from lm_idnet.exceptions import CommandUnavailableError, DataValidationError

pytestmark = [pytest.mark.unit, pytest.mark.numerical]


def test_scipy_backend_matches_small_known_distribution() -> None:
    counts = np.array([[2, 0], [1, 1]], dtype=np.int64)
    alpha = np.ones(2)
    log_likelihood = initialize_log_likelihood("scipy")

    value = log_likelihood.calculate(counts, alpha)

    # With alpha=(1, 1) and N=2, all three count vectors have probability 1/3.
    assert isinstance(log_likelihood, ScipyLogLikelihood)
    assert value == pytest.approx(2 * np.log(1 / 3))


def test_likelihood_rejects_alpha_with_wrong_length() -> None:
    with pytest.raises(DataValidationError, match="alpha length"):
        ScipyLogLikelihood().calculate(np.ones((2, 4)), np.ones(3))


def test_likelihood_rejects_negative_counts() -> None:
    with pytest.raises(DataValidationError, match="non-negative"):
        ScipyLogLikelihood().calculate(np.array([[1, -1]]), np.ones(2))


def test_lm_backend_initializes_as_placeholder() -> None:
    log_likelihood = initialize_log_likelihood("lm")

    assert isinstance(log_likelihood, LmLogLikelihood)
    with pytest.raises(CommandUnavailableError, match="not available yet"):
        log_likelihood.calculate(np.ones((2, 2)), np.ones(2))
