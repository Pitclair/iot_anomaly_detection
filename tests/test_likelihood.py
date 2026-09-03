"""Tests for selectable log-likelihood implementations."""

import numpy as np
import pytest

from lm_idnet.algorithms.log_likelihood import (
    LmLogLikelihood,
    ScipyLogLikelihood,
    initialize_log_likelihood,
)
from lm_idnet.exceptions import DataValidationError

pytestmark = [pytest.mark.unit, pytest.mark.numerical]


def test_scipy_backend_returns_parameter_dependent_kernel() -> None:
    counts = np.array([[2, 0], [1, 1]], dtype=np.int64)
    alpha = np.ones(2)
    log_likelihood = initialize_log_likelihood("scipy")

    value = log_likelihood.calculate(counts, alpha)

    # The full probabilities are both 1/3. Removing the multinomial
    # coefficients of 1 and 2 leaves kernel values of 1/3 and 1/6.
    assert isinstance(log_likelihood, ScipyLogLikelihood)
    assert value == pytest.approx(np.log(1 / 3) + np.log(1 / 6))


def test_likelihood_rejects_alpha_with_wrong_length() -> None:
    with pytest.raises(DataValidationError, match="alpha length"):
        ScipyLogLikelihood().calculate(np.ones((2, 4)), np.ones(3))


def test_likelihood_rejects_negative_counts() -> None:
    with pytest.raises(DataValidationError, match="non-negative"):
        ScipyLogLikelihood().calculate(np.array([[1, -1]]), np.ones(2))


def test_lm_backend_matches_scipy() -> None:
    counts = np.array([[2, 0], [1, 1]], dtype=np.int64)
    alpha = np.ones(2)
    log_likelihood = initialize_log_likelihood("lm", precision_digits=6)

    assert isinstance(log_likelihood, LmLogLikelihood)
    assert log_likelihood.kernel.precision_digits == 6
    assert log_likelihood.calculate(counts, alpha) == pytest.approx(
        ScipyLogLikelihood().calculate(counts, alpha),
        rel=1e-6,
        abs=1e-8,
    )


def test_lm_backend_reuses_input_validation() -> None:
    with pytest.raises(DataValidationError, match="whole numbers"):
        LmLogLikelihood().calculate(np.array([[1.5, 0]]), np.ones(2))


def test_unknown_likelihood_backend_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="unknown log-likelihood"):
        initialize_log_likelihood("unknown")  # type: ignore[arg-type]
