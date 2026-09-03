"""Tests for selectable log-likelihood implementations."""

from pathlib import Path

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


def test_lm_backend_handles_zero_categories_and_silent_rows() -> None:
    observed = np.array([[10, 0, 2, 0], [0, 4, 0, 1]], dtype=np.int64)
    silent = np.zeros((1, 4), dtype=np.int64)
    alpha = np.array([0.2, 4.0, 0.01, 0.5])
    backend = LmLogLikelihood(precision_digits=6)

    observed_value = backend.calculate(observed, alpha)

    assert backend.calculate(np.vstack((observed, silent)), alpha) == observed_value
    assert observed_value == pytest.approx(
        ScipyLogLikelihood().calculate(observed, alpha),
        abs=1e-6,
    )


def test_lm_backend_sums_independent_rows() -> None:
    counts = np.array([[4, 1, 0], [0, 7, 2], [6, 0, 3]], dtype=np.int64)
    alpha = np.array([0.3, 2.5, 1.1])
    backend = LmLogLikelihood(precision_digits=6)

    row_sum = sum(backend.calculate(counts[row : row + 1], alpha) for row in range(3))

    assert backend.calculate(counts, alpha) == pytest.approx(row_sum, abs=1e-10)


def test_lm_backend_matches_scipy_across_seeded_stress_sample() -> None:
    rng = np.random.default_rng(20260814)
    lm_backend = LmLogLikelihood(precision_digits=6)
    scipy_backend = ScipyLogLikelihood()
    largest_difference = 0.0

    for _ in range(200):
        counts = rng.integers(
            0,
            100_001,
            size=(int(rng.integers(1, 20)), 4),
            dtype=np.int64,
        )
        alpha = 10.0 ** rng.uniform(-4.0, 2.0, size=4)
        difference = abs(
            lm_backend.calculate(counts, alpha)
            - scipy_backend.calculate(counts, alpha)
        )
        largest_difference = max(largest_difference, difference)

    assert largest_difference <= 1e-6, (
        f"largest LM/SciPy absolute difference was {largest_difference:.12g}"
    )


def test_lm_backend_can_be_reused_for_different_matrix_shapes() -> None:
    backend = LmLogLikelihood(precision_digits=6)
    scipy_backend = ScipyLogLikelihood()
    cases = (
        (np.array([[2, 1]], dtype=np.int64), np.array([0.5, 1.5])),
        (np.array([[3, 0, 4], [0, 2, 1]], dtype=np.int64), np.ones(3)),
        (np.array([[7, 1, 0, 2, 5]], dtype=np.int64), np.arange(1.0, 6.0)),
    )

    for counts, alpha in cases:
        assert backend.calculate(counts, alpha) == pytest.approx(
            scipy_backend.calculate(counts, alpha),
            abs=1e-6,
        )


def test_lm_backend_is_independent_of_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert LmLogLikelihood(6).calculate(np.array([[2, 1]]), np.ones(2)) == (
        pytest.approx(ScipyLogLikelihood().calculate(np.array([[2, 1]]), np.ones(2)))
    )


@pytest.mark.parametrize("backend_type", [ScipyLogLikelihood, LmLogLikelihood])
@pytest.mark.parametrize(
    ("counts", "alpha"),
    [
        (np.ones(4), np.ones(4)),
        (np.empty((0, 4)), np.ones(4)),
        (np.empty((1, 0)), np.empty(0)),
        (np.ones((1, 1)), np.ones(1)),
        (np.ones((2, 4)), np.ones(3)),
        (np.array([[1, -1]]), np.ones(2)),
        (np.array([[1.5, 0]]), np.ones(2)),
        (np.array([[1, np.nan]]), np.ones(2)),
        (np.ones((1, 2)), np.array([1.0, 0.0])),
        (np.ones((1, 2)), np.array([1.0, np.nan])),
        (np.ones((1, 2)), np.ones((1, 2))),
    ],
)
def test_backends_reject_invalid_inputs_consistently(
    backend_type: type[ScipyLogLikelihood] | type[LmLogLikelihood],
    counts: np.ndarray,
    alpha: np.ndarray,
) -> None:
    with pytest.raises(DataValidationError):
        backend_type().calculate(counts, alpha)


def test_unknown_likelihood_backend_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="unknown log-likelihood"):
        initialize_log_likelihood("unknown")  # type: ignore[arg-type]
