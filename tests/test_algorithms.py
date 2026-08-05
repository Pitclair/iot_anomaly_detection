import numpy as np
import pytest

from lm_idnet.algorithms.dirichlet import (
    DirichletFit,
    DirichletMultinomialEstimator,
)
from lm_idnet.algorithms.log_likelihood import LogLikelihood, ScipyLogLikelihood
from lm_idnet.exceptions import DataValidationError

pytestmark = [pytest.mark.unit, pytest.mark.numerical]


class RecordingLogLikelihood(LogLikelihood):
    def __init__(self) -> None:
        self.calls = 0

    def calculate(self, counts: np.ndarray, alpha: np.ndarray) -> float:
        self.calls += 1
        return float(alpha.sum())


def test_initial_alpha_requires_multiple_categories() -> None:
    counts = np.ones((2, 1), dtype=np.int64)
    estimator = DirichletMultinomialEstimator(
        log_likelihood=ScipyLogLikelihood(),
        initial_concentration=10.0,
        tolerance=1e-6,
        max_iterations=200,
    )

    with pytest.raises(DataValidationError, match="multiple categories"):
        estimator.fit(counts)


def test_initial_alpha_uses_data_proportions_and_configured_concentration() -> None:
    counts = np.array([[3, 1], [1, 1]], dtype=np.int64)
    estimator = DirichletMultinomialEstimator(
        log_likelihood=ScipyLogLikelihood(),
        initial_concentration=9.0,
        tolerance=1e-6,
        max_iterations=200,
    )

    fit = estimator.fit(counts)

    assert fit.initial_alpha.tolist() == pytest.approx([6.0, 3.0])
    assert fit.initial_alpha.sum() == pytest.approx(9.0)


def test_initial_alpha_rejects_unobserved_category() -> None:
    counts = np.array([[3, 0], [1, 0]], dtype=np.int64)
    estimator = DirichletMultinomialEstimator(
        log_likelihood=ScipyLogLikelihood(),
        initial_concentration=10.0,
        tolerance=1e-6,
        max_iterations=200,
    )

    with pytest.raises(DataValidationError, match="every category"):
        estimator.fit(counts)


def test_estimator_returns_complete_fit() -> None:
    counts = np.array([[10, 5, 0, 0], [9, 6, 1, 0], [11, 4, 0, 1]])
    estimator = DirichletMultinomialEstimator(
        log_likelihood=ScipyLogLikelihood(),
        initial_concentration=10.0,
        tolerance=1e-6,
        max_iterations=200,
    )

    fit = estimator.fit(counts)

    assert isinstance(fit, DirichletFit)
    assert fit.initial_alpha.sum() == pytest.approx(10.0)
    assert fit.alpha.shape == (4,)
    assert (fit.alpha > 0).all()
    assert fit.concentration == pytest.approx(fit.alpha.sum())
    assert fit.psi == pytest.approx(1.0 / fit.concentration)
    assert 1 <= fit.iterations <= 200
    assert isinstance(fit.converged, bool)
    assert np.isfinite(fit.initial_log_likelihood)
    assert np.isfinite(fit.final_log_likelihood)


def test_estimator_uses_injected_log_likelihood() -> None:
    counts = np.array([[10, 5], [9, 6]], dtype=np.int64)
    log_likelihood = RecordingLogLikelihood()
    estimator = DirichletMultinomialEstimator(
        log_likelihood=log_likelihood,
        initial_concentration=10.0,
        tolerance=1e-9,
        max_iterations=2,
    )

    estimator.fit(counts)

    # One call establishes the baseline and each iteration evaluates its update.
    assert log_likelihood.calls == 3
