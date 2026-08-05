import numpy as np
import pytest

from lm_idnet.algorithms.dirichlet import create_initial_alpha, fixed_point_dirichlet
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

    with pytest.raises(DataValidationError, match="multiple categories"):
        create_initial_alpha(counts, 10.0)


def test_initial_alpha_uses_data_proportions_and_configured_concentration() -> None:
    counts = np.array([[3, 1], [1, 1]], dtype=np.int64)

    alpha = create_initial_alpha(counts, 9.0)

    assert alpha.tolist() == pytest.approx([6.0, 3.0])
    assert alpha.sum() == pytest.approx(9.0)


def test_initial_alpha_rejects_unobserved_category() -> None:
    counts = np.array([[3, 0], [1, 0]], dtype=np.int64)

    with pytest.raises(DataValidationError, match="every category"):
        create_initial_alpha(counts, 10.0)


def test_fixed_point_small():
    # small synthetic counts with clear proportions
    counts = np.array([[10, 5, 0, 0], [9,6,1,0], [11,4,0,1]])
    alpha, info = fixed_point_dirichlet(
        counts,
        ScipyLogLikelihood(),
        alpha_init=np.ones(4),
        tolerance=1e-6,
        max_iterations=200,
    )
    assert alpha.shape[0] == 4
    assert info['converged'] in (True, False)
    # alpha should be positive
    assert (alpha > 0).all()


def test_fixed_point_uses_injected_log_likelihood() -> None:
    counts = np.array([[10, 5], [9, 6]], dtype=np.int64)
    log_likelihood = RecordingLogLikelihood()

    fixed_point_dirichlet(
        counts,
        log_likelihood,
        alpha_init=np.ones(2),
        tolerance=1e-9,
        max_iterations=2,
    )

    # One call establishes the baseline and each iteration evaluates its update.
    assert log_likelihood.calls == 3
