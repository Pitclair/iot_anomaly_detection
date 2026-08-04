import numpy as np
import pytest

from lm_idnet.algorithms.dirichlet import fixed_point_dirichlet
from lm_idnet.algorithms.log_likelihood import LogLikelihood, ScipyLogLikelihood

pytestmark = [pytest.mark.unit, pytest.mark.numerical]

class RecordingLogLikelihood(LogLikelihood):
    def __init__(self) -> None:
        self.calls = 0

    def calculate(self, counts: np.ndarray, alpha: np.ndarray) -> float:
        self.calls += 1
        return float(alpha.sum())


def test_fixed_point_small():
    # small synthetic counts with clear proportions
    counts = np.array([[10, 5, 0, 0], [9,6,1,0], [11,4,0,1]])
    alpha, info = fixed_point_dirichlet(
        counts,
        ScipyLogLikelihood(),
        tol=1e-6,
        max_iter=200,
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
        max_iter=2,
    )

    # One call establishes the baseline and each iteration evaluates its update.
    assert log_likelihood.calls == 3
