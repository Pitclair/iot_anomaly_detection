"""Tests for distribution-level Dirichlet-multinomial calculations."""

from math import lgamma

import numpy as np
import pytest

from lm_idnet.algorithms.dirichlet_multinomial import (
    anomaly_score,
    log_multinomial_coefficient,
    log_probability,
)
from lm_idnet.algorithms.log_likelihood import LmLogLikelihood, ScipyLogLikelihood
from lm_idnet.exceptions import DataValidationError, NumericalPrecisionError
from lm_idnet.processing.schemas import MAX_WINDOW_COUNT

pytestmark = [pytest.mark.unit, pytest.mark.numerical]


def test_log_multinomial_coefficient_matches_known_value() -> None:
    value = log_multinomial_coefficient(np.array([2, 1], dtype=np.int64))

    assert value == pytest.approx(np.log(3))


def test_log_multinomial_coefficient_accepts_silent_window() -> None:
    value = log_multinomial_coefficient(np.zeros(4, dtype=np.int64))

    assert value == pytest.approx(0.0)


def test_anomaly_score_supports_raw_and_normalized_scores() -> None:
    counts = np.array([2, 1], dtype=np.int64)
    silent = np.zeros(2, dtype=np.int64)
    alpha = np.ones(2)
    likelihood = ScipyLogLikelihood()
    raw = log_probability(counts, alpha, likelihood)

    assert anomaly_score(counts, alpha, likelihood, "raw") == raw
    assert anomaly_score(counts, alpha, likelihood, "normalized") == pytest.approx(
        raw / 3
    )
    assert anomaly_score(silent, alpha, likelihood, "normalized") == 0.0
    with pytest.raises(DataValidationError, match="unknown anomaly score type"):
        anomaly_score(counts, alpha, likelihood, "other")


@pytest.mark.parametrize("total", [10**9, 10**12, MAX_WINDOW_COUNT])
@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("score_type", ["raw", "normalized"])
@pytest.mark.parametrize("parameters", [(1, 1, 1, 1), (1, 2, 3, 4)])
def test_precision_recovery_matches_integer_dm_closed_form(
    total, balanced, score_type, parameters, caplog,
):
    counts = np.array([total, 0, 0, 0], dtype=np.int64)
    if balanced:
        counts = np.full(4, total // 4, dtype=np.int64)
        counts[0] += total % 4
    alpha = np.array(parameters, dtype=float)
    likelihood = LmLogLikelihood()
    with pytest.raises(NumericalPrecisionError):
        likelihood.calculate(counts[np.newaxis, :], alpha)

    score = anomaly_score(counts, alpha, likelihood, score_type)

    # Integer alpha reduces the Gamma ratios to short products, without cancellation.
    expected = lgamma(sum(parameters)) - sum(
        np.log(float(total + offset)) for offset in range(1, sum(parameters))
    ) + sum(
        sum(np.log(float(int(count) + offset)) for offset in range(1, parameter))
        - lgamma(parameter) for count, parameter in zip(counts, parameters)
    )
    if score_type == "normalized":
        expected /= total
    assert score == pytest.approx(expected, rel=1e-13, abs=0.)
    assert "high-precision DM scoring" in caplog.text


def test_precision_recovery_preserves_large_concentration_and_global_precision():
    from mpmath import mp

    previous_precision = mp.dps
    counts = np.array([3, 0, 0, 0], dtype=np.int64)
    score = anomaly_score(counts, np.full(4, 1e100), LmLogLikelihood(), "raw")

    assert score == pytest.approx(-3 * np.log(4.), abs=1e-12)
    assert mp.dps == previous_precision


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (np.array([], dtype=np.int64), "non-empty vector"),
        (np.ones((1, 2), dtype=np.int64), "non-empty vector"),
        (np.array([1, -1], dtype=np.int64), "non-negative"),
    ],
)
def test_log_multinomial_coefficient_rejects_invalid_counts(
    counts: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(DataValidationError, match=message):
        log_multinomial_coefficient(counts)
