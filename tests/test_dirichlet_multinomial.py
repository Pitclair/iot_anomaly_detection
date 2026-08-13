"""Tests for distribution-level Dirichlet-multinomial calculations."""

import numpy as np
import pytest

from lm_idnet.algorithms.dirichlet_multinomial import (
    anomaly_score,
    log_multinomial_coefficient,
    log_probability,
)
from lm_idnet.algorithms.log_likelihood import ScipyLogLikelihood
from lm_idnet.exceptions import DataValidationError

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
