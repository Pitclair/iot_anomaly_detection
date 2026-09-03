"""Tests for estimator configuration wiring."""

import pytest

from lm_idnet.algorithms.dirichlet import DirichletMultinomialEstimator
from lm_idnet.algorithms.log_likelihood import LmLogLikelihood, ScipyLogLikelihood
from lm_idnet.algorithms.estimator_factory import create_estimator

pytestmark = pytest.mark.unit


def test_create_estimator_wires_configuration(config_factory) -> None:
    app_config = config_factory(
        estimator={
            "initial_alpha_concentration": 8.0,
            "tolerance_delta": 1e-5,
            "max_iterations": 250,
            "log_likelihood_backend": "scipy",
        }
    )

    estimator = create_estimator(app_config.estimator)

    assert isinstance(estimator, DirichletMultinomialEstimator)
    assert isinstance(estimator.log_likelihood, ScipyLogLikelihood)
    assert estimator.initial_concentration == 8.0
    assert estimator.tolerance == 1e-5
    assert estimator.max_iterations == 250


def test_create_estimator_forwards_lm_precision(config_factory) -> None:
    app_config = config_factory(
        estimator={"log_likelihood_backend": "lm"},
    )

    estimator = create_estimator(app_config.estimator, precision_digits=8)

    assert isinstance(estimator.log_likelihood, LmLogLikelihood)
    assert estimator.log_likelihood.kernel.precision_digits == 8
