"""Connect estimator configuration to the Dirichlet fitting algorithm."""

from lm_idnet.algorithms.dirichlet import DirichletMultinomialEstimator
from lm_idnet.algorithms.log_likelihood import initialize_log_likelihood
from lm_idnet.config import EstimatorConfig


def create_estimator(
    config: EstimatorConfig,
    precision_digits: int = 6,
) -> DirichletMultinomialEstimator:
    """Create a Dirichlet estimator from application configuration."""
    return DirichletMultinomialEstimator(
        log_likelihood=initialize_log_likelihood(
            config.log_likelihood_backend,
            precision_digits,
        ),
        initial_concentration=config.initial_alpha_concentration,
        tolerance=config.tolerance_delta,
        max_iterations=config.max_iterations,
    )
