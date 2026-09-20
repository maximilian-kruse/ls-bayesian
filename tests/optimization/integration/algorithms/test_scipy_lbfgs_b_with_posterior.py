import numpy as np
import pytest

from ls_bayesian.optimization.algorithms.scipy_lbfgs_b import (
    ScipyLBFGSBOptimizer,
    ScipyLBFGSBSettings,
)
from ls_bayesian.posterior import posterior
from tests.optimization import helpers as optimization_helpers
from tests.posterior import helpers as posterior_helpers

pytestmark = pytest.mark.integration

CONVERGENCE_ABSOLUTE_TOLERANCE = 1e-4


# ==================================================================================================
def test_lbfgs_converges_to_closed_form_map_of_linear_gaussian_posterior() -> None:
    """Cross-subpackage wiring test: runs `ScipyLBFGSBOptimizer` directly against a real
    `LogPosterior`'s bound `evaluate_cost`/`evaluate_gradient` methods, with no adapter code,
    proving the typed-`Callable` objective contract works end-to-end. For a linear forward map and
    Gaussian likelihood and prior, the posterior is exactly quadratic, so the MAP point has a
    closed form via a linear solve, independent of the optimizer under test."""
    likelihood_setup = posterior_helpers.create_likelihood_setup(seed=10)
    posterior_setup = posterior_helpers.create_posterior_setup(likelihood_setup, seed=11)
    log_posterior = posterior.LogPosterior(
        likelihood_setup.likelihood,
        posterior_setup.parameter_to_solution_map,
        posterior_setup.prior,
    )

    forward_matrix = posterior_setup.parameter_to_solution_map.matrix
    combined_observation_matrix = likelihood_setup.observation_matrix @ forward_matrix
    precision_matrix = likelihood_setup.precision_matrix
    prior_precision_matrix = posterior_setup.prior.precision_matrix
    prior_mean_vector = posterior_setup.prior.mean_vector
    data_vector = likelihood_setup.data_vector

    system_matrix = (
        combined_observation_matrix.T @ precision_matrix @ combined_observation_matrix
        + prior_precision_matrix
    )
    right_hand_side = (
        combined_observation_matrix.T @ precision_matrix @ data_vector
        + prior_precision_matrix @ prior_mean_vector
    )
    expected_map_point = np.linalg.solve(system_matrix, right_hand_side)

    optimizer = ScipyLBFGSBOptimizer(ScipyLBFGSBSettings())
    model = optimization_helpers.LogPosteriorModel(log_posterior)
    result = optimizer.run(np.zeros_like(prior_mean_vector), model)

    assert result.success
    np.testing.assert_allclose(
        result.result, expected_map_point, atol=CONVERGENCE_ABSOLUTE_TOLERANCE
    )
