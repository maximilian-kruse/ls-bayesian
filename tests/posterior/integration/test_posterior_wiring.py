import numpy as np
import pytest

from ls_bayesian.posterior import likelihood, posterior
from tests.posterior import helpers

pytestmark = pytest.mark.integration

# Pure float64 linear-algebra identities, no discretization or statistical error: machine
# precision is the only source of slack.
LINEAR_ALGEBRA_RELATIVE_TOLERANCE = 1e-10


# ==================================================================================================
def test_posterior_cost_and_gradient_with_built_likelihood(
    posterior_setup: helpers.PosteriorSetup,
) -> None:
    """Exercises the real wiring `VertexObservationSettings -> from_vertex_observations ->
    LogPosterior`, checked against a reference computed independently from the raw settings."""
    likelihood_setup = posterior_setup.likelihood_setup
    built_likelihood = likelihood.GaussianLogLikelihood.from_vertex_observations(
        likelihood.VertexObservationSettings(
            likelihood_setup.data_vector,
            helpers.NUM_VERTICES,
            helpers.OBSERVED_VERTEX_INDICES,
            likelihood_setup.precision_values,
        )
    )
    log_posterior = posterior.LogPosterior(
        built_likelihood, posterior_setup.parameter_to_solution_map, posterior_setup.prior
    )
    parameter_vector = np.random.default_rng(4).random(helpers.PARAMETER_DIM)

    cost = log_posterior.evaluate_cost(parameter_vector)
    gradient = log_posterior.evaluate_gradient(parameter_vector)

    forward_matrix = posterior_setup.parameter_to_solution_map.matrix
    solution_vector = forward_matrix @ parameter_vector
    misfit = likelihood_setup.observation_matrix @ solution_vector - likelihood_setup.data_vector
    expected_likelihood_cost = 0.5 * misfit @ likelihood_setup.precision_matrix @ misfit
    expected_prior_cost = posterior_setup.prior.evaluate_cost(parameter_vector)
    expected_cost = expected_likelihood_cost + expected_prior_cost

    expected_likelihood_gradient = forward_matrix.T @ (
        likelihood_setup.observation_matrix.T @ (likelihood_setup.precision_matrix @ misfit)
    )
    expected_gradient = expected_likelihood_gradient + posterior_setup.prior.evaluate_gradient(
        parameter_vector
    )

    np.testing.assert_allclose(cost, expected_cost, rtol=LINEAR_ALGEBRA_RELATIVE_TOLERANCE)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=LINEAR_ALGEBRA_RELATIVE_TOLERANCE)
