import numpy as np
import pytest

from ls_bayesian.optimization.algorithms.custom_lbfgs import (
    MetricLBFGSOptimizer,
    MetricLBFGSSettings,
)
from ls_bayesian.optimization.components.cautious_update import (
    CautiousUpdateSettings,
    CautiousUpdateStrategy,
)
from ls_bayesian.optimization.components.line_search import (
    ArmijoBacktrackingLineSearch,
    ArmijoBacktrackingLineSearchSettings,
)
from ls_bayesian.posterior import posterior
from tests.optimization import helpers as optimization_helpers
from tests.posterior import helpers as posterior_helpers

pytestmark = pytest.mark.integration

CONVERGENCE_ABSOLUTE_TOLERANCE = 1e-3


# ==================================================================================================
def test_metric_lbfgs_converges_to_closed_form_map_under_weighted_geometry() -> None:
    """Proof that `optimization` never needs to import `spde_prior`/`posterior` to support a
    Cameron-Martin-like use case: only the `OptimizationModel` contract is needed.

    For the linear-Gaussian posterior, the negative log-posterior is exactly quadratic,
    $J(m) = \\frac{1}{2} m^T H m - b^T m + \\text{const}$, with Hessian $H$ (`system_matrix` below)
    and closed-form MAP point $m^* = H^{-1} b$. Weighting the inner product by $H$ itself emulates
    a Hessian/prior-preconditioned representer change (the role a Cameron-Martin inner product
    plays for a real Bayesian prior): `helpers.LogPosteriorModel` returns the Riesz representer of
    the gradient under $(u, v)_H = u^T H v$, i.e. $g_H(m) = H^{-1} \\nabla J(m)$, wrapping
    `LogPosterior`'s plain (Euclidean) gradient.
    """
    likelihood_setup = posterior_helpers.create_likelihood_setup(seed=20)
    posterior_setup = posterior_helpers.create_posterior_setup(likelihood_setup, seed=21)
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

    optimizer = MetricLBFGSOptimizer(
        MetricLBFGSSettings(maximum_num_iterations=200, gradient_norm_tolerance=1e-8),
        ArmijoBacktrackingLineSearch(ArmijoBacktrackingLineSearchSettings()),
        CautiousUpdateStrategy(CautiousUpdateSettings()),
    )
    model = optimization_helpers.LogPosteriorModel(
        log_posterior, inner_product_matrix=system_matrix
    )

    result = optimizer.run(np.zeros_like(prior_mean_vector), model)

    assert result.success
    np.testing.assert_allclose(
        result.result, expected_map_point, atol=CONVERGENCE_ABSOLUTE_TOLERANCE
    )
