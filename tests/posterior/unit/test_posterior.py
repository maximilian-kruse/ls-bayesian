from pathlib import Path

import numpy as np
import pytest

from ls_bayesian.common.logging import BaseLogger, LoggerSettings
from ls_bayesian.posterior import posterior
from tests.posterior import helpers

pytestmark = pytest.mark.unit

# Central differences are only accurate up to floating-point cancellation error, which grows as
# the step size shrinks; this bound leaves headroom above that error for FINITE_DIFFERENCE_STEP.
FINITE_DIFFERENCE_RELATIVE_TOLERANCE = 1e-6


# ==================================================================================================
def _build_posterior(
    setup: helpers.PosteriorSetup, logger: BaseLogger | None = None
) -> posterior.LogPosterior:
    return posterior.LogPosterior(
        setup.likelihood_setup.likelihood,
        setup.parameter_to_solution_map,
        setup.prior,
        logger=logger,
    )


def _random_parameter(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(helpers.PARAMETER_DIM)


# ==================================================================================================
def test_posterior_cost(posterior_setup: helpers.PosteriorSetup) -> None:
    log_posterior = _build_posterior(posterior_setup)
    parameter_vector = _random_parameter(4)

    likelihood_cost, prior_cost = log_posterior.evaluate_cost(parameter_vector, split=True)
    total_cost = log_posterior.evaluate_cost(parameter_vector)

    solution_vector = posterior_setup.parameter_to_solution_map.matrix @ parameter_vector
    expected_likelihood_cost = posterior_setup.likelihood_setup.likelihood.evaluate_cost(
        solution_vector
    )
    expected_prior_cost = posterior_setup.prior.evaluate_cost(parameter_vector)
    np.testing.assert_allclose(likelihood_cost, expected_likelihood_cost)
    np.testing.assert_allclose(prior_cost, expected_prior_cost)
    np.testing.assert_allclose(total_cost, expected_likelihood_cost + expected_prior_cost)


# --------------------------------------------------------------------------------------------------
def test_posterior_gradient(posterior_setup: helpers.PosteriorSetup) -> None:
    log_posterior = _build_posterior(posterior_setup)
    parameter_vector = _random_parameter(4)

    gradient = log_posterior.evaluate_gradient(parameter_vector)

    expected_gradient = helpers.central_difference_gradient(
        log_posterior.evaluate_cost, parameter_vector, helpers.FINITE_DIFFERENCE_STEP
    )
    np.testing.assert_allclose(
        gradient, expected_gradient, rtol=FINITE_DIFFERENCE_RELATIVE_TOLERANCE
    )


# --------------------------------------------------------------------------------------------------
def test_posterior_split_gradient(posterior_setup: helpers.PosteriorSetup) -> None:
    log_posterior = _build_posterior(posterior_setup)
    parameter_vector = _random_parameter(4)

    likelihood_gradient, prior_gradient = log_posterior.evaluate_gradient(
        parameter_vector, split=True
    )

    forward_matrix = posterior_setup.parameter_to_solution_map.matrix
    expected_likelihood_gradient = forward_matrix.T @ (
        posterior_setup.likelihood_setup.likelihood.evaluate_gradient(
            forward_matrix @ parameter_vector
        )
    )
    np.testing.assert_allclose(likelihood_gradient, expected_likelihood_gradient)
    np.testing.assert_allclose(
        prior_gradient, posterior_setup.prior.evaluate_gradient(parameter_vector)
    )
    np.testing.assert_allclose(
        likelihood_gradient + prior_gradient, log_posterior.evaluate_gradient(parameter_vector)
    )


# --------------------------------------------------------------------------------------------------
def test_posterior_reuses_forward_solution(posterior_setup: helpers.PosteriorSetup) -> None:
    log_posterior = _build_posterior(posterior_setup)
    parameter_to_solution_map = posterior_setup.parameter_to_solution_map
    parameter_vector = _random_parameter(4)

    log_posterior.evaluate_cost(parameter_vector)
    log_posterior.evaluate_gradient(parameter_vector)
    log_posterior.evaluate_gradient(parameter_vector)
    log_posterior.evaluate_cost(parameter_vector)

    assert parameter_to_solution_map.num_forward_evaluations == 1
    assert parameter_to_solution_map.num_gradient_evaluations == 1


# --------------------------------------------------------------------------------------------------
def test_posterior_recomputes_for_new_parameter(
    posterior_setup: helpers.PosteriorSetup,
) -> None:
    log_posterior = _build_posterior(posterior_setup)
    parameter_to_solution_map = posterior_setup.parameter_to_solution_map

    log_posterior.evaluate_gradient(_random_parameter(4))
    log_posterior.evaluate_gradient(_random_parameter(5))

    assert parameter_to_solution_map.num_forward_evaluations == 2
    assert parameter_to_solution_map.num_gradient_evaluations == 2


# --------------------------------------------------------------------------------------------------
def test_posterior_detects_in_place_parameter_update(
    posterior_setup: helpers.PosteriorSetup,
) -> None:
    """Optimizers may reuse the parameter buffer, which must not return stale results."""
    log_posterior = _build_posterior(posterior_setup)
    parameter_vector = _random_parameter(4)
    log_posterior.evaluate_cost(parameter_vector)

    parameter_vector += 1.0
    cost = log_posterior.evaluate_cost(parameter_vector)

    expected_cost = _build_posterior(posterior_setup).evaluate_cost(parameter_vector.copy())
    np.testing.assert_allclose(cost, expected_cost)


# --------------------------------------------------------------------------------------------------
def test_posterior_split_gradient_is_writeable(
    posterior_setup: helpers.PosteriorSetup,
) -> None:
    """Returned arrays are owned by the caller and do not alias the cache."""
    log_posterior = _build_posterior(posterior_setup)
    parameter_vector = _random_parameter(4)
    likelihood_gradient, _ = log_posterior.evaluate_gradient(parameter_vector, split=True)
    expected_gradient = likelihood_gradient.copy()

    likelihood_gradient[:] = 0.0
    recomputed_gradient, _ = log_posterior.evaluate_gradient(parameter_vector, split=True)

    np.testing.assert_allclose(recomputed_gradient, expected_gradient)


# --------------------------------------------------------------------------------------------------
def test_posterior_logs_evaluations(
    posterior_setup: helpers.PosteriorSetup, tmp_path: Path
) -> None:
    logfile_path = tmp_path / "posterior.log"
    logger_settings = LoggerSettings(print_to_console=False, logfile_path=logfile_path)
    parameter_vector = _random_parameter(4)

    with BaseLogger(logger_settings, prefix="posterior") as logger:
        log_posterior = _build_posterior(posterior_setup, logger=logger)
        log_posterior.evaluate_cost(parameter_vector)
        log_posterior.evaluate_gradient(parameter_vector)

    log_content = logfile_path.read_text()
    assert "total_cost" in log_content
    assert "prior_gradient" in log_content


# --------------------------------------------------------------------------------------------------
def test_posterior_hessian_vector_product_not_implemented(
    posterior_setup: helpers.PosteriorSetup,
) -> None:
    log_posterior = _build_posterior(posterior_setup)
    parameter_vector = _random_parameter(4)

    with pytest.raises(NotImplementedError):
        log_posterior.evaluate_hessian_vector_product(parameter_vector, parameter_vector)


# --------------------------------------------------------------------------------------------------
def test_posterior_gradient_matches_finite_difference_for_nonlinear_forward_map(
    nonlinear_posterior_setup: helpers.PosteriorSetup,
) -> None:
    """Catches an argument-order bug in the call to `ParameterToSolutionMap.evaluate_gradient`.

    `LinearParameterToSolutionMap.evaluate_gradient` ignores its `solution_vector` and
    `parameter_vector` arguments entirely, so a bug swapping their positions when
    `LogPosterior` calls the interface would go undetected with a linear forward map. The
    nonlinear map's Jacobian genuinely depends on `parameter_vector`'s value, so such a swap
    either raises a shape mismatch (parameter and solution dimensions differ) or produces a
    gradient inconsistent with the finite-difference oracle.
    """
    log_posterior = _build_posterior(nonlinear_posterior_setup)
    parameter_vector = _random_parameter(4)

    gradient = log_posterior.evaluate_gradient(parameter_vector)

    expected_gradient = helpers.central_difference_gradient(
        log_posterior.evaluate_cost, parameter_vector, helpers.FINITE_DIFFERENCE_STEP
    )
    np.testing.assert_allclose(
        gradient, expected_gradient, rtol=FINITE_DIFFERENCE_RELATIVE_TOLERANCE
    )


# --------------------------------------------------------------------------------------------------
def test_posterior_does_not_mutate_input_parameter_vector(
    posterior_setup: helpers.PosteriorSetup,
) -> None:
    """The posterior must not mutate the caller's array, distinct from detecting external
    mutation between calls (covered by `test_posterior_detects_in_place_parameter_update`)."""
    log_posterior = _build_posterior(posterior_setup)
    parameter_vector = _random_parameter(4)
    original_parameter_vector = parameter_vector.copy()

    log_posterior.evaluate_cost(parameter_vector)
    log_posterior.evaluate_gradient(parameter_vector)

    np.testing.assert_array_equal(parameter_vector, original_parameter_vector)


# --------------------------------------------------------------------------------------------------
def test_posterior_split_cost_sums_to_total_for_nonlinear_forward_map(
    nonlinear_posterior_setup: helpers.PosteriorSetup,
) -> None:
    log_posterior = _build_posterior(nonlinear_posterior_setup)
    parameter_vector = _random_parameter(4)

    likelihood_cost, prior_cost = log_posterior.evaluate_cost(parameter_vector, split=True)
    total_cost = log_posterior.evaluate_cost(parameter_vector)

    np.testing.assert_allclose(total_cost, likelihood_cost + prior_cost)
