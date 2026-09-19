r"""Unit tests of the SPDE prior on exact, dense operators in DoF space.

The precision, covariance and covariance factorization are dense matrices wrapped in
[`Matrix`][ls_bayesian.spde_prior.components.Matrix] components, so that all relations hold up to
roundoff. The prior itself only composes these operators with the vertex/DoF conversion.
"""

from collections.abc import Callable

import numpy as np
import pytest

from ls_bayesian.spde_prior import spde_prior
from tests.spde_prior import helpers

pytestmark = pytest.mark.unit

SINGLE_CASE = pytest.mark.parametrize("fem_case", ["1d_p1"], indirect=True)
P1_CASES = pytest.mark.parametrize("fem_case", helpers.P1_CASE_IDS, indirect=True)
CONSTANT_SHIFT = 1.7


# ==================================================================================================
def exact_prior(setup: helpers.ExactPriorSetup, seed: int = 0) -> spde_prior.SPDEPrior:
    return helpers.create_exact_spde_prior(setup, seed)


def vertex_space_operator(setup: helpers.ExactPriorSetup, dof_operator: np.ndarray) -> np.ndarray:
    """Similarity transform of a DoF-space operator to vertex space, valid for P1 spaces."""
    selection_matrix = helpers.vertex_to_dof_selection_matrix(setup.fem_space_setup.function_space)
    return selection_matrix.T @ dof_operator @ selection_matrix


# ==================================================================================================
def test_spde_prior_cost_and_gradient_vanish_at_mean(
    exact_prior_setup: helpers.ExactPriorSetup,
) -> None:
    prior = exact_prior(exact_prior_setup)

    cost = prior.evaluate_cost(exact_prior_setup.mean_vector)
    gradient = prior.evaluate_gradient(exact_prior_setup.mean_vector)

    scale = np.linalg.norm(exact_prior_setup.precision_matrix) * np.linalg.norm(
        exact_prior_setup.mean_vector
    )
    helpers.assert_allclose_normwise(cost, 0.0, helpers.ROUNDOFF_TOLERANCE, scale=scale)
    helpers.assert_allclose_normwise(gradient, 0.0, helpers.ROUNDOFF_TOLERANCE, scale=scale)


# --------------------------------------------------------------------------------------------------
def test_spde_prior_cost_of_constant_shift(exact_prior_setup: helpers.ExactPriorSetup) -> None:
    """A constant vertex field is interpolated to constant DoFs for every Lagrange degree."""
    prior = exact_prior(exact_prior_setup)
    shifted_parameter = exact_prior_setup.mean_vector + CONSTANT_SHIFT

    cost = prior.evaluate_cost(shifted_parameter)

    constant_dofs = np.ones(exact_prior_setup.precision_matrix.shape[0])
    expected_cost = (
        0.5 * CONSTANT_SHIFT**2 * constant_dofs @ exact_prior_setup.precision_matrix @ constant_dofs
    )
    helpers.assert_allclose_normwise(cost, expected_cost, helpers.ROUNDOFF_TOLERANCE)


# --------------------------------------------------------------------------------------------------
def test_spde_prior_second_order_taylor_expansion_is_exact(
    exact_prior_setup: helpers.ExactPriorSetup,
) -> None:
    r"""The cost is quadratic, so $J(m+d) = J(m) + g(m)^T d + \frac{1}{2} d^T H d$ exactly."""
    prior = exact_prior(exact_prior_setup)
    rng = np.random.default_rng(1)
    parameter, direction = rng.random((2, exact_prior_setup.mean_vector.size))

    cost_at_parameter = prior.evaluate_cost(parameter)
    first_order_term = prior.evaluate_gradient(parameter) @ direction
    second_order_term = 0.5 * direction @ prior.evaluate_hessian_vector_product(direction)
    cost_at_shifted_parameter = prior.evaluate_cost(parameter + direction)

    taylor_expansion = cost_at_parameter + first_order_term + second_order_term
    scale = sum(
        abs(term)
        for term in (
            cost_at_parameter,
            first_order_term,
            second_order_term,
            cost_at_shifted_parameter,
        )
    )
    helpers.assert_allclose_normwise(
        cost_at_shifted_parameter, taylor_expansion, helpers.ROUNDOFF_TOLERANCE, scale=scale
    )


# --------------------------------------------------------------------------------------------------
def test_spde_prior_hessian_vector_product_is_gradient_difference(
    exact_prior_setup: helpers.ExactPriorSetup,
) -> None:
    """The gradient is affine, so its difference along a direction equals the Hessian product."""
    prior = exact_prior(exact_prior_setup)
    rng = np.random.default_rng(1)
    parameter, direction = rng.random((2, exact_prior_setup.mean_vector.size))

    hessian_vector_product = prior.evaluate_hessian_vector_product(direction)

    gradient_difference = prior.evaluate_gradient(parameter + direction) - prior.evaluate_gradient(
        parameter
    )
    helpers.assert_allclose_normwise(
        hessian_vector_product, gradient_difference, helpers.ROUNDOFF_TOLERANCE
    )


# --------------------------------------------------------------------------------------------------
@P1_CASES
def test_spde_prior_covariance_inverts_precision(
    exact_prior_setup: helpers.ExactPriorSetup,
) -> None:
    prior = exact_prior(exact_prior_setup)
    vertex_vector = np.random.default_rng(1).random(exact_prior_setup.mean_vector.size)

    recovered_vector = prior.apply_covariance_operator(
        prior.apply_precision_operator(vertex_vector)
    )

    helpers.assert_allclose_normwise(recovered_vector, vertex_vector, helpers.ROUNDOFF_TOLERANCE)


# --------------------------------------------------------------------------------------------------
@P1_CASES
def test_spde_prior_factorization_reproduces_covariance(
    exact_prior_setup: helpers.ExactPriorSetup,
) -> None:
    prior = exact_prior(exact_prior_setup)
    num_vertices = exact_prior_setup.mean_vector.size

    factorization = helpers.materialize_operator(
        prior.apply_covariance_factorization, prior.random_vector_size
    )
    covariance = helpers.materialize_operator(prior.apply_covariance_operator, num_vertices)

    helpers.assert_allclose_normwise(
        factorization @ factorization.T, covariance, helpers.ROUNDOFF_TOLERANCE
    )


# --------------------------------------------------------------------------------------------------
def test_spde_prior_random_vector_size_matches_factorization(
    exact_prior_setup: helpers.ExactPriorSetup,
) -> None:
    prior = exact_prior(exact_prior_setup)

    assert prior.random_vector_size == exact_prior_setup.covariance_factorization_matrix.shape[1]


# --------------------------------------------------------------------------------------------------
@SINGLE_CASE
@pytest.mark.parametrize(
    "comparison", ["same_seed_identical", "successive_samples_differ", "different_seeds_differ"]
)
def test_spde_prior_samples_are_reproducible(
    exact_prior_setup: helpers.ExactPriorSetup, comparison: str
) -> None:
    prior = exact_prior(exact_prior_setup, seed=0)

    first_sample = prior.generate_sample()

    if comparison == "same_seed_identical":
        np.testing.assert_array_equal(
            first_sample, exact_prior(exact_prior_setup, seed=0).generate_sample()
        )
    elif comparison == "successive_samples_differ":
        assert not np.allclose(first_sample, prior.generate_sample())
    else:
        assert not np.allclose(
            first_sample, exact_prior(exact_prior_setup, seed=1).generate_sample()
        )


# --------------------------------------------------------------------------------------------------
@P1_CASES
def test_spde_prior_whitened_samples_are_standard_normal(
    exact_prior_setup: helpers.ExactPriorSetup,
) -> None:
    prior = exact_prior(exact_prior_setup, seed=2)

    samples = np.stack([prior.generate_sample() for _ in range(helpers.NUM_SAMPLES)])

    mean_z_score, variance_z_score = helpers.whitened_sample_z_scores(
        samples,
        exact_prior_setup.mean_vector,
        vertex_space_operator(exact_prior_setup, exact_prior_setup.covariance_matrix),
    )
    assert abs(mean_z_score) < helpers.STATISTICAL_Z_THRESHOLD
    assert abs(variance_z_score) < helpers.STATISTICAL_Z_THRESHOLD


# --------------------------------------------------------------------------------------------------
@SINGLE_CASE
@pytest.mark.parametrize(
    ("operator_name", "message"),
    [
        ("precision_matrix", "Precision operator shape"),
        ("covariance_matrix", "Covariance operator shape"),
        ("covariance_factorization_matrix", "Covariance factorization output dimension"),
    ],
    ids=["precision", "covariance", "covariance_factorization"],
)
def test_spde_prior_rejects_mismatched_operator(
    exact_prior_setup: helpers.ExactPriorSetup, operator_name: str, message: str
) -> None:
    num_dofs = exact_prior_setup.precision_matrix.shape[0]
    wrong_shape_operator = np.zeros((num_dofs + 1, num_dofs))

    with pytest.raises(ValueError, match=message):
        helpers.create_exact_spde_prior(
            exact_prior_setup, seed=0, **{operator_name: wrong_shape_operator}
        )


# --------------------------------------------------------------------------------------------------
@SINGLE_CASE
@pytest.mark.parametrize(
    "method_name",
    [
        "evaluate_cost",
        "evaluate_gradient",
        "evaluate_hessian_vector_product",
        "apply_covariance_operator",
        "apply_precision_operator",
    ],
)
def test_spde_prior_rejects_wrong_length_parameter_vector(
    exact_prior_setup: helpers.ExactPriorSetup, method_name: str
) -> None:
    prior = exact_prior(exact_prior_setup)
    wrong_length_vector = np.zeros(exact_prior_setup.mean_vector.size + 1)

    with pytest.raises(ValueError, match="Expected vertex_values to have shape"):
        getattr(prior, method_name)(wrong_length_vector)


# --------------------------------------------------------------------------------------------------
@SINGLE_CASE
def test_spde_prior_rejects_wrong_length_mean_vector(
    exact_prior_setup: helpers.ExactPriorSetup,
) -> None:
    wrong_length_mean = np.zeros(exact_prior_setup.mean_vector.size + 1)

    with pytest.raises(ValueError, match="Expected vertex_values to have shape"):
        helpers.create_exact_spde_prior(exact_prior_setup, seed=0, mean_vector=wrong_length_mean)


# --------------------------------------------------------------------------------------------------
@SINGLE_CASE
def test_spde_prior_rejects_wrong_length_random_vector(
    exact_prior_setup: helpers.ExactPriorSetup,
) -> None:
    prior = exact_prior(exact_prior_setup)

    with pytest.raises(ValueError, match="Random vector shape"):
        prior.apply_covariance_factorization(np.zeros(prior.random_vector_size + 1))


# --------------------------------------------------------------------------------------------------
@SINGLE_CASE
def test_spde_prior_cost_raises_on_indefinite_precision(
    exact_prior_setup: helpers.ExactPriorSetup,
) -> None:
    """A negative cost signals an invalid precision operator, and used to be a bare `assert`."""
    prior = helpers.create_exact_spde_prior(
        exact_prior_setup, seed=0, precision_matrix=-exact_prior_setup.precision_matrix
    )

    with pytest.raises(RuntimeError, match="Prior cost is negative"):
        prior.evaluate_cost(exact_prior_setup.mean_vector + CONSTANT_SHIFT)


# --------------------------------------------------------------------------------------------------
@SINGLE_CASE
@pytest.mark.parametrize(
    "method_name",
    [
        "evaluate_gradient",
        "evaluate_hessian_vector_product",
        "apply_covariance_operator",
        "apply_precision_operator",
        "apply_covariance_factorization",
        "generate_sample",
    ],
)
def test_spde_prior_outputs_do_not_share_memory(
    exact_prior_setup: helpers.ExactPriorSetup, method_name: str
) -> None:
    """A result must not change when the method is called again, i.e. it is no internal buffer."""
    prior = exact_prior(exact_prior_setup)
    method: Callable[..., np.ndarray] = getattr(prior, method_name)
    rng = np.random.default_rng(1)
    input_size = (
        prior.random_vector_size
        if method_name == "apply_covariance_factorization"
        else exact_prior_setup.mean_vector.size
    )
    first_arguments, second_arguments = (
        ((), ())
        if method_name == "generate_sample"
        else ((rng.random(input_size),), (rng.random(input_size),))
    )
    first_result = method(*first_arguments)
    first_result_copy = first_result.copy()

    method(*second_arguments)

    np.testing.assert_array_equal(first_result, first_result_copy)
