import numpy as np
import pytest
import scipy.sparse as sp
from beartype.roar import BeartypeCallHintViolation

from ls_bayesian.posterior import likelihood
from tests.posterior import helpers

pytestmark = pytest.mark.unit

# Pure float64 linear-algebra identities, no discretization or statistical error: machine
# precision is the only source of slack.
LINEAR_ALGEBRA_RELATIVE_TOLERANCE = 1e-10
# Central differences are only accurate up to floating-point cancellation error, which grows as
# the step size shrinks; this bound leaves headroom above that error for FINITE_DIFFERENCE_STEP.
FINITE_DIFFERENCE_RELATIVE_TOLERANCE = 1e-6


# ==================================================================================================
def test_likelihood_cost(likelihood_setup: helpers.LikelihoodSetup) -> None:
    rng = np.random.default_rng(2)
    solution_vector = rng.random(helpers.NUM_VERTICES)

    cost = likelihood_setup.likelihood.evaluate_cost(solution_vector)

    misfit = likelihood_setup.observation_matrix @ solution_vector - likelihood_setup.data_vector
    expected_cost = 0.5 * misfit @ likelihood_setup.precision_matrix @ misfit
    np.testing.assert_allclose(cost, expected_cost)


# --------------------------------------------------------------------------------------------------
def test_likelihood_gradient(likelihood_setup: helpers.LikelihoodSetup) -> None:
    rng = np.random.default_rng(2)
    solution_vector = rng.random(helpers.NUM_VERTICES)

    gradient = likelihood_setup.likelihood.evaluate_gradient(solution_vector)

    expected_gradient = helpers.central_difference_gradient(
        likelihood_setup.likelihood.evaluate_cost,
        solution_vector,
        helpers.FINITE_DIFFERENCE_STEP,
    )
    np.testing.assert_allclose(
        gradient, expected_gradient, rtol=FINITE_DIFFERENCE_RELATIVE_TOLERANCE
    )


# --------------------------------------------------------------------------------------------------
def test_likelihood_hessian_vector_product(likelihood_setup: helpers.LikelihoodSetup) -> None:
    rng = np.random.default_rng(2)
    solution_vector = rng.random(helpers.NUM_VERTICES)
    direction_vector = rng.random(helpers.NUM_VERTICES)

    hvp = likelihood_setup.likelihood.evaluate_hessian_vector_product(
        solution_vector, direction_vector
    )

    observation_matrix = likelihood_setup.observation_matrix
    expected_hvp = (
        observation_matrix.T
        @ likelihood_setup.precision_matrix
        @ observation_matrix
        @ direction_vector
    )
    np.testing.assert_allclose(hvp, expected_hvp)


# --------------------------------------------------------------------------------------------------
def test_likelihood_hessian_is_derivative_of_gradient(
    likelihood_setup: helpers.LikelihoodSetup,
) -> None:
    """The gradient is affine, so its difference along a direction equals the HVP exactly."""
    rng = np.random.default_rng(2)
    solution_vector = rng.random(helpers.NUM_VERTICES)
    direction_vector = rng.random(helpers.NUM_VERTICES)
    gaussian_likelihood = likelihood_setup.likelihood

    gradient_difference = gaussian_likelihood.evaluate_gradient(
        solution_vector + direction_vector
    ) - gaussian_likelihood.evaluate_gradient(solution_vector)
    hvp = gaussian_likelihood.evaluate_hessian_vector_product(solution_vector, direction_vector)

    np.testing.assert_allclose(hvp, gradient_difference)


# --------------------------------------------------------------------------------------------------
def test_likelihood_init_raises_on_multidimensional_data() -> None:
    with pytest.raises(ValueError, match="Data vector must be one-dimensional"):
        likelihood.GaussianLogLikelihood(
            np.zeros((2, 1)), sp.coo_matrix(np.eye(2)), sp.coo_matrix(np.eye(2))
        )


# --------------------------------------------------------------------------------------------------
def test_likelihood_init_raises_on_observation_matrix_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Observation matrix shape"):
        likelihood.GaussianLogLikelihood(
            np.zeros(2), sp.coo_matrix(np.eye(3)), sp.coo_matrix(np.eye(2))
        )


# --------------------------------------------------------------------------------------------------
def test_likelihood_init_raises_on_precision_matrix_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Precision matrix shape"):
        likelihood.GaussianLogLikelihood(
            np.zeros(2), sp.coo_matrix(np.eye(2)), sp.coo_matrix(np.eye(3))
        )


# --------------------------------------------------------------------------------------------------
def test_likelihood_raises_on_solution_shape_mismatch(
    likelihood_setup: helpers.LikelihoodSetup,
) -> None:
    wrong_solution_vector = np.zeros(helpers.NUM_VERTICES + 1)

    with pytest.raises(ValueError, match="Solution-space vector shape"):
        likelihood_setup.likelihood.evaluate_cost(wrong_solution_vector)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_likelihood_init_raises_on_non_finite_data(invalid_value: float) -> None:
    with pytest.raises(ValueError, match="Data vector must be finite"):
        likelihood.GaussianLogLikelihood(
            np.array([1.0, invalid_value]), sp.coo_matrix(np.eye(2)), sp.coo_matrix(np.eye(2))
        )


# --------------------------------------------------------------------------------------------------
def test_likelihood_raises_on_direction_shape_mismatch(
    likelihood_setup: helpers.LikelihoodSetup,
) -> None:
    rng = np.random.default_rng(2)
    solution_vector = rng.random(helpers.NUM_VERTICES)
    wrong_direction_vector = np.zeros(helpers.NUM_VERTICES + 1)

    with pytest.raises(ValueError, match="Solution-space vector shape"):
        likelihood_setup.likelihood.evaluate_hessian_vector_product(
            solution_vector, wrong_direction_vector
        )


# --------------------------------------------------------------------------------------------------
def test_likelihood_cost_with_dense_correlated_precision_matrix(
    dense_spd_likelihood_setup: helpers.LikelihoodSetup,
) -> None:
    """Cost/gradient/HVP must use the full precision matrix, not assume a diagonal one."""
    rng = np.random.default_rng(5)
    solution_vector = rng.random(helpers.NUM_VERTICES)
    direction_vector = rng.random(helpers.NUM_VERTICES)
    setup = dense_spd_likelihood_setup

    cost = setup.likelihood.evaluate_cost(solution_vector)
    gradient = setup.likelihood.evaluate_gradient(solution_vector)
    hvp = setup.likelihood.evaluate_hessian_vector_product(solution_vector, direction_vector)

    misfit = setup.observation_matrix @ solution_vector - setup.data_vector
    expected_cost = 0.5 * misfit @ setup.precision_matrix @ misfit
    expected_gradient = setup.observation_matrix.T @ (setup.precision_matrix @ misfit)
    expected_hvp = setup.observation_matrix.T @ (
        setup.precision_matrix @ (setup.observation_matrix @ direction_vector)
    )
    np.testing.assert_allclose(cost, expected_cost, rtol=LINEAR_ALGEBRA_RELATIVE_TOLERANCE)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=LINEAR_ALGEBRA_RELATIVE_TOLERANCE)
    np.testing.assert_allclose(hvp, expected_hvp, rtol=LINEAR_ALGEBRA_RELATIVE_TOLERANCE)


# ==================================================================================================
def test_observation_matrix_extracts_observed_vertices() -> None:
    rng = np.random.default_rng(3)
    solution_vector = rng.random(helpers.NUM_VERTICES)

    observation_matrix = likelihood.assemble_vertex_observation_matrix(
        helpers.NUM_VERTICES, helpers.OBSERVED_VERTEX_INDICES
    )

    assert observation_matrix.shape == (
        helpers.OBSERVED_VERTEX_INDICES.shape[0],
        helpers.NUM_VERTICES,
    )
    np.testing.assert_allclose(
        observation_matrix @ solution_vector,
        solution_vector[helpers.OBSERVED_VERTEX_INDICES],
    )


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("invalid_index", [-1, helpers.NUM_VERTICES])
def test_observation_matrix_raises_on_out_of_range_index(invalid_index: int) -> None:
    with pytest.raises(ValueError, match="lie outside of"):
        likelihood.assemble_vertex_observation_matrix(
            helpers.NUM_VERTICES, np.array([0, invalid_index])
        )


# --------------------------------------------------------------------------------------------------
def test_observation_matrix_raises_on_multidimensional_indices() -> None:
    with pytest.raises(ValueError, match="must be one-dimensional"):
        likelihood.assemble_vertex_observation_matrix(
            helpers.NUM_VERTICES, np.zeros((2, 1), dtype=np.int64)
        )


# --------------------------------------------------------------------------------------------------
def test_observation_matrix_rejects_non_integer_indices() -> None:
    with pytest.raises(BeartypeCallHintViolation):
        likelihood.assemble_vertex_observation_matrix(helpers.NUM_VERTICES, np.array([0.0, 1.5]))


# --------------------------------------------------------------------------------------------------
def test_observation_matrix_raises_on_duplicate_index() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        likelihood.assemble_vertex_observation_matrix(helpers.NUM_VERTICES, np.array([0, 2, 2, 4]))


# --------------------------------------------------------------------------------------------------
def test_observation_matrix_raises_on_non_positive_vertex_number() -> None:
    with pytest.raises(BeartypeCallHintViolation):
        likelihood.assemble_vertex_observation_matrix(0, np.array([], dtype=np.int64))


# --------------------------------------------------------------------------------------------------
def test_observation_matrix_handles_zero_observations() -> None:
    observation_matrix = likelihood.assemble_vertex_observation_matrix(
        helpers.NUM_VERTICES, np.array([], dtype=np.int64)
    )

    assert observation_matrix.shape == (0, helpers.NUM_VERTICES)
    assert observation_matrix.nnz == 0


# --------------------------------------------------------------------------------------------------
def test_precision_matrix_is_diagonal() -> None:
    precision_values = np.array([0.5, 1.0, 4.0])

    precision_matrix = likelihood.assemble_diagonal_precision_matrix(precision_values)

    np.testing.assert_allclose(precision_matrix.toarray(), np.diag(precision_values))


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("invalid_value", [0.0, -1.0, np.nan, np.inf])
def test_precision_matrix_raises_on_invalid_values(invalid_value: float) -> None:
    with pytest.raises(ValueError, match="strictly positive and finite"):
        likelihood.assemble_diagonal_precision_matrix(np.array([1.0, invalid_value]))


# --------------------------------------------------------------------------------------------------
def test_precision_matrix_handles_zero_observations() -> None:
    precision_matrix = likelihood.assemble_diagonal_precision_matrix(np.array([], dtype=np.float64))

    assert precision_matrix.shape == (0, 0)
    assert precision_matrix.nnz == 0


# ==================================================================================================
def test_likelihood_from_vertex_observations(
    likelihood_setup: helpers.LikelihoodSetup,
) -> None:
    rng = np.random.default_rng(3)
    solution_vector = rng.random(helpers.NUM_VERTICES)
    direction_vector = rng.random(helpers.NUM_VERTICES)
    reference_likelihood = likelihood_setup.likelihood

    created_likelihood = likelihood.GaussianLogLikelihood.from_vertex_observations(
        likelihood.VertexObservationSettings(
            likelihood_setup.data_vector,
            helpers.NUM_VERTICES,
            helpers.OBSERVED_VERTEX_INDICES,
            likelihood_setup.precision_values,
        )
    )

    assert isinstance(created_likelihood, likelihood.GaussianLogLikelihood)
    np.testing.assert_allclose(
        created_likelihood.evaluate_cost(solution_vector),
        reference_likelihood.evaluate_cost(solution_vector),
    )
    np.testing.assert_allclose(
        created_likelihood.evaluate_gradient(solution_vector),
        reference_likelihood.evaluate_gradient(solution_vector),
    )
    np.testing.assert_allclose(
        created_likelihood.evaluate_hessian_vector_product(solution_vector, direction_vector),
        reference_likelihood.evaluate_hessian_vector_product(solution_vector, direction_vector),
    )


# --------------------------------------------------------------------------------------------------
def test_likelihood_from_vertex_observations_raises_on_non_positive_vertex_number(
    likelihood_setup: helpers.LikelihoodSetup,
) -> None:
    with pytest.raises(BeartypeCallHintViolation):
        likelihood.GaussianLogLikelihood.from_vertex_observations(
            likelihood.VertexObservationSettings(
                likelihood_setup.data_vector,
                0,
                helpers.OBSERVED_VERTEX_INDICES,
                likelihood_setup.precision_values,
            )
        )


# --------------------------------------------------------------------------------------------------
def test_likelihood_from_vertex_observations_propagates_data_length_mismatch(
    likelihood_setup: helpers.LikelihoodSetup,
) -> None:
    """`from_vertex_observations` must surface the constructor's own shape validation, not mask
    it."""
    with pytest.raises(ValueError, match="Observation matrix shape"):
        likelihood.GaussianLogLikelihood.from_vertex_observations(
            likelihood.VertexObservationSettings(
                likelihood_setup.data_vector[:-1],
                helpers.NUM_VERTICES,
                helpers.OBSERVED_VERTEX_INDICES,
                likelihood_setup.precision_values,
            )
        )
