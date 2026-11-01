import numpy as np
import pytest

from ls_bayesian.optimization.components.truncated_cg import (
    TruncatedCGSettings,
    TruncatedConjugateGradient,
    TruncationReason,
)
from tests.optimization import helpers

pytestmark = pytest.mark.unit

DIMENSION = 6
TOLERANCE = 1e-10


def _euclidean(first: np.ndarray, second: np.ndarray) -> float:
    return float(first @ second)


# ==================================================================================================
def test_solves_spd_system_exactly_within_dimension_iterations() -> None:
    rng = np.random.default_rng(0)
    matrix = helpers.random_spd_matrix(rng, DIMENSION)
    gradient = rng.normal(size=DIMENSION)
    solver = TruncatedConjugateGradient(TruncatedCGSettings())

    result = solver.solve(lambda v: matrix @ v, gradient, _euclidean, TOLERANCE)

    np.testing.assert_allclose(result.direction, -np.linalg.solve(matrix, gradient), atol=1e-8)
    assert result.reason is TruncationReason.CONVERGED
    assert result.num_iterations <= DIMENSION


# --------------------------------------------------------------------------------------------------
def test_solves_in_weighted_inner_product() -> None:
    rng = np.random.default_rng(1)
    hessian = helpers.random_spd_matrix(rng, DIMENSION)
    weight = helpers.random_spd_matrix(rng, DIMENSION)
    euclidean_gradient = rng.normal(size=DIMENSION)
    # Riesz representers under (u, v)_W = u^T W v
    representer_gradient = np.linalg.solve(weight, euclidean_gradient)

    def hessian_action(direction: np.ndarray) -> np.ndarray:
        return np.linalg.solve(weight, hessian @ direction)

    solver = TruncatedConjugateGradient(TruncatedCGSettings())
    result = solver.solve(
        hessian_action, representer_gradient, lambda a, b: float(a @ weight @ b), TOLERANCE
    )

    np.testing.assert_allclose(
        result.direction, -np.linalg.solve(hessian, euclidean_gradient), atol=1e-8
    )


# --------------------------------------------------------------------------------------------------
def test_stops_at_relative_tolerance() -> None:
    rng = np.random.default_rng(2)
    matrix = helpers.random_spd_matrix(rng, DIMENSION)
    gradient = rng.normal(size=DIMENSION)
    solver = TruncatedConjugateGradient(TruncatedCGSettings())

    loose = solver.solve(lambda v: matrix @ v, gradient, _euclidean, 0.5)
    tight = solver.solve(lambda v: matrix @ v, gradient, _euclidean, 1e-8)

    assert loose.residual_norm <= 0.5 * np.linalg.norm(gradient)
    assert loose.num_iterations < tight.num_iterations


# --------------------------------------------------------------------------------------------------
def test_negative_curvature_in_first_iteration_returns_steepest_descent() -> None:
    gradient = np.array([1.0, -2.0, 0.5])
    solver = TruncatedConjugateGradient(TruncatedCGSettings())

    result = solver.solve(lambda v: -v, gradient, _euclidean, 1e-6)

    np.testing.assert_allclose(result.direction, -gradient)
    assert result.reason is TruncationReason.NON_POSITIVE_CURVATURE


# --------------------------------------------------------------------------------------------------
def test_negative_curvature_later_returns_current_iterate_and_descent_direction() -> None:
    matrix = np.diag([4.0, 1.0, -1.0])
    gradient = np.array([1.0, 1.0, 1.0])
    solver = TruncatedConjugateGradient(TruncatedCGSettings())

    result = solver.solve(lambda v: matrix @ v, gradient, _euclidean, 1e-12)

    assert result.reason is TruncationReason.NON_POSITIVE_CURVATURE
    assert result.num_iterations > 1
    assert _euclidean(gradient, result.direction) < 0.0


# --------------------------------------------------------------------------------------------------
def test_iteration_cap_is_reported() -> None:
    rng = np.random.default_rng(3)
    matrix = helpers.random_spd_matrix(rng, DIMENSION)
    gradient = rng.normal(size=DIMENSION)
    solver = TruncatedConjugateGradient(TruncatedCGSettings(maximum_num_iterations=1))

    result = solver.solve(lambda v: matrix @ v, gradient, _euclidean, 1e-12)

    assert result.reason is TruncationReason.MAXIMUM_ITERATIONS
    assert result.num_iterations == 1


# --------------------------------------------------------------------------------------------------
def test_zero_gradient_needs_no_iterations() -> None:
    solver = TruncatedConjugateGradient(TruncatedCGSettings())

    result = solver.solve(lambda v: v, np.zeros(3), _euclidean, 1e-6)

    assert result.num_iterations == 0
    np.testing.assert_allclose(result.direction, np.zeros(3))


# ==================================================================================================
def test_trust_region_boundary_step_has_the_radius_as_norm() -> None:
    rng = np.random.default_rng(4)
    matrix = helpers.random_spd_matrix(rng, DIMENSION)
    gradient = rng.normal(size=DIMENSION)
    solver = TruncatedConjugateGradient(TruncatedCGSettings())
    full = solver.solve(lambda v: matrix @ v, gradient, _euclidean, TOLERANCE)
    radius = 0.4 * np.linalg.norm(full.direction)

    result = solver.solve(
        lambda v: matrix @ v, gradient, _euclidean, TOLERANCE, trust_region_radius=radius
    )

    assert result.reason is TruncationReason.TRUST_REGION_BOUNDARY
    assert np.linalg.norm(result.direction) == pytest.approx(radius, rel=1e-10)
    assert _euclidean(gradient, result.direction) < 0.0


# --------------------------------------------------------------------------------------------------
def test_large_trust_region_gives_the_unconstrained_solution() -> None:
    rng = np.random.default_rng(5)
    matrix = helpers.random_spd_matrix(rng, DIMENSION)
    gradient = rng.normal(size=DIMENSION)
    solver = TruncatedConjugateGradient(TruncatedCGSettings())

    unconstrained = solver.solve(lambda v: matrix @ v, gradient, _euclidean, TOLERANCE)
    constrained = solver.solve(
        lambda v: matrix @ v, gradient, _euclidean, TOLERANCE, trust_region_radius=1e6
    )

    np.testing.assert_allclose(constrained.direction, unconstrained.direction)
    assert constrained.reason is TruncationReason.CONVERGED


# --------------------------------------------------------------------------------------------------
def test_tiny_trust_region_gives_a_scaled_steepest_descent_step() -> None:
    rng = np.random.default_rng(6)
    matrix = helpers.random_spd_matrix(rng, DIMENSION)
    gradient = rng.normal(size=DIMENSION)
    radius = 1e-3
    solver = TruncatedConjugateGradient(TruncatedCGSettings())

    result = solver.solve(
        lambda v: matrix @ v, gradient, _euclidean, TOLERANCE, trust_region_radius=radius
    )

    assert result.num_iterations == 1
    np.testing.assert_allclose(
        result.direction, -radius * gradient / np.linalg.norm(gradient), rtol=1e-12
    )


# --------------------------------------------------------------------------------------------------
def test_negative_curvature_with_trust_region_goes_to_the_boundary() -> None:
    matrix = np.diag([4.0, 1.0, -1.0])
    gradient = np.array([1.0, 1.0, 1.0])
    radius = 2.0
    solver = TruncatedConjugateGradient(TruncatedCGSettings())

    result = solver.solve(
        lambda v: matrix @ v, gradient, _euclidean, 1e-12, trust_region_radius=radius
    )

    assert result.reason is TruncationReason.NON_POSITIVE_CURVATURE
    assert np.linalg.norm(result.direction) == pytest.approx(radius, rel=1e-10)
    assert _euclidean(gradient, result.direction) < 0.0


# --------------------------------------------------------------------------------------------------
def test_returned_residual_gives_the_quadratic_model_value() -> None:
    rng = np.random.default_rng(7)
    matrix = helpers.random_spd_matrix(rng, DIMENSION)
    gradient = rng.normal(size=DIMENSION)
    solver = TruncatedConjugateGradient(TruncatedCGSettings())
    full = solver.solve(lambda v: matrix @ v, gradient, _euclidean, TOLERANCE)

    for radius in (None, 0.3 * np.linalg.norm(full.direction)):
        result = solver.solve(
            lambda v: matrix @ v, gradient, _euclidean, 1e-3, trust_region_radius=radius
        )
        direction = result.direction
        model_value = gradient @ direction + 0.5 * direction @ matrix @ direction
        assert result.residual is not None
        assert 0.5 * (gradient @ direction + direction @ result.residual) == pytest.approx(
            model_value, rel=1e-10
        )
        np.testing.assert_allclose(result.residual, matrix @ direction + gradient, atol=1e-10)


# --------------------------------------------------------------------------------------------------
def test_non_positive_radius_is_rejected() -> None:
    solver = TruncatedConjugateGradient(TruncatedCGSettings())
    with pytest.raises(ValueError, match="trust_region_radius"):
        solver.solve(lambda v: v, np.ones(3), _euclidean, 1e-6, trust_region_radius=0.0)
