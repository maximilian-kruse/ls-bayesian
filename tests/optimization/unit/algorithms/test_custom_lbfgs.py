import numpy as np
import pytest

from ls_bayesian.optimization.algorithms.custom_lbfgs import (
    CorrectionPairStore,
    CustomLBFGSOptimizer,
    CustomLBFGSSettings,
)
from ls_bayesian.optimization.components.cautious_update import (
    AlwaysAcceptStrategy,
    CautiousUpdateSettings,
    CautiousUpdateStrategy,
)
from ls_bayesian.optimization.components.line_search import (
    ArmijoBacktrackingLineSearch,
    ArmijoBacktrackingLineSearchSettings,
)
from tests.optimization import helpers

pytestmark = pytest.mark.unit

CONVERGENCE_ABSOLUTE_TOLERANCE = 1e-3


def _default_optimizer(settings: CustomLBFGSSettings | None = None) -> CustomLBFGSOptimizer:
    return CustomLBFGSOptimizer(
        settings or CustomLBFGSSettings(),
        ArmijoBacktrackingLineSearch(ArmijoBacktrackingLineSearchSettings()),
        CautiousUpdateStrategy(CautiousUpdateSettings()),
    )


# ==================================================================================================
def test_store_evicts_oldest_beyond_memory_size() -> None:
    store = CorrectionPairStore(memory_size=2)
    store.add(np.array([1.0]), np.array([1.0]), inner_product_reciprocal=1.0)
    store.add(np.array([2.0]), np.array([2.0]), inner_product_reciprocal=2.0)
    store.add(np.array([3.0]), np.array([3.0]), inner_product_reciprocal=3.0)

    assert len(store) == 2
    assert [pair.inner_product_reciprocal for pair in store] == [2.0, 3.0]


# --------------------------------------------------------------------------------------------------
def test_store_iterates_in_both_documented_orders() -> None:
    store = CorrectionPairStore(memory_size=3)
    store.add(np.array([1.0]), np.array([1.0]), inner_product_reciprocal=1.0)
    store.add(np.array([2.0]), np.array([2.0]), inner_product_reciprocal=2.0)

    assert [pair.inner_product_reciprocal for pair in store] == [1.0, 2.0]
    assert [pair.inner_product_reciprocal for pair in reversed(store)] == [2.0, 1.0]


# ==================================================================================================
def test_two_loop_recursion_with_no_pairs_is_steepest_descent() -> None:
    optimizer = _default_optimizer()
    store = CorrectionPairStore(memory_size=5)
    gradient = np.array([1.0, -2.0, 3.0])
    model = helpers.ZeroModel()

    direction = optimizer._two_loop_recursion(gradient, store, model)

    np.testing.assert_allclose(direction, -gradient)


# --------------------------------------------------------------------------------------------------
def test_two_loop_recursion_matches_closed_form_bfgs_update_with_one_pair() -> None:
    """With one correction pair `(s, y)` and the identity seed, the two-loop recursion is
    mathematically equivalent to one step of the explicit BFGS inverse-Hessian update
    `H_1 = (I - rho s y^T) H_0 (I - rho y s^T) + rho s s^T` applied to the gradient; this test
    checks that equivalence directly against the closed-form matrix, in the Euclidean inner
    product for an independently-computable reference."""
    optimizer = _default_optimizer()
    model = helpers.ZeroModel()
    state_difference = np.array([1.0, 0.5])
    gradient_difference = np.array([0.3, 0.8])
    inner_product_reciprocal = 1.0 / np.dot(state_difference, gradient_difference)
    store = CorrectionPairStore(memory_size=1)
    store.add(
        state_difference, gradient_difference, inner_product_reciprocal=inner_product_reciprocal
    )
    gradient = np.array([0.2, -0.4])

    direction = optimizer._two_loop_recursion(gradient, store, model)

    identity = np.eye(2)
    updated_inverse_hessian = (
        identity - inner_product_reciprocal * np.outer(state_difference, gradient_difference)
    ) @ (
        identity - inner_product_reciprocal * np.outer(gradient_difference, state_difference)
    ) + inner_product_reciprocal * np.outer(state_difference, state_difference)
    expected_direction = -(updated_inverse_hessian @ gradient)
    np.testing.assert_allclose(direction, expected_direction)


# ==================================================================================================
def test_converges_to_quadratic_minimizer_under_weighted_inner_product(
    quadratic_matrix: np.ndarray, quadratic_minimizer: np.ndarray
) -> None:
    """The model's inner product is weighted by the same matrix `A` as the quadratic loss, so the
    Riesz representer of the gradient under this inner product is simply `m - m*` (see
    `helpers.QuadraticModel`'s docstring): this checks the optimizer actually uses the model's
    non-Euclidean geometry rather than silently falling back to a Euclidean one."""
    optimizer = _default_optimizer(
        CustomLBFGSSettings(maximum_num_iterations=200, gradient_norm_tolerance=1e-8)
    )
    model = helpers.QuadraticModel(
        quadratic_matrix, quadratic_minimizer, inner_product_matrix=quadratic_matrix
    )
    initial_guess = np.random.default_rng(7).normal(size=quadratic_minimizer.shape)

    result = optimizer.run(initial_guess, model)

    assert result.success
    np.testing.assert_allclose(
        result.result, quadratic_minimizer, atol=CONVERGENCE_ABSOLUTE_TOLERANCE
    )


# --------------------------------------------------------------------------------------------------
def test_converges_on_rosenbrock_with_euclidean_space() -> None:
    """Cross-check against `ScipyLBFGSBOptimizer` on the same problem: with the default (Euclidean)
    model, this backend should also converge to the known minimizer."""
    optimizer = _default_optimizer(
        CustomLBFGSSettings(maximum_num_iterations=500, gradient_norm_tolerance=1e-6)
    )
    initial_guess = np.array([-1.2, 1.0, -1.0, 1.5])
    model = helpers.RosenbrockModel()

    result = optimizer.run(initial_guess, model)

    assert result.success
    np.testing.assert_allclose(
        result.result, np.ones_like(initial_guess), atol=CONVERGENCE_ABSOLUTE_TOLERANCE
    )


# --------------------------------------------------------------------------------------------------
def test_stops_at_maximum_num_iterations_when_not_converged() -> None:
    optimizer = _default_optimizer(
        CustomLBFGSSettings(maximum_num_iterations=1, gradient_norm_tolerance=1e-12)
    )
    initial_guess = np.array([-1.2, 1.0])
    model = helpers.RosenbrockModel()

    result = optimizer.run(initial_guess, model)

    assert not result.success
    assert result.num_iterations == 1


# --------------------------------------------------------------------------------------------------
def test_zero_iterations_when_initial_guess_already_converged() -> None:
    optimizer = _default_optimizer(CustomLBFGSSettings(gradient_norm_tolerance=1e-6))
    minimizer = np.zeros(2)
    model = helpers.QuadraticModel(np.eye(2), minimizer)

    result = optimizer.run(minimizer, model)

    assert result.success
    assert result.num_iterations == 0


# --------------------------------------------------------------------------------------------------
def test_raises_on_non_positive_curvature_pair_accepted_by_a_permissive_strategy() -> None:
    """`AlwaysAcceptStrategy` does not enforce the curvature condition that keeps the
    limited-memory inverse-Hessian approximation well-defined and positive definite. On a linear
    objective, the gradient never changes between iterates, so the very first correction pair has
    exactly zero curvature $(s, y) = 0$: the optimizer must reject it explicitly rather than
    letting `1.0 / (s, y)` raise `ZeroDivisionError`."""
    optimizer = CustomLBFGSOptimizer(
        CustomLBFGSSettings(),
        ArmijoBacktrackingLineSearch(ArmijoBacktrackingLineSearchSettings()),
        AlwaysAcceptStrategy(),
    )
    model = helpers.LinearModel(np.array([1.0, 1.0]))

    with pytest.raises(ValueError, match="non-positive curvature"):
        optimizer.run(np.zeros(2), model)
