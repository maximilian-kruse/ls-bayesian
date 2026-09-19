import numpy as np
import pytest

from ls_bayesian.optimization.algorithms.custom_lbfgs import (
    CorrectionPairStore,
    MetricLBFGSOptimizer,
    MetricLBFGSSettings,
    identity_seed_operator,
    two_loop_recursion,
)
from ls_bayesian.optimization.components.cautious_update import (
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


def _default_optimizer(settings: MetricLBFGSSettings | None = None) -> MetricLBFGSOptimizer:
    return MetricLBFGSOptimizer(
        settings or MetricLBFGSSettings(),
        ArmijoBacktrackingLineSearch(ArmijoBacktrackingLineSearchSettings()),
        CautiousUpdateStrategy(CautiousUpdateSettings()),
    )


# ==================================================================================================
def test_store_evicts_oldest_beyond_memory_size() -> None:
    store = CorrectionPairStore(memory_size=2)
    store.add(np.array([1.0]), np.array([1.0]), rho=1.0)
    store.add(np.array([2.0]), np.array([2.0]), rho=2.0)
    store.add(np.array([3.0]), np.array([3.0]), rho=3.0)

    assert len(store) == 2
    assert [pair.rho for pair in store] == [2.0, 3.0]


# --------------------------------------------------------------------------------------------------
def test_store_iterates_in_both_documented_orders() -> None:
    store = CorrectionPairStore(memory_size=3)
    store.add(np.array([1.0]), np.array([1.0]), rho=1.0)
    store.add(np.array([2.0]), np.array([2.0]), rho=2.0)

    assert [pair.rho for pair in store] == [1.0, 2.0]
    assert [pair.rho for pair in reversed(store)] == [2.0, 1.0]


# ==================================================================================================
def test_two_loop_recursion_with_no_pairs_is_steepest_descent() -> None:
    store = CorrectionPairStore(memory_size=5)
    gradient = np.array([1.0, -2.0, 3.0])
    model = helpers.ZeroModel()

    direction = two_loop_recursion(gradient, store, model, identity_seed_operator)

    np.testing.assert_allclose(direction, -gradient)


# --------------------------------------------------------------------------------------------------
def test_two_loop_recursion_matches_closed_form_bfgs_update_with_one_pair() -> None:
    """With one correction pair `(s, y)` and the identity seed, the two-loop recursion is
    mathematically equivalent to one step of the explicit BFGS inverse-Hessian update
    `H_1 = (I - rho s y^T) H_0 (I - rho y s^T) + rho s s^T` (`optimization.tex`, eq. 29/43)
    applied to the gradient; this test checks that equivalence directly against the closed-form
    matrix, in the Euclidean inner product for an independently-computable reference."""
    model = helpers.ZeroModel()
    s = np.array([1.0, 0.5])
    y = np.array([0.3, 0.8])
    rho = 1.0 / np.dot(s, y)
    store = CorrectionPairStore(memory_size=1)
    store.add(s, y, rho=rho)
    gradient = np.array([0.2, -0.4])

    direction = two_loop_recursion(gradient, store, model, identity_seed_operator)

    identity = np.eye(2)
    updated_inverse_hessian = (identity - rho * np.outer(s, y)) @ (
        identity - rho * np.outer(y, s)
    ) + rho * np.outer(s, s)
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
        MetricLBFGSSettings(maximum_num_iterations=200, gradient_norm_tolerance=1e-8)
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
    """Cross-check against `LBFGSOptimizer` on the same problem: with the default (Euclidean)
    model, this backend should also converge to the known minimizer."""
    optimizer = _default_optimizer(
        MetricLBFGSSettings(maximum_num_iterations=500, gradient_norm_tolerance=1e-6)
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
        MetricLBFGSSettings(maximum_num_iterations=1, gradient_norm_tolerance=1e-12)
    )
    initial_guess = np.array([-1.2, 1.0])
    model = helpers.RosenbrockModel()

    result = optimizer.run(initial_guess, model)

    assert not result.success
    assert result.num_iterations == 1


# --------------------------------------------------------------------------------------------------
def test_zero_iterations_when_initial_guess_already_converged() -> None:
    optimizer = _default_optimizer(MetricLBFGSSettings(gradient_norm_tolerance=1e-6))
    minimizer = np.zeros(2)
    model = helpers.QuadraticModel(np.eye(2), minimizer)

    result = optimizer.run(minimizer, model)

    assert result.success
    assert result.num_iterations == 0
