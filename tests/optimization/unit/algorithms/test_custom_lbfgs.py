import numpy as np
import pytest

from ls_bayesian.optimization.algorithms.custom_lbfgs import (
    CorrectionPairStore,
    CustomLBFGSOptimizer,
    CustomLBFGSSettings,
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
