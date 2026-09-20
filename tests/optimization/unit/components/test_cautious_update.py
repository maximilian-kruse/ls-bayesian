import numpy as np
import pytest

from ls_bayesian.optimization.components.cautious_update import (
    AlwaysAcceptStrategy,
    CautiousUpdateSettings,
    CautiousUpdateStrategy,
)
from tests.optimization import helpers

pytestmark = pytest.mark.unit


# ==================================================================================================
def test_epsilon_zero_reduces_to_standard_curvature_condition() -> None:
    model = helpers.ZeroModel()
    strategy = CautiousUpdateStrategy(CautiousUpdateSettings(epsilon=0.0, alpha=1.0))
    gradient = np.array([1.0, 1.0])

    s_positive_curvature = np.array([1.0, 0.0])
    y_positive_curvature = np.array([1.0, 0.0])
    assert strategy.accept_update(s_positive_curvature, y_positive_curvature, gradient, model)

    s_negative_curvature = np.array([1.0, 0.0])
    y_negative_curvature = np.array([-1.0, 0.0])
    assert not strategy.accept_update(s_negative_curvature, y_negative_curvature, gradient, model)


# --------------------------------------------------------------------------------------------------
def test_epsilon_zero_rejects_exactly_zero_curvature() -> None:
    """The condition is documented (see `CautiousUpdateSettings.epsilon`) to reduce to the
    strict standard curvature condition $(y, s) > 0$ at `epsilon = 0`, so a pair with exactly zero
    curvature must be rejected, not accepted on equality."""
    model = helpers.ZeroModel()
    strategy = CautiousUpdateStrategy(CautiousUpdateSettings(epsilon=0.0, alpha=1.0))
    gradient = np.array([1.0, 1.0])
    s_zero_curvature = np.array([1.0, 0.0])
    y_zero_curvature = np.array([0.0, 1.0])

    assert not strategy.accept_update(s_zero_curvature, y_zero_curvature, gradient, model)


# --------------------------------------------------------------------------------------------------
def test_rejects_pair_violating_scaled_condition_for_positive_epsilon() -> None:
    model = helpers.ZeroModel()
    strategy = CautiousUpdateStrategy(CautiousUpdateSettings(epsilon=10.0, alpha=1.0))
    gradient = np.array([1.0, 0.0])
    s = np.array([1.0, 0.0])
    y = np.array([1.0, 0.0])

    assert not strategy.accept_update(s, y, gradient, model)


# --------------------------------------------------------------------------------------------------
def test_always_accept_strategy_accepts_any_pair() -> None:
    model = helpers.ZeroModel()
    strategy = AlwaysAcceptStrategy()
    gradient = np.array([1.0, 0.0])
    s = np.array([1.0, 0.0])
    y = np.array([-1.0, 0.0])  # negative curvature, rejected by CautiousUpdateStrategy

    assert strategy.accept_update(s, y, gradient, model)
