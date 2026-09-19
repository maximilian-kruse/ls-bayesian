import numpy as np
import pytest

from ls_bayesian.optimization.components.line_search import (
    ArmijoBacktrackingLineSearch,
    ArmijoBacktrackingLineSearchSettings,
)

pytestmark = pytest.mark.unit


def _quadratic_loss(point: np.ndarray) -> float:
    return float(0.5 * point @ point)


# ==================================================================================================
def test_accepts_initial_step_when_condition_holds_immediately() -> None:
    settings = ArmijoBacktrackingLineSearchSettings()
    line_search = ArmijoBacktrackingLineSearch(settings)
    current_point = np.array([1.0, 1.0])
    gradient = current_point
    search_direction = -gradient
    directional_derivative = float(gradient @ search_direction)

    result = line_search.find_step_size(
        current_point,
        search_direction,
        _quadratic_loss(current_point),
        directional_derivative,
        _quadratic_loss,
    )

    assert result.step_size == settings.initial_step_size
    np.testing.assert_allclose(
        result.loss, _quadratic_loss(current_point + result.step_size * search_direction)
    )


# --------------------------------------------------------------------------------------------------
def test_backtracks_when_initial_step_is_too_large() -> None:
    settings = ArmijoBacktrackingLineSearchSettings(initial_step_size=100.0)
    line_search = ArmijoBacktrackingLineSearch(settings)
    current_point = np.array([1.0, 1.0])
    gradient = current_point
    search_direction = -gradient
    directional_derivative = float(gradient @ search_direction)

    result = line_search.find_step_size(
        current_point,
        search_direction,
        _quadratic_loss(current_point),
        directional_derivative,
        _quadratic_loss,
    )

    assert 0.0 < result.step_size < settings.initial_step_size
    np.testing.assert_allclose(
        result.loss, _quadratic_loss(current_point + result.step_size * search_direction)
    )


# --------------------------------------------------------------------------------------------------
def test_raises_when_direction_is_not_descent() -> None:
    settings = ArmijoBacktrackingLineSearchSettings(max_backtracking_steps=5)
    line_search = ArmijoBacktrackingLineSearch(settings)
    current_point = np.array([1.0, 1.0])
    gradient = current_point
    search_direction = gradient  # ascent direction, never satisfies Armijo for a quadratic bowl
    directional_derivative = float(gradient @ search_direction)

    with pytest.raises(RuntimeError, match="descent"):
        line_search.find_step_size(
            current_point,
            search_direction,
            _quadratic_loss(current_point),
            directional_derivative,
            _quadratic_loss,
        )
