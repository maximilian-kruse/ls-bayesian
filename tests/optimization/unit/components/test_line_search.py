import numpy as np
import pytest

from ls_bayesian.optimization.components.line_search import (
    ArmijoBacktrackingLineSearch,
    ArmijoBacktrackingLineSearchSettings,
    LineSearchProblem,
    StrongWolfeLineSearch,
    StrongWolfeLineSearchSettings,
)
from ls_bayesian.optimization.components.seed_scaling import BarzilaiBorweinSeedScalingSettings
from tests.optimization import helpers

pytestmark = pytest.mark.unit


def _quadratic_loss(point: np.ndarray) -> float:
    return float(0.5 * point @ point)


def _quadratic_gradient(point: np.ndarray) -> np.ndarray:
    return point


def _euclidean_inner_product(first: np.ndarray, second: np.ndarray) -> float:
    return float(first @ second)


# ==================================================================================================
# ArmijoBacktrackingLineSearch
# ==================================================================================================
def test_accepts_initial_step_when_condition_holds_immediately() -> None:
    settings = ArmijoBacktrackingLineSearchSettings()
    line_search = ArmijoBacktrackingLineSearch(settings)
    current_point = np.array([1.0, 1.0])
    gradient = current_point
    search_direction = -gradient

    result = line_search.find_step_size(
        LineSearchProblem(
            current_point=current_point,
            search_direction=search_direction,
            current_loss=_quadratic_loss(current_point),
            current_gradient=gradient,
            loss_function=_quadratic_loss,
            gradient_function=_quadratic_gradient,
            inner_product=_euclidean_inner_product,
        )
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

    result = line_search.find_step_size(
        LineSearchProblem(
            current_point=current_point,
            search_direction=search_direction,
            current_loss=_quadratic_loss(current_point),
            current_gradient=gradient,
            loss_function=_quadratic_loss,
            gradient_function=_quadratic_gradient,
            inner_product=_euclidean_inner_product,
        )
    )

    assert 0.0 < result.step_size < settings.initial_step_size
    np.testing.assert_allclose(
        result.loss, _quadratic_loss(current_point + result.step_size * search_direction)
    )


# --------------------------------------------------------------------------------------------------
def test_armijo_fails_when_direction_is_not_descent() -> None:
    settings = ArmijoBacktrackingLineSearchSettings(max_backtracking_steps=5)
    line_search = ArmijoBacktrackingLineSearch(settings)
    current_point = np.array([1.0, 1.0])
    gradient = current_point
    search_direction = gradient  # ascent direction, never satisfies Armijo for a quadratic bowl

    result = line_search.find_step_size(
        LineSearchProblem(
            current_point=current_point,
            search_direction=search_direction,
            current_loss=_quadratic_loss(current_point),
            current_gradient=gradient,
            loss_function=_quadratic_loss,
            gradient_function=_quadratic_gradient,
            inner_product=_euclidean_inner_product,
        )
    )

    assert not result.success
    assert np.isnan(result.step_size)


# ==================================================================================================
# StrongWolfeLineSearch
# ==================================================================================================
def test_accepts_initial_step_when_both_conditions_hold_immediately() -> None:
    """A full step along steepest descent on an isotropic quadratic bowl lands exactly on the
    minimizer, where both the sufficient-decrease and curvature conditions hold trivially."""
    settings = StrongWolfeLineSearchSettings()
    line_search = StrongWolfeLineSearch(settings)
    model = helpers.QuadraticModel(np.eye(2), minimizer=np.zeros(2))
    current_point = np.array([1.0, 1.0])
    gradient = model.evaluate_gradient(current_point)
    search_direction = -gradient

    result = line_search.find_step_size(
        LineSearchProblem(
            current_point=current_point,
            search_direction=search_direction,
            current_loss=model.evaluate_cost(current_point),
            current_gradient=gradient,
            loss_function=model.evaluate_cost,
            gradient_function=model.evaluate_gradient,
            inner_product=model.evaluate_inner_product,
        )
    )

    assert result.step_size == settings.initial_step_size
    np.testing.assert_allclose(result.loss, 0.0, atol=1e-12)


# --------------------------------------------------------------------------------------------------
def test_zoom_phase_finds_a_step_satisfying_both_strong_wolfe_conditions() -> None:
    """An ill-conditioned quadratic forces the bracketing phase's initial full step to fail
    sufficient decrease, entering the zoom phase; the returned step must still genuinely satisfy
    both strong-Wolfe conditions, checked directly against the closed-form model."""
    settings = StrongWolfeLineSearchSettings()
    line_search = StrongWolfeLineSearch(settings)
    model = helpers.QuadraticModel(np.diag([1.0, 50.0, 3.0]), minimizer=np.zeros(3))
    current_point = np.array([10.0, 1.0, 5.0])
    gradient = model.evaluate_gradient(current_point)
    search_direction = -gradient
    directional_derivative = model.evaluate_inner_product(gradient, search_direction)

    result = line_search.find_step_size(
        LineSearchProblem(
            current_point=current_point,
            search_direction=search_direction,
            current_loss=model.evaluate_cost(current_point),
            current_gradient=gradient,
            loss_function=model.evaluate_cost,
            gradient_function=model.evaluate_gradient,
            inner_product=model.evaluate_inner_product,
        )
    )

    assert result.step_size != settings.initial_step_size, "test setup must actually force zoom"
    sufficient_decrease_holds = result.loss <= model.evaluate_cost(current_point) + (
        settings.sufficient_decrease_constant * result.step_size * directional_derivative
    )
    new_point = current_point + result.step_size * search_direction
    new_directional_derivative = model.evaluate_inner_product(
        model.evaluate_gradient(new_point), search_direction
    )
    curvature_condition_holds = abs(new_directional_derivative) <= (
        settings.curvature_constant * abs(directional_derivative)
    )
    assert sufficient_decrease_holds
    assert curvature_condition_holds


# --------------------------------------------------------------------------------------------------
def test_satisfies_strong_wolfe_conditions_under_weighted_inner_product() -> None:
    """Both conditions are checked in `model`'s own (non-Euclidean) inner product, not silently in
    the Euclidean one, confirming the search is generalized correctly."""
    settings = StrongWolfeLineSearchSettings()
    line_search = StrongWolfeLineSearch(settings)
    weight_matrix = np.array([[3.0, 0.5], [0.5, 2.0]])
    model = helpers.QuadraticModel(
        np.diag([2.0, 10.0]), minimizer=np.array([1.0, -1.0]), inner_product_matrix=weight_matrix
    )
    current_point = np.array([5.0, 5.0])
    gradient = model.evaluate_gradient(current_point)
    search_direction = -gradient
    directional_derivative = model.evaluate_inner_product(gradient, search_direction)

    result = line_search.find_step_size(
        LineSearchProblem(
            current_point=current_point,
            search_direction=search_direction,
            current_loss=model.evaluate_cost(current_point),
            current_gradient=gradient,
            loss_function=model.evaluate_cost,
            gradient_function=model.evaluate_gradient,
            inner_product=model.evaluate_inner_product,
        )
    )

    sufficient_decrease_holds = result.loss <= model.evaluate_cost(current_point) + (
        settings.sufficient_decrease_constant * result.step_size * directional_derivative
    )
    new_point = current_point + result.step_size * search_direction
    new_directional_derivative = model.evaluate_inner_product(
        model.evaluate_gradient(new_point), search_direction
    )
    curvature_condition_holds = abs(new_directional_derivative) <= (
        settings.curvature_constant * abs(directional_derivative)
    )
    assert sufficient_decrease_holds
    assert curvature_condition_holds


# --------------------------------------------------------------------------------------------------
def test_strong_wolfe_fails_when_direction_is_not_descent() -> None:
    settings = StrongWolfeLineSearchSettings()
    line_search = StrongWolfeLineSearch(settings)
    model = helpers.QuadraticModel(np.eye(2), minimizer=np.zeros(2))
    current_point = np.array([1.0, 1.0])
    gradient = model.evaluate_gradient(current_point)
    search_direction = gradient  # ascent direction

    result = line_search.find_step_size(
        LineSearchProblem(
            current_point=current_point,
            search_direction=search_direction,
            current_loss=model.evaluate_cost(current_point),
            current_gradient=gradient,
            loss_function=model.evaluate_cost,
            gradient_function=model.evaluate_gradient,
            inner_product=model.evaluate_inner_product,
        )
    )

    assert not result.success
    assert np.isnan(result.step_size)


# --------------------------------------------------------------------------------------------------
def test_fails_when_bracketing_budget_exhausted() -> None:
    """An unbounded linear objective: the gradient is constant everywhere, so the directional
    derivative never satisfies the curvature condition (`|phi'| <= c2|phi'(0)|` would require
    `c2 >= 1`) and never changes sign either, so bracketing only ever extrapolates -- exhausting
    its iteration budget without ever finding or needing a bracket."""
    settings = StrongWolfeLineSearchSettings(max_bracketing_iterations=3)
    line_search = StrongWolfeLineSearch(settings)
    gradient_value = np.array([1.0, 1.0])

    def loss_function(point: np.ndarray) -> float:
        return float(gradient_value @ point)

    current_point = np.zeros(2)
    search_direction = -gradient_value

    def gradient_function(point: np.ndarray) -> np.ndarray:
        return gradient_value

    result = line_search.find_step_size(
        LineSearchProblem(
            current_point=current_point,
            search_direction=search_direction,
            current_loss=loss_function(current_point),
            current_gradient=gradient_value,
            loss_function=loss_function,
            gradient_function=gradient_function,
            inner_product=_euclidean_inner_product,
        )
    )

    assert not result.success
    assert np.isnan(result.step_size)


# --------------------------------------------------------------------------------------------------
def test_fails_when_zoom_budget_exhausted() -> None:
    """The same ill-conditioned quadratic as
    `test_zoom_phase_finds_a_step_satisfying_both_strong_wolfe_conditions` (which needs several
    zoom iterations to converge), but with `max_zoom_iterations` too small to reach one."""
    settings = StrongWolfeLineSearchSettings(max_zoom_iterations=1)
    line_search = StrongWolfeLineSearch(settings)
    model = helpers.QuadraticModel(np.diag([1.0, 50.0, 3.0]), minimizer=np.zeros(3))
    current_point = np.array([10.0, 1.0, 5.0])
    gradient = model.evaluate_gradient(current_point)
    search_direction = -gradient

    result = line_search.find_step_size(
        LineSearchProblem(
            current_point=current_point,
            search_direction=search_direction,
            current_loss=model.evaluate_cost(current_point),
            current_gradient=gradient,
            loss_function=model.evaluate_cost,
            gradient_function=model.evaluate_gradient,
            inner_product=model.evaluate_inner_product,
        )
    )

    assert not result.success
    assert np.isnan(result.step_size)


# --------------------------------------------------------------------------------------------------
def test_strong_wolfe_step_avoids_the_degenerate_gamma_armijo_accepts() -> None:
    r"""Regression test for the incident that motivated this class: on
    $\phi(\alpha) = c\alpha + \frac{\delta}{2}\alpha^2 + \frac{\mu}{4}\alpha^4$ with a tiny base
    curvature $\delta$ and a quartic term $\mu$ that only grows significant further out,
    `ArmijoBacktrackingLineSearch` accepts its very first trial step (full step, well within the
    near-flat region), producing a correction pair whose implied Barzilai-Borwein-style
    $\gamma=(s,y)/(y,y)$ is wildly outside `CustomLBFGSSettings`' default `[gamma_min, gamma_max]`
    clamp range. `StrongWolfeLineSearch`'s curvature condition forces it to keep searching past
    that point (into the higher-curvature region), landing on a step whose implied $\gamma$ is
    unremarkable."""
    base_curvature = 1e-6
    quartic_coefficient = 1.0
    linear_coefficient = np.array([-0.02])

    def gradient_function(point: np.ndarray) -> np.ndarray:
        return linear_coefficient + base_curvature * point + quartic_coefficient * point**3

    def loss_function(point: np.ndarray) -> float:
        return float(
            linear_coefficient @ point
            + 0.5 * base_curvature * (point @ point)
            + 0.25 * quartic_coefficient * np.sum(point**4)
        )

    current_point = np.zeros(1)
    gradient = gradient_function(current_point)
    search_direction = -gradient

    def implied_gamma(step_size: float) -> float:
        new_point = current_point + step_size * search_direction
        state_difference = new_point - current_point
        gradient_difference = gradient_function(new_point) - gradient
        return float(state_difference @ gradient_difference) / float(
            gradient_difference @ gradient_difference
        )

    armijo = ArmijoBacktrackingLineSearch(ArmijoBacktrackingLineSearchSettings())
    armijo_result = armijo.find_step_size(
        LineSearchProblem(
            current_point=current_point,
            search_direction=search_direction,
            current_loss=loss_function(current_point),
            current_gradient=gradient,
            loss_function=loss_function,
            gradient_function=gradient_function,
            inner_product=_euclidean_inner_product,
        )
    )
    armijo_gamma = implied_gamma(armijo_result.step_size)

    wolfe = StrongWolfeLineSearch(StrongWolfeLineSearchSettings())
    wolfe_result = wolfe.find_step_size(
        LineSearchProblem(
            current_point=current_point,
            search_direction=search_direction,
            current_loss=loss_function(current_point),
            current_gradient=gradient,
            loss_function=loss_function,
            gradient_function=gradient_function,
            inner_product=_euclidean_inner_product,
        )
    )
    wolfe_gamma = implied_gamma(wolfe_result.step_size)

    default_gamma_max = BarzilaiBorweinSeedScalingSettings().gamma_max
    assert armijo_gamma > default_gamma_max, "test setup must reproduce the original blow-up"
    assert wolfe_gamma < default_gamma_max
