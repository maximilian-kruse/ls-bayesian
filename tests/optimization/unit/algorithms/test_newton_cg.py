import numpy as np
import pytest

from ls_bayesian.optimization.algorithms.newton_cg import NewtonCGOptimizer, NewtonCGSettings
from ls_bayesian.optimization.components.line_search import (
    ArmijoBacktrackingLineSearch,
    ArmijoBacktrackingLineSearchSettings,
)
from tests.optimization import helpers

pytestmark = pytest.mark.unit

DIMENSION = 8
CONVERGENCE_ABSOLUTE_TOLERANCE = 1e-6


def _optimizer(settings: NewtonCGSettings | None = None) -> NewtonCGOptimizer:
    return NewtonCGOptimizer(
        settings or NewtonCGSettings(gradient_norm_tolerance=1e-8),
        ArmijoBacktrackingLineSearch(ArmijoBacktrackingLineSearchSettings()),
    )


# ==================================================================================================
def test_requires_hessian() -> None:
    assert NewtonCGOptimizer.requires_hessian is True


# --------------------------------------------------------------------------------------------------
def test_settings_reject_inverted_forcing_bounds() -> None:
    with pytest.raises(ValueError, match="minimum_forcing_term"):
        NewtonCGSettings(minimum_forcing_term=0.6, maximum_forcing_term=0.5)


# --------------------------------------------------------------------------------------------------
def test_forcing_term_is_clipped() -> None:
    optimizer = _optimizer(NewtonCGSettings(maximum_forcing_term=0.5, minimum_forcing_term=1e-3))
    assert optimizer._forcing_term(100.0) == 0.5
    assert optimizer._forcing_term(0.04) == pytest.approx(0.2)
    assert optimizer._forcing_term(1e-12) == 1e-3


# --------------------------------------------------------------------------------------------------
def test_quadratic_converges_in_one_iteration_with_exact_cg() -> None:
    rng = np.random.default_rng(0)
    matrix = helpers.random_spd_matrix(rng, DIMENSION)
    minimizer = rng.random(DIMENSION)
    model = helpers.QuadraticModel(matrix, minimizer)
    settings = NewtonCGSettings(
        gradient_norm_tolerance=1e-6, maximum_forcing_term=1e-8, minimum_forcing_term=1e-10
    )

    result = _optimizer(settings).run(np.zeros(DIMENSION), model)

    assert result.success
    np.testing.assert_allclose(result.result, minimizer, atol=CONVERGENCE_ABSOLUTE_TOLERANCE)
    assert result.num_iterations == 1


# --------------------------------------------------------------------------------------------------
def test_quadratic_in_cameron_martin_like_geometry_needs_few_cg_iterations() -> None:
    rng = np.random.default_rng(1)
    matrix = helpers.random_spd_matrix(rng, DIMENSION)
    minimizer = rng.random(DIMENSION)
    # inner product = Hessian: the gradient representer is m - m*, the Hessian action the identity
    model = helpers.QuadraticModel(matrix, minimizer, inner_product_matrix=matrix)

    result = _optimizer().run(np.zeros(DIMENSION), model)

    assert result.success
    assert result.num_iterations == 1
    np.testing.assert_allclose(result.result, minimizer, atol=CONVERGENCE_ABSOLUTE_TOLERANCE)


# --------------------------------------------------------------------------------------------------
def test_line_search_failure_returns_last_iterate() -> None:
    class AscentModel(helpers.QuadraticModel):
        def evaluate_cost(self, parameter_vector):  # noqa: ANN001, ANN201
            # the quadratic model's descent direction increases this loss
            return float(-np.sum(parameter_vector**2))

    rng = np.random.default_rng(2)
    matrix = helpers.random_spd_matrix(rng, 3)
    model = AscentModel(matrix, rng.random(3))

    result = _optimizer().run(np.ones(3), model)

    assert not result.success
    assert "Line search failed" in result.status_message
