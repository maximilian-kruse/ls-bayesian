r"""Integration tests for `NewtonCGOptimizer.run()`: the full outer iteration with a real line
search and the truncated CG inner solve against concrete objectives."""

from typing import override

import numpy as np
import pytest

from ls_bayesian.optimization.algorithms.newton_cg import NewtonCGOptimizer, NewtonCGSettings
from ls_bayesian.optimization.components.line_search import (
    ArmijoBacktrackingLineSearch,
    ArmijoBacktrackingLineSearchSettings,
    StrongWolfeLineSearch,
    StrongWolfeLineSearchSettings,
)
from tests.optimization import helpers

pytestmark = pytest.mark.integration

CONVERGENCE_ABSOLUTE_TOLERANCE = 1e-3


# ==================================================================================================
class _RosenbrockWithHessianModel(helpers.RosenbrockModel):
    """Rosenbrock function with its exact (indefinite in places) Hessian-vector product."""

    @override
    def evaluate_hessian_vector_product(
        self, parameter_vector: np.ndarray, direction_vector: np.ndarray
    ) -> np.ndarray:
        x = parameter_vector[:-1]
        y = parameter_vector[1:]
        dim = parameter_vector.size
        hessian = np.zeros((dim, dim))
        for i in range(dim - 1):
            hessian[i, i] += 1200.0 * x[i] ** 2 - 400.0 * y[i] + 2.0
            hessian[i, i + 1] += -400.0 * x[i]
            hessian[i + 1, i] += -400.0 * x[i]
            hessian[i + 1, i + 1] += 200.0
        return hessian @ direction_vector


# --------------------------------------------------------------------------------------------------
def _armijo() -> ArmijoBacktrackingLineSearch:
    return ArmijoBacktrackingLineSearch(ArmijoBacktrackingLineSearchSettings())


# ==================================================================================================
def test_converges_on_rosenbrock_with_armijo_line_search() -> None:
    optimizer = NewtonCGOptimizer(
        NewtonCGSettings(maximum_num_iterations=500, gradient_norm_tolerance=1e-8), _armijo()
    )

    result = optimizer.run(np.array([-1.2, 1.0, -1.2, 1.0]), _RosenbrockWithHessianModel())

    assert result.success
    np.testing.assert_allclose(result.result, np.ones(4), atol=CONVERGENCE_ABSOLUTE_TOLERANCE)


# --------------------------------------------------------------------------------------------------
def test_converges_on_rosenbrock_with_strong_wolfe_line_search() -> None:
    optimizer = NewtonCGOptimizer(
        NewtonCGSettings(maximum_num_iterations=500, gradient_norm_tolerance=1e-8),
        StrongWolfeLineSearch(StrongWolfeLineSearchSettings()),
    )

    result = optimizer.run(np.array([-1.2, 1.0, -1.2, 1.0]), _RosenbrockWithHessianModel())

    assert result.success
    np.testing.assert_allclose(result.result, np.ones(4), atol=CONVERGENCE_ABSOLUTE_TOLERANCE)


# --------------------------------------------------------------------------------------------------
def test_ill_conditioned_quadratic_converges_in_few_iterations() -> None:
    """An eigenvalue spread of 1e8 makes steepest descent useless; CG resolves it within the
    dimension of the space."""
    rng = np.random.default_rng(0)
    dim = 20
    orthogonal, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    eigenvalues = np.logspace(0.0, 8.0, dim)
    matrix = orthogonal @ np.diag(eigenvalues) @ orthogonal.T
    minimizer = rng.random(dim)
    model = helpers.QuadraticModel(matrix, minimizer)
    optimizer = NewtonCGOptimizer(
        NewtonCGSettings(
            gradient_norm_tolerance=1e-4, maximum_forcing_term=1e-6, minimum_forcing_term=1e-10
        ),
        _armijo(),
    )

    result = optimizer.run(np.zeros(dim), model)

    assert result.success
    assert result.num_iterations <= 5
    np.testing.assert_allclose(result.result, minimizer, atol=1e-6)


# --------------------------------------------------------------------------------------------------
def test_history_has_one_entry_per_iteration() -> None:
    rng = np.random.default_rng(3)
    model = helpers.QuadraticModel(helpers.random_spd_matrix(rng, 5), rng.random(5))
    optimizer = NewtonCGOptimizer(NewtonCGSettings(gradient_norm_tolerance=1e-10), _armijo())

    result = optimizer.run(np.zeros(5), model)

    assert result.loss_history.size == result.num_iterations
    assert result.gradient_norm_history.size == result.num_iterations
    assert np.all(np.diff(result.loss_history) <= 0.0)
