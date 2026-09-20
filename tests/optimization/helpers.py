"""Constants, setup objects and pure helper functions for the `optimization` tests.

This is a regular module, imported by test modules and `conftest.py` files alike. Fixtures live in
the `conftest.py` files, everything that is imported by name lives here.
"""

from typing import override

import numpy as np

from ls_bayesian.optimization.algorithms.scipy_lbfgs_b import ScipyLBFGSBSettings
from ls_bayesian.optimization.model import OptimizationModel
from ls_bayesian.optimization.optimizer import (
    BaseOptimizer,
    IterationCallback,
    OptimizationHistory,
    OptimizationResult,
)
from tests import notebook_helpers


# ==================================================================================================
def random_spd_matrix(rng: np.random.Generator, dim: int) -> np.ndarray:
    factor = rng.random((dim, dim))
    return factor @ factor.T + dim * np.eye(dim)


# ==================================================================================================
def default_scipy_lbfgs_b_settings() -> ScipyLBFGSBSettings:
    return ScipyLBFGSBSettings()


# ==================================================================================================
class ZeroModel(OptimizationModel):
    """Trivial model: constant zero loss and gradient, no Hessian-vector product support,
    standard Euclidean inner product. Exercises `BaseOptimizer.run()`'s validation and
    orchestration in isolation from any real objective."""

    @override
    def evaluate_cost(self, parameter_vector: np.ndarray) -> float:
        return 0.0

    @override
    def evaluate_gradient(self, parameter_vector: np.ndarray) -> np.ndarray:
        return np.zeros_like(parameter_vector)

    @override
    def evaluate_hessian_vector_product(
        self, parameter_vector: np.ndarray, direction_vector: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    @override
    def evaluate_inner_product(self, first_vector: np.ndarray, second_vector: np.ndarray) -> float:
        return float(np.dot(first_vector, second_vector))


# ==================================================================================================
class ConstantGradientModel(OptimizationModel):
    """Model with zero loss, a constant gradient and an identity Hessian-vector product, for
    exercising `BaseOptimizer.run()`'s Hessian-vector-product wiring in isolation from any real
    objective."""

    def __init__(self, gradient_value: float) -> None:
        self.gradient_value = gradient_value

    @override
    def evaluate_cost(self, parameter_vector: np.ndarray) -> float:
        return 0.0

    @override
    def evaluate_gradient(self, parameter_vector: np.ndarray) -> np.ndarray:
        return np.full_like(parameter_vector, self.gradient_value)

    @override
    def evaluate_hessian_vector_product(
        self, parameter_vector: np.ndarray, direction_vector: np.ndarray
    ) -> np.ndarray:
        return direction_vector

    @override
    def evaluate_inner_product(self, first_vector: np.ndarray, second_vector: np.ndarray) -> float:
        return float(np.dot(first_vector, second_vector))


# ==================================================================================================
class LinearModel(OptimizationModel):
    r"""Linear objective $I(m) = c^T m$, standard Euclidean inner product.

    The gradient $c$ is constant, so the gradient difference between any two iterates is exactly
    zero: used to engineer a correction pair with exactly zero curvature $(s, y) = 0$, to exercise
    `CustomLBFGSOptimizer`'s handling of a non-positive-curvature pair accepted by a permissive
    acceptance strategy.
    """

    def __init__(self, gradient_value: np.ndarray) -> None:
        self.gradient_value = gradient_value

    @override
    def evaluate_cost(self, parameter_vector: np.ndarray) -> float:
        return float(np.dot(self.gradient_value, parameter_vector))

    @override
    def evaluate_gradient(self, parameter_vector: np.ndarray) -> np.ndarray:
        return self.gradient_value

    @override
    def evaluate_hessian_vector_product(
        self, parameter_vector: np.ndarray, direction_vector: np.ndarray
    ) -> np.ndarray:
        return np.zeros_like(direction_vector)

    @override
    def evaluate_inner_product(self, first_vector: np.ndarray, second_vector: np.ndarray) -> float:
        return float(np.dot(first_vector, second_vector))


# ==================================================================================================
class QuadraticModel(OptimizationModel):
    r"""Quadratic bowl $\frac{1}{2}(m-m^*)^T A (m-m^*)$ with known minimizer $m^*$ and constant
    Hessian $A$.

    `evaluate_gradient`/`evaluate_hessian_vector_product` return the Riesz representer under the
    (possibly weighted) inner product `inner_product_matrix` (defaults to the identity, i.e. the
    standard Euclidean inner product): with weight $W$, the representer of the Euclidean gradient
    $A(m-m^*)$ is $W^{-1}A(m-m^*)$. In particular, `inner_product_matrix=A` yields the trivial
    representer $m - m^*$, used to emulate a Cameron-Martin-like geometry in tests.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        minimizer: np.ndarray,
        inner_product_matrix: np.ndarray | None = None,
    ) -> None:
        self.matrix = matrix
        self.minimizer = minimizer
        self.inner_product_matrix = (
            np.eye(matrix.shape[0]) if inner_product_matrix is None else inner_product_matrix
        )

    @override
    def evaluate_cost(self, parameter_vector: np.ndarray) -> float:
        difference_vector = parameter_vector - self.minimizer
        return float(0.5 * difference_vector @ self.matrix @ difference_vector)

    @override
    def evaluate_gradient(self, parameter_vector: np.ndarray) -> np.ndarray:
        euclidean_gradient = self.matrix @ (parameter_vector - self.minimizer)
        return np.linalg.solve(self.inner_product_matrix, euclidean_gradient)

    @override
    def evaluate_hessian_vector_product(
        self, parameter_vector: np.ndarray, direction_vector: np.ndarray
    ) -> np.ndarray:
        return np.linalg.solve(self.inner_product_matrix, self.matrix @ direction_vector)

    @override
    def evaluate_inner_product(self, first_vector: np.ndarray, second_vector: np.ndarray) -> float:
        return float(first_vector @ self.inner_product_matrix @ second_vector)


# ==================================================================================================
class RosenbrockModel(OptimizationModel):
    """Standard n-dimensional Rosenbrock function, non-convex, minimizer at all-ones, standard
    Euclidean inner product. No Hessian-vector product support."""

    @override
    def evaluate_cost(self, parameter_vector: np.ndarray) -> float:
        x = parameter_vector[:-1]
        y = parameter_vector[1:]
        return float(np.sum(100.0 * (y - x**2) ** 2 + (1.0 - x) ** 2))

    @override
    def evaluate_gradient(self, parameter_vector: np.ndarray) -> np.ndarray:
        gradient = np.zeros_like(parameter_vector)
        x = parameter_vector[:-1]
        y = parameter_vector[1:]
        gradient[:-1] += -400.0 * x * (y - x**2) - 2.0 * (1.0 - x)
        gradient[1:] += 200.0 * (y - x**2)
        return gradient

    @override
    def evaluate_hessian_vector_product(
        self, parameter_vector: np.ndarray, direction_vector: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    @override
    def evaluate_inner_product(self, first_vector: np.ndarray, second_vector: np.ndarray) -> float:
        return float(np.dot(first_vector, second_vector))


# ==================================================================================================
class WeightedInnerProductModel(OptimizationModel):
    """Model exposing only a weighted inner product $(u, v)_W = u^T W v$, for isolating
    `OptimizationModel.evaluate_norm`'s wiring to `evaluate_inner_product` from any concrete
    objective or Hessian-vector product."""

    def __init__(self, weight_matrix: np.ndarray) -> None:
        self.weight_matrix = weight_matrix

    @override
    def evaluate_cost(self, parameter_vector: np.ndarray) -> float:
        raise NotImplementedError

    @override
    def evaluate_gradient(self, parameter_vector: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @override
    def evaluate_hessian_vector_product(
        self, parameter_vector: np.ndarray, direction_vector: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    @override
    def evaluate_inner_product(self, first_vector: np.ndarray, second_vector: np.ndarray) -> float:
        return float(first_vector @ self.weight_matrix @ second_vector)


# ==================================================================================================
class ConstantInnerProductModel(OptimizationModel):
    """Model whose `evaluate_inner_product` returns a fixed value regardless of its arguments, for
    exercising `OptimizationModel.evaluate_norm`'s round-off guard at precise, hand-picked squared
    norms rather than ones incidentally produced by a real inner product."""

    def __init__(self, inner_product_value: float) -> None:
        self.inner_product_value = inner_product_value

    @override
    def evaluate_cost(self, parameter_vector: np.ndarray) -> float:
        raise NotImplementedError

    @override
    def evaluate_gradient(self, parameter_vector: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @override
    def evaluate_hessian_vector_product(
        self, parameter_vector: np.ndarray, direction_vector: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    @override
    def evaluate_inner_product(self, first_vector: np.ndarray, second_vector: np.ndarray) -> float:
        return self.inner_product_value


# ==================================================================================================
class FakeOptimizer(BaseOptimizer):
    """Deterministic double for testing `BaseOptimizer.run()`'s orchestration in isolation from
    any real backend: performs a fixed number of gradient-descent-like steps, calling `callback`
    once per step."""

    requires_hessian: bool = False

    def __init__(self, num_iterations: int = 2, step_size: float = 0.1, logger=None) -> None:
        super().__init__(logger=logger)
        self.num_iterations = num_iterations
        self.step_size = step_size

    @override
    def _run_impl(
        self,
        initial_guess: np.ndarray,
        model: OptimizationModel,
        callback: IterationCallback,
    ) -> np.ndarray:
        point = initial_guess.copy()
        for _ in range(self.num_iterations):
            loss = model.evaluate_cost(point)
            gradient = model.evaluate_gradient(point)
            gradient_norm = model.evaluate_norm(gradient)
            point = point - self.step_size * gradient
            callback(loss, gradient_norm)
        return point

    @override
    def _create_optimization_result(
        self, raw_result: np.ndarray, history: OptimizationHistory
    ) -> OptimizationResult:
        return OptimizationResult(
            result=raw_result,
            loss_history=history.loss_history,
            gradient_norm_history=history.gradient_norm_history,
            num_iterations=self.num_iterations,
            success=True,
            status_message="fake optimizer finished",
        )


# ==================================================================================================
class FakeHessianOptimizer(FakeOptimizer):
    """Like `FakeOptimizer`, but `requires_hessian=True` and actually calls
    `evaluate_hessian_vector_product` once, so a model that does not support it raises
    `NotImplementedError` from within a real optimizer run."""

    requires_hessian: bool = True

    @override
    def _run_impl(
        self,
        initial_guess: np.ndarray,
        model: OptimizationModel,
        callback: IterationCallback,
    ) -> np.ndarray:
        model.evaluate_hessian_vector_product(initial_guess, initial_guess)
        return super()._run_impl(initial_guess, model, callback)


# ==================================================================================================
OPTIMIZATION_TUTORIALS_DIR = notebook_helpers.REPO_ROOT / "tutorials" / "optimization"
SCIPY_LBFGS_NOTEBOOK = OPTIMIZATION_TUTORIALS_DIR / "scipy_lbfgsb.ipynb"
CUSTOM_LBFGS_NOTEBOOK = OPTIMIZATION_TUTORIALS_DIR / "custom_lbfgs.ipynb"
NOTEBOOK_EXECUTION_TIMEOUT_SECONDS = 120
