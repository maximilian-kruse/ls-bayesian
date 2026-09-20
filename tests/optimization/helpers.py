"""Constants, setup objects and pure helper functions for the `optimization` tests.

This is a regular module, imported by test modules and `conftest.py` files alike. Fixtures live in
the `conftest.py` files, everything that is imported by name lives here.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, override

import nbformat
import numpy as np
from nbclient import NotebookClient

from ls_bayesian.optimization.algorithms.scipy_lbfgs_b import ScipyLBFGSBSettings
from ls_bayesian.optimization.model import OptimizationModel
from ls_bayesian.optimization.optimizer import (
    BaseOptimizer,
    IterationCallback,
    OptimizationHistory,
    OptimizationResult,
)
from ls_bayesian.posterior.posterior import LogPosterior


# ==================================================================================================
def quadratic_loss(
    parameter_vector: np.ndarray, matrix: np.ndarray, minimizer: np.ndarray
) -> float:
    """Convex quadratic bowl $\\frac{1}{2}(m - m^*)^T A (m - m^*)$ with known minimizer $m^*$."""
    difference_vector = parameter_vector - minimizer
    return float(0.5 * difference_vector @ matrix @ difference_vector)


def quadratic_gradient(
    parameter_vector: np.ndarray, matrix: np.ndarray, minimizer: np.ndarray
) -> np.ndarray:
    """Euclidean gradient of `quadratic_loss`."""
    return matrix @ (parameter_vector - minimizer)


def random_spd_matrix(rng: np.random.Generator, dim: int) -> np.ndarray:
    factor = rng.random((dim, dim))
    return factor @ factor.T + dim * np.eye(dim)


# ==================================================================================================
def rosenbrock_loss(parameter_vector: np.ndarray) -> float:
    """Standard n-dimensional Rosenbrock function, non-convex, minimizer at all-ones."""
    x = parameter_vector[:-1]
    y = parameter_vector[1:]
    return float(np.sum(100.0 * (y - x**2) ** 2 + (1.0 - x) ** 2))


def rosenbrock_gradient(parameter_vector: np.ndarray) -> np.ndarray:
    """Euclidean gradient of `rosenbrock_loss`."""
    gradient = np.zeros_like(parameter_vector)
    x = parameter_vector[:-1]
    y = parameter_vector[1:]
    gradient[:-1] += -400.0 * x * (y - x**2) - 2.0 * (1.0 - x)
    gradient[1:] += 200.0 * (y - x**2)
    return gradient


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
        return quadratic_loss(parameter_vector, self.matrix, self.minimizer)

    @override
    def evaluate_gradient(self, parameter_vector: np.ndarray) -> np.ndarray:
        euclidean_gradient = quadratic_gradient(parameter_vector, self.matrix, self.minimizer)
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
        return rosenbrock_loss(parameter_vector)

    @override
    def evaluate_gradient(self, parameter_vector: np.ndarray) -> np.ndarray:
        return rosenbrock_gradient(parameter_vector)

    @override
    def evaluate_hessian_vector_product(
        self, parameter_vector: np.ndarray, direction_vector: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    @override
    def evaluate_inner_product(self, first_vector: np.ndarray, second_vector: np.ndarray) -> float:
        return float(np.dot(first_vector, second_vector))


# ==================================================================================================
class LogPosteriorModel(OptimizationModel):
    """Adapts a `LogPosterior` to the `OptimizationModel` interface, with an optional
    inner-product weight matrix.

    With `inner_product_matrix=None`, the standard Euclidean inner product is used and
    `evaluate_gradient` returns `log_posterior.evaluate_gradient` unchanged. With a weight matrix
    $W$, `evaluate_gradient` returns the Riesz representer $W^{-1}\\nabla J(m)$ under
    $(u,v)_W = u^T W v$, emulating a Hessian/prior-preconditioned representer change -- the role a
    Cameron-Martin inner product plays for a real Bayesian prior -- without depending on
    `spde_prior`.
    """

    def __init__(
        self, log_posterior: LogPosterior, inner_product_matrix: np.ndarray | None = None
    ) -> None:
        self.log_posterior = log_posterior
        self.inner_product_matrix = inner_product_matrix

    @override
    def evaluate_cost(self, parameter_vector: np.ndarray) -> float:
        return self.log_posterior.evaluate_cost(parameter_vector)

    @override
    def evaluate_gradient(self, parameter_vector: np.ndarray) -> np.ndarray:
        euclidean_gradient = self.log_posterior.evaluate_gradient(parameter_vector)
        if self.inner_product_matrix is None:
            return euclidean_gradient
        return np.linalg.solve(self.inner_product_matrix, euclidean_gradient)

    @override
    def evaluate_hessian_vector_product(
        self, parameter_vector: np.ndarray, direction_vector: np.ndarray
    ) -> np.ndarray:
        return self.log_posterior.evaluate_hessian_vector_product(
            parameter_vector, direction_vector
        )

    @override
    def evaluate_inner_product(self, first_vector: np.ndarray, second_vector: np.ndarray) -> float:
        if self.inner_product_matrix is None:
            return float(np.dot(first_vector, second_vector))
        return float(first_vector @ self.inner_product_matrix @ second_vector)


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
REPO_ROOT = Path(__file__).resolve().parents[2]
OPTIMIZATION_TUTORIALS_DIR = REPO_ROOT / "tutorials" / "optimization"
SCIPY_LBFGS_NOTEBOOK = OPTIMIZATION_TUTORIALS_DIR / "scipy_lbfgs.ipynb"
CUSTOM_LBFGS_NOTEBOOK = OPTIMIZATION_TUTORIALS_DIR / "custom_lbfgs.ipynb"
NOTEBOOK_EXECUTION_TIMEOUT_SECONDS = 120


def execute_notebook_and_extract_values(
    notebook_path: Path, expressions: Mapping[str, str]
) -> dict[str, Any]:
    """Execute a tutorial notebook and evaluate expressions against its final namespace.

    The notebook is executed unmodified in its own kernel, except for one appended code cell that
    evaluates the given expressions and serializes the results to stdout as JSON. Expressions
    reduce large results (e.g. vectors) to compact scalars, such as a norm, so that reference
    values stay small numeric literals in test code rather than stored array data.

    Args:
        notebook_path (Path): Path to the `.ipynb` file to execute.
        expressions (Mapping[str, str]): Mapping from a result key to a Python expression,
            evaluated in the notebook's namespace after all of its own cells have run. Expression
            results must be JSON-serializable.

    Returns:
        dict[str, Any]: Mapping from result key to its JSON-deserialized value.
    """
    notebook = nbformat.read(notebook_path, as_version=4)
    probe_source = (
        "import json as _json\n"
        f"_probe_values = {{key: eval(expr) for key, expr in {dict(expressions)!r}.items()}}\n"
        "print(_json.dumps(_probe_values))"
    )
    notebook.cells.append(nbformat.v4.new_code_cell(source=probe_source))
    client = NotebookClient(
        notebook,
        timeout=NOTEBOOK_EXECUTION_TIMEOUT_SECONDS,
        kernel_name=notebook.metadata["kernelspec"]["name"],
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()

    probe_outputs = notebook.cells[-1]["outputs"]
    stdout_text = "".join(
        output["text"] for output in probe_outputs if output.get("name") == "stdout"
    )
    return json.loads(stdout_text)
