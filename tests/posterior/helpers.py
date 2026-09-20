r"""Constants, setup objects and pure helper functions for the `posterior` tests.

This is a regular module, imported by test modules and `conftest.py` files alike. Fixtures live in
the `conftest.py` files, everything that is imported by name lives here.
"""

from dataclasses import dataclass
from typing import override

import numpy as np
import scipy.sparse as sp

from ls_bayesian.posterior import interfaces, likelihood
from tests import notebook_helpers

# ==================================================================================================
PARAMETER_DIM = 4
NUM_VERTICES = 5
OBSERVED_VERTEX_INDICES = np.array([0, 1, 2, 4], dtype=np.int64)
FINITE_DIFFERENCE_STEP = 1e-6


def random_spd_matrix(rng: np.random.Generator, dim: int) -> np.ndarray:
    factor = rng.random((dim, dim))
    return factor @ factor.T + dim * np.eye(dim)


def central_difference_gradient(function, point: np.ndarray, step: float) -> np.ndarray:
    gradient = np.zeros_like(point)
    for i in range(point.shape[0]):
        perturbation = np.zeros_like(point)
        perturbation[i] = step
        gradient[i] = (function(point + perturbation) - function(point - perturbation)) / (2 * step)
    return gradient


# ==================================================================================================
class LinearParameterToSolutionMap(interfaces.ParameterToSolutionMap):
    """Linear forward map $F(m) = A m$, counting its evaluations."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix
        self.num_forward_evaluations = 0
        self.num_gradient_evaluations = 0

    @override
    def evaluate_forward(self, parameter_vector: np.ndarray) -> np.ndarray:
        self.num_forward_evaluations += 1
        return self.matrix @ parameter_vector

    @override
    def evaluate_gradient(
        self, solution_vector: np.ndarray, parameter_vector: np.ndarray, adjoint_vector: np.ndarray
    ) -> np.ndarray:
        self.num_gradient_evaluations += 1
        return self.matrix.T @ adjoint_vector

    @override
    def evaluate_jacobian_vector_product(
        self,
        solution_vector: np.ndarray,
        parameter_vector: np.ndarray,
        direction_vector: np.ndarray,
    ) -> np.ndarray:
        return self.matrix @ direction_vector

    @override
    def evaluate_hessian_vector_product(
        self,
        solution_vector: np.ndarray,
        parameter_vector: np.ndarray,
        direction_vector: np.ndarray,
        adjoint_vector: np.ndarray,
        gradient_vector: np.ndarray,
    ) -> np.ndarray:
        return np.zeros_like(parameter_vector)


# ==================================================================================================
class NonlinearParameterToSolutionMap(interfaces.ParameterToSolutionMap):
    r"""Forward map $F(m) = A (m \odot m)$, with Jacobian $\nabla_m F(m) = A\,\mathrm{diag}(2m)$.

    Unlike [`LinearParameterToSolutionMap`][], `evaluate_gradient` genuinely depends on the value
    of `parameter_vector`, so a test using this map can catch an argument-order bug in a caller
    that swaps `solution_vector` and `parameter_vector` when invoking `evaluate_gradient`.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    @override
    def evaluate_forward(self, parameter_vector: np.ndarray) -> np.ndarray:
        return self.matrix @ (parameter_vector * parameter_vector)

    @override
    def evaluate_gradient(
        self, solution_vector: np.ndarray, parameter_vector: np.ndarray, adjoint_vector: np.ndarray
    ) -> np.ndarray:
        return (2.0 * parameter_vector) * (self.matrix.T @ adjoint_vector)

    @override
    def evaluate_jacobian_vector_product(
        self,
        solution_vector: np.ndarray,
        parameter_vector: np.ndarray,
        direction_vector: np.ndarray,
    ) -> np.ndarray:
        return self.matrix @ (2.0 * parameter_vector * direction_vector)

    @override
    def evaluate_hessian_vector_product(
        self,
        solution_vector: np.ndarray,
        parameter_vector: np.ndarray,
        direction_vector: np.ndarray,
        adjoint_vector: np.ndarray,
        gradient_vector: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


# ==================================================================================================
class QuadraticPrior(interfaces.GaussianPrior):
    r"""Dense Gaussian prior with precision matrix $P$ and mean $\bar{m}$."""

    def __init__(self, mean_vector: np.ndarray, precision_matrix: np.ndarray, seed: int) -> None:
        self.mean_vector = mean_vector
        self.precision_matrix = precision_matrix
        self.covariance_matrix = np.linalg.inv(precision_matrix)
        self.covariance_factor = np.linalg.cholesky(self.covariance_matrix)
        self._rng = np.random.default_rng(seed)

    @property
    @override
    def random_vector_size(self) -> int:
        return self.covariance_factor.shape[1]

    @override
    def evaluate_cost(self, parameter_vector: np.ndarray) -> float:
        difference_vector = parameter_vector - self.mean_vector
        return float(0.5 * difference_vector @ self.precision_matrix @ difference_vector)

    @override
    def evaluate_gradient(self, parameter_vector: np.ndarray) -> np.ndarray:
        return self.precision_matrix @ (parameter_vector - self.mean_vector)

    @override
    def evaluate_hessian_vector_product(self, direction_vector: np.ndarray) -> np.ndarray:
        return self.precision_matrix @ direction_vector

    @override
    def generate_sample(self) -> np.ndarray:
        random_vector = self._rng.standard_normal(self.random_vector_size)
        return self.mean_vector + self.apply_covariance_factorization(random_vector)

    @override
    def apply_covariance_operator(self, parameter_vector: np.ndarray) -> np.ndarray:
        return self.covariance_matrix @ parameter_vector

    @override
    def apply_covariance_factorization(self, random_vector: np.ndarray) -> np.ndarray:
        return self.covariance_factor @ random_vector

    @override
    def apply_precision_operator(self, parameter_vector: np.ndarray) -> np.ndarray:
        return self.precision_matrix @ parameter_vector


# ==================================================================================================
@dataclass
class LikelihoodSetup:
    data_vector: np.ndarray
    precision_values: np.ndarray
    likelihood: likelihood.GaussianLogLikelihood
    observation_matrix: np.ndarray
    precision_matrix: np.ndarray


@dataclass
class PosteriorSetup:
    likelihood_setup: LikelihoodSetup
    parameter_to_solution_map: LinearParameterToSolutionMap
    prior: QuadraticPrior


def create_likelihood_setup(seed: int) -> LikelihoodSetup:
    rng = np.random.default_rng(seed)
    num_observations = OBSERVED_VERTEX_INDICES.shape[0]
    data_vector = rng.random(num_observations)
    precision_values = rng.uniform(0.5, 2.0, num_observations)
    observation_matrix = np.zeros((num_observations, NUM_VERTICES))
    observation_matrix[np.arange(num_observations), OBSERVED_VERTEX_INDICES] = 1.0
    precision_matrix = np.diag(precision_values)
    gaussian_likelihood = likelihood.GaussianLogLikelihood(
        data_vector, sp.coo_matrix(observation_matrix), sp.coo_matrix(precision_matrix)
    )
    return LikelihoodSetup(
        data_vector, precision_values, gaussian_likelihood, observation_matrix, precision_matrix
    )


def create_dense_spd_likelihood_setup(seed: int) -> LikelihoodSetup:
    """Likelihood setup with a dense, non-diagonal SPD precision matrix."""
    rng = np.random.default_rng(seed)
    num_observations = OBSERVED_VERTEX_INDICES.shape[0]
    data_vector = rng.random(num_observations)
    observation_matrix = np.zeros((num_observations, NUM_VERTICES))
    observation_matrix[np.arange(num_observations), OBSERVED_VERTEX_INDICES] = 1.0
    precision_matrix = random_spd_matrix(rng, num_observations)
    gaussian_likelihood = likelihood.GaussianLogLikelihood(
        data_vector, sp.coo_matrix(observation_matrix), sp.coo_matrix(precision_matrix)
    )
    return LikelihoodSetup(
        data_vector,
        np.diag(precision_matrix).copy(),
        gaussian_likelihood,
        observation_matrix,
        precision_matrix,
    )


def create_posterior_setup(likelihood_setup: LikelihoodSetup, seed: int) -> PosteriorSetup:
    rng = np.random.default_rng(seed)
    parameter_to_solution_map = LinearParameterToSolutionMap(
        rng.random((NUM_VERTICES, PARAMETER_DIM))
    )
    prior = QuadraticPrior(rng.random(PARAMETER_DIM), random_spd_matrix(rng, PARAMETER_DIM), seed=2)
    return PosteriorSetup(likelihood_setup, parameter_to_solution_map, prior)


def create_nonlinear_posterior_setup(
    likelihood_setup: LikelihoodSetup, seed: int
) -> PosteriorSetup:
    rng = np.random.default_rng(seed)
    parameter_to_solution_map = NonlinearParameterToSolutionMap(
        rng.random((NUM_VERTICES, PARAMETER_DIM))
    )
    prior = QuadraticPrior(rng.random(PARAMETER_DIM), random_spd_matrix(rng, PARAMETER_DIM), seed=2)
    return PosteriorSetup(likelihood_setup, parameter_to_solution_map, prior)


# ==================================================================================================
POSTERIOR_TUTORIALS_DIR = notebook_helpers.REPO_ROOT / "tutorials" / "posterior"
POSTERIOR_NOTEBOOK = POSTERIOR_TUTORIALS_DIR / "posterior.ipynb"
NOTEBOOK_EXECUTION_TIMEOUT_SECONDS = 120
