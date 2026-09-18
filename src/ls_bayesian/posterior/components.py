from abc import ABC, abstractmethod
from typing import override

import numpy as np
import scipy.sparse as sp


# ==================================================================================================
class Likelihood(ABC):
    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_cost(self, solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_gradient(
        self, solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_hessian_vector_product(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError


# ==================================================================================================
class ParameterToSolutionMap(ABC):
    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_forward(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_gradient(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        adjoint_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_jacobian_vector_product(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_hessian_vector_product(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        adjoint_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        gradient_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError


# ==================================================================================================
class Prior(ABC):
    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_cost(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_gradient(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_hessian_vector_product(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def generate_sample(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError


# ==================================================================================================
class GaussianLogLikelihood(Likelihood):
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        data_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        observation_matrix: sp.coo_matrix,
        precision_matrix: sp.coo_matrix,
    ):
        self.observation_matrix = observation_matrix
        self.precision_matrix = precision_matrix
        self._data_vector = data_vector

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_cost(self, solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        difference_vector = self.observation_matrix @ solution_vector - self._data_vector
        cost = 0.5 * difference_vector.T @ self.precision_matrix @ difference_vector
        return cost

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_gradient(
        self, solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        difference_vector = self.observation_matrix @ solution_vector - self._data_vector
        gradient = self.observation_matrix.T @ self.precision_matrix @ difference_vector
        return gradient

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_hessian_vector_product(
        self,
        _solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        hvp = (
            self.observation_matrix.T
            @ self.precision_matrix
            @ self.observation_matrix
            @ direction_vector
        )
        return hvp
