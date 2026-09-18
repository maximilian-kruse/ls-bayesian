import numpy as np

from . import components, logging


# ==================================================================================================
class CachedState:
    # ----------------------------------------------------------------------------------------------
    def __init__(self):
        self._parameter_vector = None
        self._solution_vector = None
        self._gradient_vector = None

    # ----------------------------------------------------------------------------------------------
    def get_parameter_vector(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._parameter_vector

    # ----------------------------------------------------------------------------------------------
    def set_parameter_vector(self, value: np.ndarray[tuple[int], np.dtype[np.float64]]) -> None:
        self._parameter_vector = value
        self._solution_vector = None
        self._gradient_vector = None

    # ----------------------------------------------------------------------------------------------
    def get_solution_vector(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        if self._parameter_vector is None or not np.allclose(
            parameter_vector, self._parameter_vector
        ):
            return None
        else:
            return self._solution_vector

    # ----------------------------------------------------------------------------------------------
    def set_solution_vector(
        self,
        value: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        if self._parameter_vector is None or not np.allclose(
            parameter_vector, self._parameter_vector
        ):
            raise ValueError(
                "given parameter vector does not match cached parameter vector, or"
                " no parameter vector is cached."
            )
        self._solution_vector = value

    # ----------------------------------------------------------------------------------------------
    def get_gradient_vector(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        if self._parameter_vector is None or not np.allclose(
            parameter_vector, self._parameter_vector
        ):
            return None
        else:
            return self._gradient_vector

    # ----------------------------------------------------------------------------------------------
    def set_gradient_vector(
        self,
        value: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        if self._parameter_vector is None or not np.allclose(
            parameter_vector, self._parameter_vector
        ):
            raise ValueError(
                "given parameter vector does not match cached parameter vector, or"
                " no parameter vector is cached."
            )
        self._gradient_vector = value


# ==================================================================================================
class LogPosterior:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        likelihood: components.Likelihood,
        parameter_to_solution_map: components.ParameterToSolutionMap,
        prior: components.Prior,
        logger: logging.LSBIPLogger | None = None,
    ):
        self.likelihood = likelihood
        self.parameter_to_solution_map = parameter_to_solution_map
        self.prior = prior
        self._cached_state = CachedState()
        self._logger = logger

    # ----------------------------------------------------------------------------------------------
    def evaluate_cost(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        split: bool = False,
    ) -> float | tuple[float, float]:
        if self._logger:
            self._logger.log_header("Cost Evaluation")
            self._logger.info(
                f"Parameter_vector in: [{np.min(parameter_vector)}, {np.max(parameter_vector)}]"
            )
        solution_vector = self.parameter_to_solution_map.evaluate_forward(parameter_vector)
        likelihood_cost = self.likelihood.evaluate_cost(solution_vector)
        prior_cost = self.prior.evaluate_cost(parameter_vector)
        total_cost = likelihood_cost + prior_cost
        if self._logger:
            self._logger.info(f"prior_cost: {prior_cost}")
            self._logger.info(f"likelihood_cost: {likelihood_cost}")
            self._logger.info(f"total_cost: {total_cost}")
        self._cached_state.set_parameter_vector(parameter_vector)
        self._cached_state.set_solution_vector(solution_vector, parameter_vector)
        if split:
            return likelihood_cost, prior_cost
        else:
            return total_cost

    # ----------------------------------------------------------------------------------------------
    def evaluate_gradient(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]], split: bool = False
    ) -> (
        np.ndarray[tuple[int], np.dtype[np.float64]]
        | tuple[
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ]
    ):
        if self._logger:
            self._logger.log_header("Gradient Evaluation")
            self._logger.info(
                f"Parameter_vector in: [{np.min(parameter_vector)}, {np.max(parameter_vector)}]"
            )
        solution_vector = self._cached_state.get_solution_vector(parameter_vector)
        if solution_vector is None:
            solution_vector = self.parameter_to_solution_map.evaluate_forward(parameter_vector)
            self._cached_state.set_parameter_vector(parameter_vector)
            self._cached_state.set_solution_vector(solution_vector, parameter_vector)
        likelihood_gradient = self.likelihood.evaluate_gradient(solution_vector)
        if self._logger:
            self._logger.info(
                f"likelihood_gradient in: [{np.min(likelihood_gradient)}, {np.max(likelihood_gradient)}]"
            )
            self._logger.info(f"likelihood_gradient norm: {np.linalg.norm(likelihood_gradient)}")
        pts_gradient = self.parameter_to_solution_map.evaluate_gradient(
            solution_vector, parameter_vector, likelihood_gradient
        )
        if self._logger:
            self._logger.info(f"pts_gradient in: [{np.min(pts_gradient)}, {np.max(pts_gradient)}]")
            self._logger.info(f"pts_gradient norm: {np.linalg.norm(pts_gradient)}")
        prior_gradient = self.prior.evaluate_gradient(parameter_vector)
        if self._logger:
            self._logger.info(
                f"prior_gradient in: [{np.min(prior_gradient)}, {np.max(prior_gradient)}]"
            )
            self._logger.info(f"prior_gradient norm: {np.linalg.norm(prior_gradient)}")
        total_gradient = pts_gradient + prior_gradient
        self._cached_state.set_gradient_vector(pts_gradient, parameter_vector)
        if split:
            return pts_gradient, prior_gradient
        else:
            return total_gradient

    # ----------------------------------------------------------------------------------------------
    def evaluate_hessian_vector_product(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError
