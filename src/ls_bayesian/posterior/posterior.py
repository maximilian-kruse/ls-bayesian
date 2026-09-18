r"""Negative log-posterior, composed of likelihood, parameter-to-solution map and prior.

Classes:
    LogPosterior: Negative log-posterior $J(m) = \Phi(F(m)) + R(m)$.
"""

from enum import Enum, auto

import numpy as np

from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.posterior import cache, interfaces


# ==================================================================================================
class _CachedQuantity(Enum):
    """Intermediate results of a posterior evaluation that are reused across evaluations."""

    SOLUTION = auto()
    LIKELIHOOD_SOLUTION_GRADIENT = auto()
    LIKELIHOOD_PARAMETER_GRADIENT = auto()


# ==================================================================================================
class LogPosterior:
    r"""Negative log-posterior $J(m) = \Phi(F(m)) + R(m)$.

    The posterior composes three independent components, each adhering to an interface in the
    [`interfaces`][ls_bayesian.posterior.interfaces] module:

    - a [`ParameterToSolutionMap`][ls_bayesian.posterior.interfaces.ParameterToSolutionMap] $F$,
    - a [`Likelihood`][ls_bayesian.posterior.interfaces.Likelihood] $\Phi$ on the solution space,
    - a [`GaussianPrior`][ls_bayesian.posterior.interfaces.GaussianPrior] with negative
      log-density $R$ on the parameter space.

    The gradient follows from the chain rule,
    $\nabla_m J(m) = (\nabla_m F(m))^T \nabla_u \Phi(u) + \nabla_m R(m)$ with $u = F(m)$.

    Intermediate results that depend only on $m$, in particular the solution $u = F(m)$, are
    stored in an [`EvaluationCache`][ls_bayesian.posterior.cache.EvaluationCache]. Evaluating cost
    and gradient at the same parameter therefore solves the forward problem only once. For this to
    be valid, the components must be deterministic functions of their inputs and must not modify
    their input arrays in-place, as cached arrays are passed to them without copying. The components
    are not exposed for modification after construction.

    Methods:
        evaluate_cost: Evaluate $J(m)$, optionally split into likelihood and prior contributions.
        evaluate_gradient: Evaluate $\nabla_m J(m)$, optionally split into likelihood and prior
            contributions.
        evaluate_hessian_vector_product: Not implemented yet.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        likelihood: interfaces.Likelihood,
        parameter_to_solution_map: interfaces.ParameterToSolutionMap,
        prior: interfaces.GaussianPrior,
        logger: BaseLogger | None = None,
    ) -> None:
        r"""Initialize the posterior from its components.

        Args:
            likelihood (interfaces.Likelihood): Negative log-likelihood $\Phi$.
            parameter_to_solution_map (interfaces.ParameterToSolutionMap): Forward map $F$.
            prior (interfaces.GaussianPrior): Gaussian prior with negative log-density $R$.
            logger (BaseLogger | None, optional): Logger for evaluation diagnostics. Nothing is
                logged if `None`. Defaults to `None`.
        """
        self._likelihood = likelihood
        self._parameter_to_solution_map = parameter_to_solution_map
        self._prior = prior
        self._logger = logger
        self._cache = cache.EvaluationCache()

    # ----------------------------------------------------------------------------------------------
    def evaluate_cost(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        split: bool = False,
    ) -> float | tuple[float, float]:
        r"""Evaluate the negative log-posterior $J(m) = \Phi(F(m)) + R(m)$.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter $m$.
            split (bool, optional): If `True`, return the likelihood and prior contributions
                separately. Defaults to `False`.

        Returns:
            float | tuple[float, float]: $J(m)$, or $(\Phi(F(m)), R(m))$ if `split` is `True`.
        """
        # The cache stores the parameter vector as given, so it must not alias the caller's array,
        # which optimizers may modify in-place between evaluations.
        parameter_vector = parameter_vector.copy()
        self._log_message("Cost evaluation")
        self._log_vector_statistics("parameter_vector", parameter_vector)

        solution_vector = self._retrieve_or_compute_forward_solution(parameter_vector)
        likelihood_cost = self._likelihood.evaluate_cost(solution_vector)
        prior_cost = self._prior.evaluate_cost(parameter_vector)
        total_cost = likelihood_cost + prior_cost

        self._log_message(f"likelihood_cost: {likelihood_cost}")
        self._log_message(f"prior_cost: {prior_cost}")
        self._log_message(f"total_cost: {total_cost}")
        if split:
            return likelihood_cost, prior_cost
        return total_cost

    # ----------------------------------------------------------------------------------------------
    def evaluate_gradient(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        split: bool = False,
    ) -> (
        np.ndarray[tuple[int], np.dtype[np.float64]]
        | tuple[
            np.ndarray[tuple[int], np.dtype[np.float64]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ]
    ):
        r"""Evaluate the gradient $\nabla_m J(m)$ of the negative log-posterior.

        The likelihood contribution $(\nabla_m F(m))^T \nabla_u \Phi(u)$ is cached, so repeated
        gradient evaluations at the same parameter only re-evaluate the prior gradient.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter $m$.
            split (bool, optional): If `True`, return the likelihood and prior contributions
                separately. Defaults to `False`.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]] | tuple[...]: $\nabla_m J(m)$, or the
                likelihood and prior contributions if `split` is `True`.
        """
        # The cache stores the parameter vector as given, so it must not alias the caller's array,
        # which optimizers may modify in-place between evaluations.
        parameter_vector = parameter_vector.copy()
        self._log_message("Gradient evaluation")
        self._log_vector_statistics("parameter_vector", parameter_vector)

        likelihood_gradient = self._retrieve_or_compute_likelihood_parameter_gradient(
            parameter_vector
        )
        prior_gradient = self._prior.evaluate_gradient(parameter_vector)

        self._log_vector_statistics("likelihood_gradient", likelihood_gradient)
        self._log_vector_statistics("prior_gradient", prior_gradient)
        if split:
            return likelihood_gradient.copy(), prior_gradient
        return likelihood_gradient + prior_gradient

    # ----------------------------------------------------------------------------------------------
    def evaluate_hessian_vector_product(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Evaluate the Hessian-vector product $\nabla_m^2 J(m)\, \hat{m}$.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter $m$.
            direction_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Direction $\hat{m}$.

        Raises:
            NotImplementedError: Always, not implemented yet.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    def _retrieve_or_compute_forward_solution(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Return the forward solution $F(m)$, solving the forward problem only if not cached."""
        solution_vector = self._cache.retrieve_quantity(_CachedQuantity.SOLUTION, parameter_vector)
        if solution_vector is None:
            solution_vector = self._parameter_to_solution_map.evaluate_forward(parameter_vector)
            self._cache.store_quantity(_CachedQuantity.SOLUTION, parameter_vector, solution_vector)
        return solution_vector

    # ----------------------------------------------------------------------------------------------
    def _retrieve_or_compute_likelihood_solution_gradient(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Return $\nabla_u \Phi(u)$ at $u = F(m)$, computing it only if not cached."""
        likelihood_solution_gradient = self._cache.retrieve_quantity(
            _CachedQuantity.LIKELIHOOD_SOLUTION_GRADIENT, parameter_vector
        )
        if likelihood_solution_gradient is None:
            solution_vector = self._retrieve_or_compute_forward_solution(parameter_vector)
            likelihood_solution_gradient = self._likelihood.evaluate_gradient(solution_vector)
            self._cache.store_quantity(
                _CachedQuantity.LIKELIHOOD_SOLUTION_GRADIENT,
                parameter_vector,
                likelihood_solution_gradient,
            )
        return likelihood_solution_gradient

    # ----------------------------------------------------------------------------------------------
    def _retrieve_or_compute_likelihood_parameter_gradient(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Return $(\nabla_m F(m))^T \nabla_u \Phi(u)$, computing it only if not cached."""
        likelihood_parameter_gradient = self._cache.retrieve_quantity(
            _CachedQuantity.LIKELIHOOD_PARAMETER_GRADIENT, parameter_vector
        )
        if likelihood_parameter_gradient is None:
            solution_vector = self._retrieve_or_compute_forward_solution(parameter_vector)
            likelihood_solution_gradient = self._retrieve_or_compute_likelihood_solution_gradient(
                parameter_vector
            )
            likelihood_parameter_gradient = self._parameter_to_solution_map.evaluate_gradient(
                solution_vector, parameter_vector, likelihood_solution_gradient
            )
            self._cache.store_quantity(
                _CachedQuantity.LIKELIHOOD_PARAMETER_GRADIENT,
                parameter_vector,
                likelihood_parameter_gradient,
            )
        return likelihood_parameter_gradient

    # ----------------------------------------------------------------------------------------------
    def _log_message(self, message: str) -> None:
        """Log a message, if a logger is attached."""
        if self._logger is not None:
            self._logger.info(message)

    # ----------------------------------------------------------------------------------------------
    def _log_vector_statistics(
        self, name: str, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> None:
        """Log value range and Euclidean norm of a vector, if a logger is attached."""
        if self._logger is not None:
            self._logger.info(
                f"{name} in: [{np.min(vector)}, {np.max(vector)}], norm: {np.linalg.norm(vector)}"
            )
