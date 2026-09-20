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
        evaluate_cost: Evaluate $J(m)$.
        evaluate_cost_components: Evaluate the likelihood and prior contributions to $J(m)$
            separately.
        evaluate_gradient: Evaluate $\nabla_m J(m)$.
        evaluate_gradient_components: Evaluate the likelihood and prior contributions to
            $\nabla_m J(m)$ separately.
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
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        r"""Evaluate the negative log-posterior $J(m) = \Phi(F(m)) + R(m)$.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter $m$.

        Returns:
            float: $J(m)$.
        """
        self._log_debug_evaluation_start("Cost evaluation", parameter_vector)
        likelihood_cost, prior_cost = self._compute_cost_components(parameter_vector)
        total_cost = likelihood_cost + prior_cost
        self._log_debug_message(f"likelihood_cost: {likelihood_cost}")
        self._log_debug_message(f"prior_cost: {prior_cost}")
        self._log_debug_message(f"total_cost: {total_cost}")
        self._warn_if_not_finite("likelihood_cost", likelihood_cost)
        self._warn_if_not_finite("prior_cost", prior_cost)
        self._warn_if_not_finite("total_cost", total_cost)
        return total_cost

    # ----------------------------------------------------------------------------------------------
    def evaluate_cost_components(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> tuple[float, float]:
        r"""Evaluate the likelihood and prior contributions to $J(m)$ separately.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter $m$.

        Returns:
            tuple[float, float]: $(\Phi(F(m)), R(m))$.
        """
        self._log_debug_evaluation_start("Cost evaluation", parameter_vector)
        likelihood_cost, prior_cost = self._compute_cost_components(parameter_vector)
        self._log_debug_message(f"likelihood_cost: {likelihood_cost}")
        self._log_debug_message(f"prior_cost: {prior_cost}")
        self._warn_if_not_finite("likelihood_cost", likelihood_cost)
        self._warn_if_not_finite("prior_cost", prior_cost)
        return likelihood_cost, prior_cost

    # ----------------------------------------------------------------------------------------------
    def evaluate_gradient(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Evaluate the gradient $\nabla_m J(m)$ of the negative log-posterior.

        The likelihood contribution $(\nabla_m F(m))^T \nabla_u \Phi(u)$ is cached, so repeated
        gradient evaluations at the same parameter only re-evaluate the prior gradient.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter $m$.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: $\nabla_m J(m)$.
        """
        self._log_debug_evaluation_start("Gradient evaluation", parameter_vector)
        likelihood_gradient, prior_gradient = self._compute_gradient_components(parameter_vector)
        total_gradient = likelihood_gradient + prior_gradient
        self._log_debug_vector_statistics("likelihood_gradient", likelihood_gradient)
        self._log_debug_vector_statistics("prior_gradient", prior_gradient)
        self._log_debug_vector_statistics("total_gradient", total_gradient)
        self._warn_if_not_finite_vector("likelihood_gradient", likelihood_gradient)
        self._warn_if_not_finite_vector("prior_gradient", prior_gradient)
        self._warn_if_not_finite_vector("total_gradient", total_gradient)
        return total_gradient

    # ----------------------------------------------------------------------------------------------
    def evaluate_gradient_components(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]
    ]:
        r"""Evaluate the likelihood and prior contributions to $\nabla_m J(m)$ separately.

        The likelihood contribution $(\nabla_m F(m))^T \nabla_u \Phi(u)$ is cached, so repeated
        gradient evaluations at the same parameter only re-evaluate the prior gradient.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter $m$.

        Returns:
            tuple[np.ndarray[tuple[int], np.dtype[np.float64]], ...]: The likelihood contribution
                $(\nabla_m F(m))^T \nabla_u \Phi(u)$ and the prior contribution $\nabla_m R(m)$,
                each the same shape as the parameter. Neither array aliases the cache.
        """
        self._log_debug_evaluation_start("Gradient evaluation", parameter_vector)
        likelihood_gradient, prior_gradient = self._compute_gradient_components(parameter_vector)
        self._log_debug_vector_statistics("likelihood_gradient", likelihood_gradient)
        self._log_debug_vector_statistics("prior_gradient", prior_gradient)
        self._warn_if_not_finite_vector("likelihood_gradient", likelihood_gradient)
        self._warn_if_not_finite_vector("prior_gradient", prior_gradient)
        return likelihood_gradient.copy(), prior_gradient

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
    def _compute_cost_components(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> tuple[float, float]:
        r"""Compute $(\Phi(F(m)), R(m))$, without logging."""
        # The cache stores the parameter vector as given, so it must not alias the caller's array,
        # which optimizers may modify in-place between evaluations.
        parameter_vector = parameter_vector.copy()
        solution_vector = self._retrieve_or_compute_forward_solution(parameter_vector)
        likelihood_cost = self._likelihood.evaluate_cost(solution_vector)
        prior_cost = self._prior.evaluate_cost(parameter_vector)
        return likelihood_cost, prior_cost

    # ----------------------------------------------------------------------------------------------
    def _compute_gradient_components(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]
    ]:
        r"""Compute the likelihood contribution $(\nabla_m F(m))^T \nabla_u \Phi(u)$ and the prior
        contribution $\nabla_m R(m)$, without logging. The likelihood contribution aliases the
        cache and must not be exposed to callers without copying first."""
        # The cache stores the parameter vector as given, so it must not alias the caller's array,
        # which optimizers may modify in-place between evaluations.
        parameter_vector = parameter_vector.copy()
        likelihood_gradient = self._retrieve_or_compute_likelihood_parameter_gradient(
            parameter_vector
        )
        prior_gradient = self._prior.evaluate_gradient(parameter_vector)
        return likelihood_gradient, prior_gradient

    # ----------------------------------------------------------------------------------------------
    def _retrieve_or_compute_forward_solution(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Return the forward solution $F(m)$, solving the forward problem only if not cached."""
        solution_vector = self._cache.retrieve_quantity(_CachedQuantity.SOLUTION, parameter_vector)
        if solution_vector is None:
            self._log_debug_message("Cache miss for SOLUTION, solving forward problem.")
            solution_vector = self._parameter_to_solution_map.evaluate_forward(parameter_vector)
            self._cache.store_quantity(_CachedQuantity.SOLUTION, parameter_vector, solution_vector)
        else:
            self._log_debug_message("Cache hit for SOLUTION.")
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
            self._log_debug_message(
                "Cache miss for LIKELIHOOD_SOLUTION_GRADIENT, evaluating gradient."
            )
            solution_vector = self._retrieve_or_compute_forward_solution(parameter_vector)
            likelihood_solution_gradient = self._likelihood.evaluate_gradient(solution_vector)
            self._cache.store_quantity(
                _CachedQuantity.LIKELIHOOD_SOLUTION_GRADIENT,
                parameter_vector,
                likelihood_solution_gradient,
            )
        else:
            self._log_debug_message("Cache hit for LIKELIHOOD_SOLUTION_GRADIENT.")
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
            self._log_debug_message(
                "Cache miss for LIKELIHOOD_PARAMETER_GRADIENT, evaluating gradient."
            )
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
        else:
            self._log_debug_message("Cache hit for LIKELIHOOD_PARAMETER_GRADIENT.")
        return likelihood_parameter_gradient

    # ----------------------------------------------------------------------------------------------
    def _log_debug_evaluation_start(
        self, message: str, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> None:
        """Log the start of an evaluation and the parameter it is evaluated at."""
        self._log_debug_message(message)
        self._log_debug_vector_statistics("parameter_vector", parameter_vector)

    # ----------------------------------------------------------------------------------------------
    def _log_debug_message(self, message: str) -> None:
        """Log a message, if a logger is attached."""
        if self._logger is not None:
            self._logger.debug(message)

    # ----------------------------------------------------------------------------------------------
    def _log_debug_vector_statistics(
        self, name: str, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> None:
        """Log value range and Euclidean norm of a vector, if a logger is attached."""
        if self._logger is not None:
            self._logger.debug(
                f"{name} in: [{np.min(vector)}, {np.max(vector)}], norm: {np.linalg.norm(vector)}"
            )

    # ----------------------------------------------------------------------------------------------
    def _warn_if_not_finite(self, name: str, value: float) -> None:
        """Log a warning if a scalar value is not finite, if a logger is attached.

        A non-finite cost usually signals a diverged forward solve or a numerically unstable
        likelihood/prior evaluation; this is only a warning, not an error, since a line search may
        legitimately probe points where the cost is temporarily non-finite before rejecting them.
        """
        if self._logger is not None and not np.isfinite(value):
            self._logger.warning(f"{name} is not finite: {value}.")

    # ----------------------------------------------------------------------------------------------
    def _warn_if_not_finite_vector(
        self, name: str, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> None:
        """Log a warning if any vector entry is not finite, if a logger is attached.

        See
        [`_warn_if_not_finite`][ls_bayesian.posterior.posterior.LogPosterior._warn_if_not_finite]
        for why this is a warning rather than an error.
        """
        if self._logger is not None and not np.all(np.isfinite(vector)):
            self._logger.warning(f"{name} contains non-finite values.")
