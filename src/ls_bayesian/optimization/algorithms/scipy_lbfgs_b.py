"""L-BFGS-B optimizer, wrapping `scipy.optimize.minimize`.

Classes:
    ScipyLBFGSBSettings: Settings for the L-BFGS-B optimizer.
    ScipyLBFGSBOptimizer: L-BFGS-B optimizer.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Annotated, override

import numpy as np
import scipy.optimize as spo
from beartype.vale import Is

from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.optimization.model import OptimizationModel
from ls_bayesian.optimization.optimizer import (
    BaseOptimizer,
    IterationCallback,
    OptimizationHistory,
    OptimizationResult,
)


# ==================================================================================================
@dataclass
class ScipyLBFGSBSettings:
    """Settings for the L-BFGS-B optimizer, passed to `scipy.optimize.minimize`.

    The field constraints are validated on initialization. Defaults reproduce
    `scipy.optimize.minimize`'s own defaults for `method="L-BFGS-B"` (`maxiter`, `ftol`, `gtol`,
    `maxls`), so that constructing `ScipyLBFGSBSettings` with no arguments reproduces scipy's
    out-of-the-box behavior.

    Attributes:
        maximum_num_iterations (int): Maximum number of iterations. Defaults to `15000`.
        relative_function_tolerance (Real): Relative tolerance (`ftol`) on the loss decrease
            between iterations for convergence. Defaults to machine epsilon for `float64`.
        relative_gradient_tolerance (Real): Tolerance (`gtol`) on the (projected) gradient norm
            for convergence. Defaults to `1e-5`.
        max_line_search_steps (int): Maximum number of line-search steps per iteration. Defaults
            to `20`.
    """

    maximum_num_iterations: Annotated[int, Is[lambda x: x > 0]] = 15000
    relative_function_tolerance: Annotated[Real, Is[lambda x: x > 0]] = float(
        np.finfo(np.float64).eps
    )
    relative_gradient_tolerance: Annotated[Real, Is[lambda x: x > 0]] = 1e-5
    max_line_search_steps: Annotated[int, Is[lambda x: x > 0]] = 20


# ==================================================================================================
class ScipyLBFGSBOptimizer(BaseOptimizer):
    """L-BFGS-B optimizer, wrapping `scipy.optimize.minimize(method="L-BFGS-B")`.

    A mature, well-tested quasi-Newton optimizer for the standard Euclidean geometry. For problems
    requiring a custom inner product (e.g. a Cameron-Martin metric induced by a Bayesian prior),
    use a metric-generic backend instead; scipy's L-BFGS-B cannot be parameterized by a custom
    inner product.

    Attributes:
        requires_hessian (bool): Always `False`, L-BFGS-B does not use Hessian information.
    """

    requires_hessian: bool = False

    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: ScipyLBFGSBSettings, logger: BaseLogger | None = None) -> None:
        """Initialize the optimizer.

        Args:
            settings (ScipyLBFGSBSettings): Settings for the L-BFGS-B optimizer.
            logger (BaseLogger | None, optional): Logger for iteration-by-iteration progress
                reports. Defaults to `None`.
        """
        super().__init__(logger=logger)
        self._settings = settings

    # ----------------------------------------------------------------------------------------------
    @override
    def _run_impl(
        self,
        initial_guess: np.ndarray[tuple[int], np.dtype[np.float64]],
        model: OptimizationModel,
        callback: IterationCallback,
    ) -> spo.OptimizeResult:
        """Call `scipy.optimize.minimize(method="L-BFGS-B")`.

        `model.evaluate_hessian_vector_product` and `model.evaluate_inner_product` are never
        called: L-BFGS-B only requires gradient information, and scipy's implementation is
        hardcoded to the Euclidean inner product.

        scipy's own callback reports the loss of each accepted iterate directly (via the
        `intermediate_result.fun` convention, see `scipy.optimize.minimize`'s `callback`
        parameter), but not the gradient there. `latest_gradient_norm` bridges that gap: `jac`
        caches the norm of whatever gradient it last computed, which -- since scipy always
        evaluates `fun`/`jac` together at a candidate point before deciding to accept it as the
        next iterate -- is exactly the gradient norm at the point scipy reports through the
        callback. This cache is not itself a record; only `callback` (the run's `IterationCallback`,
        invoked from `report_iteration`) writes into the run's history.
        """
        options = {
            "maxiter": self._settings.maximum_num_iterations,
            "ftol": self._settings.relative_function_tolerance,
            "gtol": self._settings.relative_gradient_tolerance,
            "maxls": self._settings.max_line_search_steps,
        }
        latest_gradient_norm = np.nan

        # cache gradient to avoid extra evaluation for reporting
        def evaluate_gradient_and_cache_norm(
            parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
            nonlocal latest_gradient_norm
            gradient = model.evaluate_gradient(parameter_vector)
            latest_gradient_norm = model.evaluate_norm(gradient)
            return gradient

        # Custom callback with latest cached gradient
        def report_iteration(intermediate_result: spo.OptimizeResult) -> None:
            callback(float(intermediate_result.fun), latest_gradient_norm)

        return spo.minimize(
            fun=model.evaluate_cost,
            x0=initial_guess,
            jac=evaluate_gradient_and_cache_norm,
            method="L-BFGS-B",
            callback=report_iteration,
            options=options,
        )

    # ----------------------------------------------------------------------------------------------
    @override
    def _create_optimization_result(
        self, raw_result: spo.OptimizeResult, history: OptimizationHistory
    ) -> OptimizationResult:
        """Map `scipy.optimize.OptimizeResult` and the recorded history onto
        `OptimizationResult`."""
        return OptimizationResult(
            result=np.asarray(raw_result.x, dtype=np.float64),
            loss_history=history.loss_history,
            gradient_norm_history=history.gradient_norm_history,
            num_iterations=int(raw_result.nit),
            success=bool(raw_result.success),
            status_message=str(raw_result.message),
        )
