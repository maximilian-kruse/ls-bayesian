"""L-BFGS-B optimizer, wrapping `scipy.optimize.minimize`.

Classes:
    LBFGSSettings: Settings for the L-BFGS-B optimizer.
    LBFGSOptimizer: L-BFGS-B optimizer.
"""

from collections.abc import Callable
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
    OptimizationHistory,
    OptimizationResult,
)

# Defaults for `LBFGSSettings`, matching `scipy.optimize.minimize`'s own defaults for
# `method="L-BFGS-B"` (`maxiter`, `ftol`, `gtol`, `maxls`), so that constructing `LBFGSSettings`
# with no arguments reproduces scipy's out-of-the-box behavior.
DEFAULT_MAXIMUM_NUM_ITERATIONS = 15000
DEFAULT_RELATIVE_FUNCTION_TOLERANCE = float(np.finfo(np.float64).eps)
DEFAULT_RELATIVE_GRADIENT_TOLERANCE = 1e-5
DEFAULT_MAX_LINE_SEARCH_STEPS = 20


# ==================================================================================================
@dataclass
class LBFGSSettings:
    """Settings for the L-BFGS-B optimizer, passed to `scipy.optimize.minimize`.

    The field constraints are validated on initialization. Defaults reproduce
    `scipy.optimize.minimize`'s own defaults for `method="L-BFGS-B"`.

    Attributes:
        maximum_num_iterations (int): Maximum number of iterations. Defaults to
            `DEFAULT_MAXIMUM_NUM_ITERATIONS`.
        relative_function_tolerance (Real): Relative tolerance (`ftol`) on the loss decrease
            between iterations for convergence. Defaults to
            `DEFAULT_RELATIVE_FUNCTION_TOLERANCE`.
        relative_gradient_tolerance (Real): Tolerance (`gtol`) on the (projected) gradient norm
            for convergence. Defaults to `DEFAULT_RELATIVE_GRADIENT_TOLERANCE`.
        max_line_search_steps (int): Maximum number of line-search steps per iteration. Defaults
            to `DEFAULT_MAX_LINE_SEARCH_STEPS`.
    """

    maximum_num_iterations: Annotated[int, Is[lambda x: x > 0]] = DEFAULT_MAXIMUM_NUM_ITERATIONS
    relative_function_tolerance: Annotated[Real, Is[lambda x: x > 0]] = (
        DEFAULT_RELATIVE_FUNCTION_TOLERANCE
    )
    relative_gradient_tolerance: Annotated[Real, Is[lambda x: x > 0]] = (
        DEFAULT_RELATIVE_GRADIENT_TOLERANCE
    )
    max_line_search_steps: Annotated[int, Is[lambda x: x > 0]] = DEFAULT_MAX_LINE_SEARCH_STEPS


# ==================================================================================================
class LBFGSOptimizer(BaseOptimizer):
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
    def __init__(self, settings: LBFGSSettings, logger: BaseLogger | None = None) -> None:
        """Initialize the optimizer.

        Args:
            settings (LBFGSSettings): Settings for the L-BFGS-B optimizer.
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
        callback: Callable[..., None],
    ) -> spo.OptimizeResult:
        """Call `scipy.optimize.minimize(method="L-BFGS-B")`.

        `model.evaluate_hessian_vector_product` and `model.evaluate_inner_product` are never
        called: L-BFGS-B only requires gradient information, and scipy's implementation is
        hardcoded to the Euclidean inner product.
        """
        options = {
            "maxiter": self._settings.maximum_num_iterations,
            "ftol": self._settings.relative_function_tolerance,
            "gtol": self._settings.relative_gradient_tolerance,
            "maxls": self._settings.max_line_search_steps,
        }
        return spo.minimize(
            fun=model.evaluate_cost,
            x0=initial_guess,
            jac=model.evaluate_gradient,
            method="L-BFGS-B",
            callback=callback,
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
