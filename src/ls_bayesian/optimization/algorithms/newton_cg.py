r"""Truncated Newton (Newton-CG) in an arbitrary inner-product space.

Classes:
    NewtonCGSettings: Settings for `NewtonCGOptimizer`.
    NewtonCGOptimizer: Inexact Newton method with a truncated CG inner solve and a line search.
"""

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Annotated, override

import numpy as np
from beartype.vale import Is

from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.optimization.components.line_search import (
    LineSearchProblem,
    LineSearchStrategy,
)
from ls_bayesian.optimization.components.truncated_cg import (
    TruncatedCGResult,
    TruncatedCGSettings,
    TruncatedConjugateGradient,
)
from ls_bayesian.optimization.model import OptimizationModel
from ls_bayesian.optimization.optimizer import (
    BaseOptimizer,
    IterationCallback,
    OptimizationHistory,
    OptimizationResult,
)


# ==================================================================================================
@dataclass
class NewtonCGSettings:
    r"""Settings for `NewtonCGOptimizer`.

    The field constraints are validated on initialization.

    Attributes:
        maximum_num_iterations (int): Maximum number of outer (Newton) iterations. Defaults to
            `100`; a Newton-type method needs far fewer iterations than a quasi-Newton method.
        gradient_norm_tolerance (Real): Convergence tolerance on the gradient norm, evaluated in
            the optimizer's inner-product space. Defaults to `1e-6`, a typical convergence
            tolerance for double-precision optimization.
        maximum_forcing_term (Real): Upper bound $\eta_\mathrm{max}$ of the relative residual
            tolerance of the inner CG solve. Defaults to `0.5`, the bound for which the
            forcing sequence $\eta_k = \min\{\eta_\mathrm{max}, \sqrt{\|\mathbf{g}_k\|}\}$
            gives superlinear convergence (Nocedal & Wright, "Numerical Optimization", 2006,
            Sec. 7.1).
        minimum_forcing_term (Real): Lower bound of the relative residual tolerance. Defaults to
            `1e-6`: below this, CG would chase the noise of a Hessian-vector product that is
            evaluated with a single-precision forward model.
        cg_settings (TruncatedCGSettings): Settings of the inner CG solve.
    """

    maximum_num_iterations: Annotated[int, Is[lambda x: x > 0]] = 100
    gradient_norm_tolerance: Annotated[Real, Is[lambda x: x > 0]] = 1e-6
    maximum_forcing_term: Annotated[Real, Is[lambda x: 0 < x < 1]] = 0.5
    minimum_forcing_term: Annotated[Real, Is[lambda x: 0 < x < 1]] = 1e-6
    cg_settings: TruncatedCGSettings = field(default_factory=TruncatedCGSettings)

    def __post_init__(self) -> None:
        if self.minimum_forcing_term > self.maximum_forcing_term:
            raise ValueError(
                f"minimum_forcing_term ({self.minimum_forcing_term}) must not exceed "
                f"maximum_forcing_term ({self.maximum_forcing_term})."
            )


# ==================================================================================================
@dataclass(frozen=True)
class _NewtonCGRawResult:
    """Raw result of `NewtonCGOptimizer._run_impl`, mapped onto `OptimizationResult` by
    `_create_optimization_result`."""

    final_point: np.ndarray[tuple[int], np.dtype[np.float64]]
    num_iterations: int
    num_cg_iterations: int
    converged: bool
    final_gradient_norm: float
    line_search_failed: bool = False


# ==================================================================================================
class NewtonCGOptimizer(BaseOptimizer):
    r"""Truncated Newton method (Newton-CG) in an arbitrary inner-product space.

    In every iteration the Newton system $\mathbf{H}_k\mathbf{p} = -\mathbf{g}_k$ is solved
    approximately by
    [`TruncatedConjugateGradient`][ls_bayesian.optimization.components.truncated_cg.TruncatedConjugateGradient]
    with the relative tolerance $\eta_k = \min\{\eta_\mathrm{max}, \sqrt{\|\mathbf{g}_k\|}\}$
    (clipped from below), using only Hessian-vector products from
    `model.evaluate_hessian_vector_product`. The step size comes from an injected line search, so
    the iteration is globalized by the loss itself and the Hessian may be a Gauss-Newton
    approximation. If the CG direction is not a descent direction in floating point, the
    steepest-descent direction $-\mathbf{g}_k$ is used instead.

    **`model.evaluate_gradient` and `model.evaluate_hessian_vector_product` must return Riesz
    representers under `model.evaluate_inner_product`**, as for
    [`CustomLBFGSOptimizer`][ls_bayesian.optimization.algorithms.custom_lbfgs.CustomLBFGSOptimizer].
    In a Cameron-Martin space the prior is the identity part of the Hessian, which is what makes
    CG converge in few iterations: it only has to resolve the data-dominated directions.

    On a line-search failure the run terminates gracefully with the last accepted iterate and
    `success=False`.

    Attributes:
        requires_hessian (bool): Always `True`.
    """

    requires_hessian: bool = True

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        settings: NewtonCGSettings,
        line_search: LineSearchStrategy,
        logger: BaseLogger | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            settings (NewtonCGSettings): Settings for the optimizer.
            line_search (LineSearchStrategy): Step-size selection strategy.
            logger (BaseLogger | None, optional): Logger for iteration-by-iteration progress
                reports. Defaults to `None`.
        """
        super().__init__(logger=logger)
        self._settings = settings
        self._line_search = line_search
        self._conjugate_gradient = TruncatedConjugateGradient(settings.cg_settings)

    # ----------------------------------------------------------------------------------------------
    def _forcing_term(self, gradient_norm: float) -> float:
        r"""Relative residual tolerance $\eta_k$ of the inner solve for the current gradient."""
        forcing_term = min(float(self._settings.maximum_forcing_term), math.sqrt(gradient_norm))
        return max(forcing_term, float(self._settings.minimum_forcing_term))

    # ----------------------------------------------------------------------------------------------
    @override
    def _run_impl(
        self,
        initial_guess: np.ndarray[tuple[int], np.dtype[np.float64]],
        model: OptimizationModel,
        callback: IterationCallback,
    ) -> _NewtonCGRawResult:
        """Run the outer iteration: truncated CG for the direction, line search for the step,
        until convergence, `maximum_num_iterations`, or a line-search failure. In the last case the
        last accepted iterate is returned and the failed iteration is not reported to `callback`.
        """
        loss_function = model.evaluate_cost
        gradient_function = model.evaluate_gradient
        inner_product = model.evaluate_inner_product

        current_point = initial_guess.copy()
        current_loss = loss_function(current_point)
        current_gradient = gradient_function(current_point)
        gradient_norm = model.evaluate_norm(current_gradient)
        iteration = 0
        total_cg_iterations = 0
        converged = gradient_norm <= self._settings.gradient_norm_tolerance
        line_search_failed = False

        while not converged and iteration < self._settings.maximum_num_iterations:
            point = current_point

            def hessian_action(
                direction: np.ndarray[tuple[int], np.dtype[np.float64]],
                point: np.ndarray[tuple[int], np.dtype[np.float64]] = point,
            ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
                return model.evaluate_hessian_vector_product(point, direction)

            cg_result = self._conjugate_gradient.solve(
                hessian_action,
                current_gradient,
                inner_product,
                self._forcing_term(gradient_norm),
            )
            total_cg_iterations += cg_result.num_iterations
            search_direction = cg_result.direction
            if inner_product(current_gradient, search_direction) >= 0.0:
                search_direction = -current_gradient

            line_search_result = self._line_search.find_step_size(
                LineSearchProblem(
                    current_point=current_point,
                    search_direction=search_direction,
                    current_loss=current_loss,
                    current_gradient=current_gradient,
                    loss_function=loss_function,
                    gradient_function=gradient_function,
                    inner_product=inner_product,
                )
            )
            if not line_search_result.success:
                line_search_failed = True
                break
            current_point = current_point + line_search_result.step_size * search_direction
            current_loss = line_search_result.loss
            current_gradient = gradient_function(current_point)
            gradient_norm = model.evaluate_norm(current_gradient)
            iteration += 1
            self._log_debug_iteration_detail(iteration, line_search_result.step_size, cg_result)
            callback(current_loss, gradient_norm)
            converged = gradient_norm <= self._settings.gradient_norm_tolerance

        return _NewtonCGRawResult(
            final_point=current_point,
            num_iterations=iteration,
            num_cg_iterations=total_cg_iterations,
            converged=converged,
            final_gradient_norm=gradient_norm,
            line_search_failed=line_search_failed,
        )

    # ----------------------------------------------------------------------------------------------
    def _log_debug_iteration_detail(
        self, iteration: int, step_size: float, cg_result: TruncatedCGResult
    ) -> None:
        """Log the step size and the inner solve of one iteration, if a logger is attached."""
        if self._logger is not None:
            self._logger.debug(
                f"Iteration {iteration}: step size {step_size:.3e}, "
                f"{cg_result.num_iterations} CG iterations ({cg_result.reason.name.lower()}), "
                f"CG residual norm {cg_result.residual_norm:.3e}."
            )

    # ----------------------------------------------------------------------------------------------
    @override
    def _create_optimization_result(
        self, raw_result: _NewtonCGRawResult, history: OptimizationHistory
    ) -> OptimizationResult:
        """Map the raw iteration result and the recorded history onto `OptimizationResult`."""
        if raw_result.converged:
            status_message = (
                f"Converged: gradient norm {raw_result.final_gradient_norm:.3e} <= tolerance "
                f"{self._settings.gradient_norm_tolerance:.3e} "
                f"({raw_result.num_cg_iterations} CG iterations in total)."
            )
        elif raw_result.line_search_failed:
            status_message = (
                f"Line search failed after {raw_result.num_iterations} iterations (gradient norm "
                f"{raw_result.final_gradient_norm:.3e}); returning the last accepted iterate."
            )
        else:
            status_message = (
                f"Did not converge within {self._settings.maximum_num_iterations} iterations "
                f"({raw_result.num_cg_iterations} CG iterations in total)."
            )
        return OptimizationResult(
            result=raw_result.final_point,
            loss_history=history.loss_history,
            gradient_norm_history=history.gradient_norm_history,
            num_iterations=raw_result.num_iterations,
            success=raw_result.converged,
            status_message=status_message,
        )
