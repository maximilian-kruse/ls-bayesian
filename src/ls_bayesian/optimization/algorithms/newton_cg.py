r"""Truncated Newton (Newton-CG) in an arbitrary inner-product space.

Classes:
    TrustRegionSettings: Trust-region globalization of `NewtonCGOptimizer`.
    NewtonCGSettings: Settings for `NewtonCGOptimizer`.
    NewtonCGOptimizer: Inexact Newton method with a truncated CG inner solve and a line search.
"""

import math
from collections.abc import Callable
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
    TruncationReason,
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
class TrustRegionSettings:
    r"""Trust-region globalization of `NewtonCGOptimizer`, in place of a line search.

    The inner CG solve is restricted to $\|\mathbf{p}\| \le \Delta$ (Steihaug-Toint). With the
    quadratic model's predicted reduction, the ratio of actual to predicted reduction $\rho$
    decides the step and the radius (Nocedal & Wright, "Numerical Optimization", 2006,
    Algorithm 4.1; the thresholds are those of hIPPYlib's `ReducedSpaceNewtonCG`, Villa, Petra &
    Ghattas, 2021): the step is taken if $\rho$ exceeds `acceptance_threshold`, the radius is
    multiplied by `shrink_factor` if $\rho$ is below `shrink_threshold`, and by `expand_factor`
    if $\rho$ is above `expand_threshold` and the CG iteration ended at the boundary.

    Attributes:
        initial_radius (Real | None): Initial radius $\Delta_0$ in the optimizer's norm. `None`
            (default) solves the first system without a radius and starts from the norm of that
            step.
        acceptance_threshold (Real): Smallest $\rho$ for which a step is taken. Defaults to
            `0.05`, as in hIPPYlib.
        shrink_threshold (Real): The radius shrinks below this $\rho$. Defaults to `0.25`.
        expand_threshold (Real): The radius grows above this $\rho$, if the step reached the
            boundary. Defaults to `0.75`.
        shrink_factor (Real): Factor the radius is multiplied with when it shrinks. Defaults to
            `0.5`.
        expand_factor (Real): Factor the radius is multiplied with when it grows. Defaults to
            `2.0`.
        maximum_num_consecutive_rejections (int): The run stops after this many rejected steps in
            a row, as the radius has then shrunk by `shrink_factor` to this power. Defaults to
            `20`, a factor of about 1e-6 with the default `shrink_factor`.
    """

    initial_radius: Annotated[Real, Is[lambda x: x > 0]] | None = None
    acceptance_threshold: Annotated[Real, Is[lambda x: 0 <= x < 0.25]] = 0.05
    shrink_threshold: Annotated[Real, Is[lambda x: 0 < x < 1]] = 0.25
    expand_threshold: Annotated[Real, Is[lambda x: 0 < x < 1]] = 0.75
    shrink_factor: Annotated[Real, Is[lambda x: 0 < x < 1]] = 0.5
    expand_factor: Annotated[Real, Is[lambda x: x > 1]] = 2.0
    maximum_num_consecutive_rejections: Annotated[int, Is[lambda x: x > 0]] = 20

    def __post_init__(self) -> None:
        if self.acceptance_threshold > self.shrink_threshold:
            raise ValueError(
                f"acceptance_threshold ({self.acceptance_threshold}) must not exceed "
                f"shrink_threshold ({self.shrink_threshold})."
            )
        if self.shrink_threshold >= self.expand_threshold:
            raise ValueError(
                f"shrink_threshold ({self.shrink_threshold}) must be smaller than "
                f"expand_threshold ({self.expand_threshold})."
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
            forcing sequence
            $\eta_k = \min\{\eta_\mathrm{max}, \sqrt{\|\mathbf{g}_k\| / \|\mathbf{g}_0\|}\}$
            gives superlinear convergence (Nocedal & Wright, "Numerical Optimization", 2006,
            Sec. 7.1). The gradient norm is taken relative to the one at the start of the run, so
            the sequence does not depend on the scale of the objective (as in hIPPYlib, Villa,
            Petra & Ghattas, 2021); with the absolute norm the cap $\eta_\mathrm{max}$ would
            apply for the whole run whenever $\|\mathbf{g}\|$ is large.
        minimum_forcing_term (Real): Lower bound of the relative residual tolerance. Defaults to
            `1e-6`: below this, CG would chase the noise of a Hessian-vector product that is
            evaluated with a single-precision forward model.
        cg_settings (TruncatedCGSettings): Settings of the inner CG solve.
        minimum_predicted_reduction (Real | None): Early stopping: the run stops, and counts as
            converged, when the quadratic model predicts a decrease of the loss smaller than this
            for the (inexact) Newton step $\mathbf{p}$, namely
            $-(\mathbf{g}, \mathbf{p}) - \frac{1}{2}(\mathbf{p}, \mathbf{H}\mathbf{p})$. It is
            meant for losses whose evaluation noise (e.g. a
            single-precision forward model) is larger than the decrease that is still worth
            having: below the noise, steps are accepted or rejected by chance and further
            iterations only cost Hessian-vector products. The predicted decrease can exceed the
            attainable one (a Gauss-Newton model may underestimate the curvature), so the value
            should be calibrated on the debug log of a run. With a trust region, a step that ended
            at the boundary does not trigger the stop, as its predicted decrease reflects the
            radius. Defaults to `None`: no early stopping.
        trust_region (TrustRegionSettings | None): Use a trust region instead of a line search to
            globalize the iteration. Defaults to `None`: the optimizer then needs a line search.
    """

    maximum_num_iterations: Annotated[int, Is[lambda x: x > 0]] = 100
    gradient_norm_tolerance: Annotated[Real, Is[lambda x: x > 0]] = 1e-6
    maximum_forcing_term: Annotated[Real, Is[lambda x: 0 < x < 1]] = 0.5
    minimum_forcing_term: Annotated[Real, Is[lambda x: 0 < x < 1]] = 1e-6
    cg_settings: TruncatedCGSettings = field(default_factory=TruncatedCGSettings)
    minimum_predicted_reduction: Annotated[Real, Is[lambda x: x > 0]] | None = None
    trust_region: TrustRegionSettings | None = None

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
    globalization_failed: bool = False
    stopped_early: bool = False
    final_predicted_reduction: float = math.nan


# ==================================================================================================
class NewtonCGOptimizer(BaseOptimizer):
    r"""Truncated Newton method (Newton-CG) in an arbitrary inner-product space.

    In every iteration the Newton system $\mathbf{H}_k\mathbf{p} = -\mathbf{g}_k$ is solved
    approximately by
    [`TruncatedConjugateGradient`][ls_bayesian.optimization.components.truncated_cg.TruncatedConjugateGradient]
    with the relative tolerance
    $\eta_k = \min\{\eta_\mathrm{max}, \sqrt{\|\mathbf{g}_k\| / \|\mathbf{g}_0\|}\}$ (clipped from
    below), using only Hessian-vector products from
    `model.evaluate_hessian_vector_product`. The iteration is globalized either by an injected
    line search along the CG direction, or, with `settings.trust_region`, by a trust region
    (Steihaug-Toint CG with a ratio test, see `TrustRegionSettings`). Either way the loss itself
    decides which steps are taken, so the Hessian may be a Gauss-Newton approximation. If the CG
    direction is not a descent direction in floating point, the steepest-descent direction
    $-\mathbf{g}_k$ is used instead.

    **`model.evaluate_gradient` and `model.evaluate_hessian_vector_product` must return Riesz
    representers under `model.evaluate_inner_product`**, as for
    [`CustomLBFGSOptimizer`][ls_bayesian.optimization.algorithms.custom_lbfgs.CustomLBFGSOptimizer].
    In a Cameron-Martin space the prior is the identity part of the Hessian, which is what makes
    CG converge in few iterations: it only has to resolve the data-dominated directions.

    On a line-search failure, or when `settings.trust_region.maximum_num_consecutive_rejections`
    steps in a row are rejected, the run terminates gracefully with the last accepted iterate and
    `success=False`. A rejected trust-region step is not reported to the iteration callback.

    Attributes:
        requires_hessian (bool): Always `True`.
    """

    requires_hessian: bool = True

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        settings: NewtonCGSettings,
        line_search: LineSearchStrategy | None = None,
        logger: BaseLogger | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            settings (NewtonCGSettings): Settings for the optimizer.
            line_search (LineSearchStrategy | None, optional): Step-size selection strategy.
                Required, and only allowed, if `settings.trust_region` is `None`. Defaults to
                `None`.
            logger (BaseLogger | None, optional): Logger for iteration-by-iteration progress
                reports. Defaults to `None`.

        Raises:
            ValueError: If the globalization is not defined by exactly one of `line_search` and
                `settings.trust_region`.
        """
        super().__init__(logger=logger)
        if (line_search is None) == (settings.trust_region is None):
            raise ValueError(
                "Exactly one of line_search and settings.trust_region must be given to select "
                "the globalization."
            )
        self._settings = settings
        self._line_search = line_search
        self._conjugate_gradient = TruncatedConjugateGradient(settings.cg_settings)

    # ----------------------------------------------------------------------------------------------
    def _forcing_term(self, gradient_norm: float, initial_gradient_norm: float) -> float:
        r"""Relative residual tolerance $\eta_k$ of the inner solve.

        Args:
            gradient_norm (float): Norm of the current gradient, $\|\mathbf{g}_k\|$.
            initial_gradient_norm (float): Norm of the gradient at the start of the run,
                $\|\mathbf{g}_0\|$.
        """
        relative_gradient_norm = gradient_norm / initial_gradient_norm
        forcing_term = min(
            float(self._settings.maximum_forcing_term), math.sqrt(relative_gradient_norm)
        )
        return max(forcing_term, float(self._settings.minimum_forcing_term))

    # ----------------------------------------------------------------------------------------------
    def _updated_radius(self, radius: float, ratio: float, reached_boundary: bool) -> float:
        r"""Radius of the next trust region.

        Args:
            radius (float): Radius $\Delta_k$ of the step just tried.
            ratio (float): Ratio $\rho$ of actual to predicted reduction of that step.
            reached_boundary (bool): Whether the CG iteration of the step ended at the boundary.
        """
        settings = self._settings.trust_region
        assert settings is not None
        if ratio < settings.shrink_threshold:
            return radius * float(settings.shrink_factor)
        if ratio > settings.expand_threshold and reached_boundary:
            return radius * float(settings.expand_factor)
        return radius

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _predicted_reduction(
        gradient: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction: np.ndarray[tuple[int], np.dtype[np.float64]],
        residual: np.ndarray[tuple[int], np.dtype[np.float64]] | None,
        hessian_action: Callable[
            [np.ndarray[tuple[int], np.dtype[np.float64]]],
            np.ndarray[tuple[int], np.dtype[np.float64]],
        ],
        inner_product: Callable[
            [
                np.ndarray[tuple[int], np.dtype[np.float64]],
                np.ndarray[tuple[int], np.dtype[np.float64]],
            ],
            float,
        ],
    ) -> float:
        r"""Decrease $-\{(\mathbf{g}, \mathbf{p}) + \frac{1}{2}(\mathbf{p}, \mathbf{H}\mathbf{p})\}$
        of the quadratic model for the step $\mathbf{p}$.

        With the CG residual $\mathbf{r} = \mathbf{H}\mathbf{p} + \mathbf{g}$ this needs no
        Hessian-vector product; without it (steepest-descent fallback) it needs one.
        """
        if residual is None:
            model_value = inner_product(gradient, direction) + 0.5 * inner_product(
                direction, hessian_action(direction)
            )
        else:
            model_value = 0.5 * (
                inner_product(gradient, direction) + inner_product(direction, residual)
            )
        return -model_value

    # ----------------------------------------------------------------------------------------------
    @override
    def _run_impl(
        self,
        initial_guess: np.ndarray[tuple[int], np.dtype[np.float64]],
        model: OptimizationModel,
        callback: IterationCallback,
    ) -> _NewtonCGRawResult:
        """Run the outer iteration: truncated CG for the direction, then a line search or a trust
        region ratio test for the step, until convergence, `maximum_num_iterations`, or a failure
        of the globalization. In the last case the last accepted iterate is returned and the
        failed iteration is not reported to `callback`.
        """
        loss_function = model.evaluate_cost
        gradient_function = model.evaluate_gradient
        inner_product = model.evaluate_inner_product
        trust_region = self._settings.trust_region

        current_point = initial_guess.copy()
        current_loss = loss_function(current_point)
        current_gradient = gradient_function(current_point)
        gradient_norm = model.evaluate_norm(current_gradient)
        initial_gradient_norm = gradient_norm
        radius = None if trust_region is None else trust_region.initial_radius
        num_rejections = 0
        iteration = 0
        total_cg_iterations = 0
        converged = gradient_norm <= self._settings.gradient_norm_tolerance
        globalization_failed = False
        stopped_early = False
        predicted_reduction = math.nan
        minimum_predicted_reduction = self._settings.minimum_predicted_reduction

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
                self._forcing_term(gradient_norm, initial_gradient_norm),
                None if radius is None else float(radius),
            )
            total_cg_iterations += cg_result.num_iterations
            search_direction = cg_result.direction
            residual = cg_result.residual
            if inner_product(current_gradient, search_direction) >= 0.0:
                search_direction = -current_gradient
                residual = None

            if trust_region is not None or minimum_predicted_reduction is not None:
                predicted_reduction = self._predicted_reduction(
                    current_gradient, search_direction, residual, hessian_action, inner_product
                )
            if (
                minimum_predicted_reduction is not None
                and predicted_reduction < minimum_predicted_reduction
                and cg_result.reason is not TruncationReason.TRUST_REGION_BOUNDARY
            ):
                stopped_early = True
                break

            if trust_region is None:
                assert self._line_search is not None
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
                    globalization_failed = True
                    break
                step_size = line_search_result.step_size
                next_point = current_point + step_size * search_direction
                next_loss = line_search_result.loss
                detail = f"step size {step_size:.3e}, predicted decrease {predicted_reduction:.3e}"
            else:
                next_point = current_point + search_direction
                next_loss = loss_function(next_point)
                ratio = (
                    (current_loss - next_loss) / predicted_reduction
                    if predicted_reduction > 0.0
                    else -math.inf
                )
                step_norm = model.evaluate_norm(search_direction)
                if radius is None:
                    radius = step_norm
                radius = self._updated_radius(
                    radius, ratio, cg_result.reason is TruncationReason.TRUST_REGION_BOUNDARY
                )
                accepted = ratio > trust_region.acceptance_threshold
                detail = (
                    f"step norm {step_norm:.3e}, predicted decrease {predicted_reduction:.3e}, "
                    f"ratio {ratio:.3e}, new radius {radius:.3e}, "
                    f"step {'accepted' if accepted else 'rejected'}"
                )
                if not accepted:
                    num_rejections += 1
                    self._log_debug_iteration_detail(iteration + 1, detail, cg_result)
                    if num_rejections >= trust_region.maximum_num_consecutive_rejections:
                        globalization_failed = True
                        break
                    continue
                num_rejections = 0

            current_point, current_loss = next_point, next_loss
            current_gradient = gradient_function(current_point)
            gradient_norm = model.evaluate_norm(current_gradient)
            iteration += 1
            self._log_debug_iteration_detail(iteration, detail, cg_result)
            callback(current_loss, gradient_norm)
            converged = gradient_norm <= self._settings.gradient_norm_tolerance

        return _NewtonCGRawResult(
            final_point=current_point,
            num_iterations=iteration,
            num_cg_iterations=total_cg_iterations,
            converged=converged,
            final_gradient_norm=gradient_norm,
            globalization_failed=globalization_failed,
            stopped_early=stopped_early,
            final_predicted_reduction=predicted_reduction,
        )

    # ----------------------------------------------------------------------------------------------
    def _log_debug_iteration_detail(
        self, iteration: int, detail: str, cg_result: TruncatedCGResult
    ) -> None:
        """Log the globalization detail and the inner solve of one iteration, if a logger is
        attached."""
        if self._logger is not None:
            self._logger.debug(
                f"Iteration {iteration}: {detail}, "
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
        elif raw_result.stopped_early:
            status_message = (
                f"Stopped early: the predicted decrease {raw_result.final_predicted_reduction:.3e} "
                f"is below {self._settings.minimum_predicted_reduction:.3e} (gradient norm "
                f"{raw_result.final_gradient_norm:.3e}, {raw_result.num_cg_iterations} CG "
                "iterations in total)."
            )
        elif raw_result.globalization_failed:
            failure = "Line search" if self._settings.trust_region is None else "Trust region"
            status_message = (
                f"{failure} failed after {raw_result.num_iterations} iterations (gradient norm "
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
            success=raw_result.converged or raw_result.stopped_early,
            status_message=status_message,
        )
