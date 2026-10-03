r"""Step-size selection strategies for metric-generic optimization backends.

Classes:
    LineSearchResult: Outcome of a line search: the accepted step size and the loss already
        evaluated at the resulting point.
    LineSearch: ABC interface for step-size selection strategies.
    ArmijoBacktrackingLineSearchSettings: Settings for `ArmijoBacktrackingLineSearch`.
    ArmijoBacktrackingLineSearch: Backtracking line search on the Armijo sufficient-decrease
        condition.
    StrongWolfeLineSearchSettings: Settings for `StrongWolfeLineSearch`.
    StrongWolfeLineSearch: Strong-Wolfe line search via bracketing and zoom.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Annotated, override

import numpy as np
from beartype.vale import Is

from ls_bayesian.common.logging import BaseLogger

type LossFunction = Callable[[np.ndarray[tuple[int], np.dtype[np.float64]]], float]
type GradientFunction = Callable[
    [np.ndarray[tuple[int], np.dtype[np.float64]]], np.ndarray[tuple[int], np.dtype[np.float64]]
]
type InnerProductFunction = Callable[
    [np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]],
    float,
]


# ==================================================================================================
@dataclass(frozen=True)
class LineSearchResult:
    r"""Outcome of a line search.

    Bundles the accepted step size with the loss value the search already evaluated at the
    resulting point ($I(\mathbf{m}_k + \tau\mathbf{p}_k)$), so callers do not need to re-evaluate
    the (potentially expensive) loss function at that point themselves.

    Attributes:
        step_size (float): Accepted step size $\tau$.
        loss (float): Loss at $\mathbf{m}_k + \tau\mathbf{p}_k$.
    """

    step_size: float
    loss: float


# ==================================================================================================
class LineSearchStrategy(ABC):
    """ABC interface for step-size selection strategies.

    Methods:
        find_step_size: Find a step size along a search direction satisfying the strategy's
            acceptance condition.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def find_step_size(
        self,
        current_point: np.ndarray[tuple[int], np.dtype[np.float64]],
        search_direction: np.ndarray[tuple[int], np.dtype[np.float64]],
        current_loss: float,
        current_gradient: np.ndarray[tuple[int], np.dtype[np.float64]],
        loss_function: LossFunction,
        gradient_function: GradientFunction,
        inner_product: InnerProductFunction,
    ) -> LineSearchResult:
        r"""Find a step size $\tau$ along `search_direction` satisfying the strategy's acceptance
        condition.

        Args:
            current_point (np.ndarray[tuple[int], np.dtype[np.float64]]): Current iterate
                $\mathbf{m}_k$.
            search_direction (np.ndarray[tuple[int], np.dtype[np.float64]]): Search direction
                $\mathbf{p}_k$.
            current_loss (float): Loss at `current_point`, $I(\mathbf{m}_k)$.
            current_gradient (np.ndarray[tuple[int], np.dtype[np.float64]]): Gradient
                $\mathbf{g}_k = \nabla I(\mathbf{m}_k)$ at `current_point`, so the strategy
                need not re-evaluate it.
            loss_function (LossFunction): Callable evaluating the loss at a point.
            gradient_function (GradientFunction): Callable evaluating the gradient
                $\nabla I$ at a point.
            inner_product (InnerProductFunction): Inner product $(\cdot, \cdot)$ in which the
                search direction was computed. The strategy evaluates the directional derivatives
                $\phi'(\tau) = (\nabla I(\mathbf{m}_k + \tau\mathbf{p}_k), \mathbf{p}_k)$ itself
                with it, so they are always taken in this inner product. $\phi'(0) =
                (\mathbf{g}_k, \mathbf{p}_k)$ follows from `current_gradient`.

        Returns:
            LineSearchResult: Accepted step size and the loss already evaluated at the resulting
                point.
        """


# ==================================================================================================
@dataclass
class ArmijoBacktrackingLineSearchSettings:
    r"""Settings for `ArmijoBacktrackingLineSearch`.

    The field constraints are validated on initialization.

    Attributes:
        initial_step_size (Real): Initial step size $\tau^{(0)}$. Defaults to `1.0`, the standard
            initialization for quasi-Newton line searches: the quasi-Newton direction itself
            already incorporates curvature information, so a full step is usually accepted or
            nearly so.
        sufficient_decrease_constant (Real): Armijo constant $c_1 \in (0, 1)$. Defaults to
            `1e-4`, the standard Armijo sufficient-decrease constant (Nocedal & Wright, "Numerical
            Optimization", 2006, Sec. 3.1): a small value close to 0 accepts almost any decrease,
            which is typical practice for quasi-Newton methods (as opposed to steepest descent,
            which needs a stricter tolerance).
        backtracking_factor (Real): Backtracking divisor $\beta > 1$;
            $\tau^{(i+1)} = \tau^{(i)} / \beta$. Defaults to `2.0`, halving the step on each
            backtrack, the standard, simplest choice (Nocedal & Wright, "Numerical Optimization",
            2006, Sec. 3.1).
        max_backtracking_steps (int): Maximum number of backtracking steps before raising.
            Defaults to `50`: with the defaults above, 50 backtracking steps shrink the step size
            to 2^-50 (~1e-15) before giving up, comfortably below double-precision step sizes that
            could still be meaningful. Beyond that, a further decrease cannot be represented
            reliably, indicating the search direction is likely not a descent direction (e.g. due
            to numerical error), so the search raises instead of looping indefinitely.
    """

    initial_step_size: Annotated[Real, Is[lambda x: x > 0]] = 1.0
    sufficient_decrease_constant: Annotated[Real, Is[lambda x: 0 < x < 1]] = 1e-4
    backtracking_factor: Annotated[Real, Is[lambda x: x > 1]] = 2.0
    max_backtracking_steps: Annotated[int, Is[lambda x: x > 0]] = 50


# ==================================================================================================
class ArmijoBacktrackingLineSearch(LineSearchStrategy):
    r"""Backtracking line search on the Armijo sufficient-decrease condition.

    For step size $\tau$, accepts the first $\tau^{(i)} = \tau^{(0)} / \beta^i$ satisfying
    $I(\mathbf{m}_k + \tau\mathbf{p}_k) \leq I(\mathbf{m}_k) + c_1\tau(\mathbf{g}_k,\mathbf{p}_k)$.
    The backtracking loop is capped at `settings.max_backtracking_steps`, raising instead of
    looping indefinitely if no acceptable step is found; this can only happen if
    `search_direction` is not a genuine descent direction, e.g. from numerical error.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, settings: ArmijoBacktrackingLineSearchSettings, logger: BaseLogger | None = None
    ) -> None:
        """Initialize the line search.

        Args:
            settings (ArmijoBacktrackingLineSearchSettings): Settings for the line search.
            logger (BaseLogger | None, optional): Logger for rejected trial steps (debug level)
                and for the search-direction warning issued if no acceptable step is found
                (warning level). Defaults to `None`.
        """
        self._settings = settings
        self._logger = logger

    # ----------------------------------------------------------------------------------------------
    @override
    def find_step_size(
        self,
        current_point: np.ndarray[tuple[int], np.dtype[np.float64]],
        search_direction: np.ndarray[tuple[int], np.dtype[np.float64]],
        current_loss: float,
        current_gradient: np.ndarray[tuple[int], np.dtype[np.float64]],
        loss_function: LossFunction,
        gradient_function: GradientFunction,
        inner_product: InnerProductFunction,
    ) -> LineSearchResult:
        r"""Find a step size satisfying the Armijo sufficient-decrease condition.

        `gradient_function` is accepted (to satisfy `LineSearchStrategy`'s shared interface) but
        never called: the Armijo condition only needs $\phi'(0)$, computed here from
        `current_gradient`, and the loss at trial points.

        Raises:
            RuntimeError: If no acceptable step size is found within
                `settings.max_backtracking_steps` backtracking steps.
        """
        directional_derivative = inner_product(current_gradient, search_direction)
        step_size = self._settings.initial_step_size
        for backtracking_step in range(self._settings.max_backtracking_steps + 1):
            candidate_loss = loss_function(current_point + step_size * search_direction)
            if candidate_loss <= current_loss + (
                self._settings.sufficient_decrease_constant * step_size * directional_derivative
            ):
                return LineSearchResult(step_size=step_size, loss=candidate_loss)
            if self._logger is not None:
                self._logger.debug(
                    f"Backtracking step {backtracking_step}: step size {step_size:.3e} rejected "
                    f"(loss {candidate_loss:.6e})."
                )
            step_size /= self._settings.backtracking_factor

        message = (
            "Armijo backtracking line search did not find an acceptable step size within "
            f"{self._settings.max_backtracking_steps} steps. The search direction may not be a "
            "descent direction."
        )
        if self._logger is not None:
            self._logger.warning(message)
        raise RuntimeError(message)


# ==================================================================================================
@dataclass
class StrongWolfeLineSearchSettings:
    r"""Settings for `StrongWolfeLineSearch`.

    The field constraints are validated on initialization.

    Attributes:
        initial_step_size (Real): Initial trial step size $\tau^{(1)}$. Defaults to `1.0`, for the
            same reason as `ArmijoBacktrackingLineSearchSettings.initial_step_size`: the
            quasi-Newton direction already incorporates curvature information, so a full step is
            usually near-optimal.
        sufficient_decrease_constant (Real): Armijo constant $c_1 \in (0, 1)$. Defaults to `1e-4`,
            identical to `ArmijoBacktrackingLineSearchSettings.sufficient_decrease_constant`.
        curvature_constant (Real): Strong-Wolfe curvature constant $c_2 \in (c_1, 1)$. Defaults to
            `0.9`, the standard choice for (quasi-)Newton methods: a value this close to 1 accepts
            almost any step with a sufficiently reduced gradient magnitude, suiting methods whose
            search direction already carries good curvature information -- as opposed to `0.1`,
            recommended for nonlinear conjugate-gradient methods, which need a tighter curvature
            condition.
        step_growth_factor (Real): Extrapolation multiplier $\sigma > 1$ used while bracketing,
            $\tau^{(i+1)} = \sigma\tau^{(i)}$. Defaults to `2.0`, mirroring
            `ArmijoBacktrackingLineSearchSettings.backtracking_factor`'s doubling/halving choice
            but in the growing, not shrinking, direction.
        interpolation_safeguard_fraction (Real): Minimum fractional distance $\delta \in (0, 0.5)$
            a zoom-phase cubic-interpolated trial step must keep from either bracket endpoint
            before falling back to bisection. Defaults to `0.1`, a standard safeguard fraction
            preventing interpolation from stalling progress by repeatedly proposing points
            arbitrarily close to an already-rejected endpoint.
        max_bracketing_iterations (int): Maximum number of bracketing-phase
            iterations before raising. Defaults to `20`: with `step_growth_factor=2.0`, 20
            iterations grow the step size to $2^{20}\approx10^6$ times `initial_step_size`,
            comfortably covering any well-scaled quasi-Newton step; beyond that, the search
            direction is likely not a genuine descent direction.
        max_zoom_iterations (int): Maximum number of zoom-phase (Algorithm 3.6) iterations before
            raising. Defaults to `30`, a generous cap for cubic-safeguarded zoom to shrink a
            bracket to double-precision resolution while still catching genuine failures early.
    """

    initial_step_size: Annotated[Real, Is[lambda x: x > 0]] = 1.0
    sufficient_decrease_constant: Annotated[Real, Is[lambda x: 0 < x < 1]] = 1e-4
    curvature_constant: Annotated[Real, Is[lambda x: 0 < x < 1]] = 0.9
    step_growth_factor: Annotated[Real, Is[lambda x: x > 1]] = 2.0
    interpolation_safeguard_fraction: Annotated[Real, Is[lambda x: 0 < x < 0.5]] = 0.1
    max_bracketing_iterations: Annotated[int, Is[lambda x: x > 0]] = 20
    max_zoom_iterations: Annotated[int, Is[lambda x: x > 0]] = 30

    def __post_init__(self) -> None:
        """Check `curvature_constant > sufficient_decrease_constant`"""
        if self.curvature_constant <= self.sufficient_decrease_constant:
            raise ValueError(
                f"curvature_constant ({self.curvature_constant}) must be strictly greater than "
                f"sufficient_decrease_constant ({self.sufficient_decrease_constant})."
            )


# ==================================================================================================
class StrongWolfeLineSearch(LineSearchStrategy):
    r"""Strong-Wolfe line search via bracketing and zoom (Nocedal & Wright, "Numerical
    Optimization").

    Finds a step size $\tau$ satisfying both the Armijo sufficient-decrease condition and the
    strong curvature condition $|\phi'(\tau)| \leq c_2|\phi'(0)|$, where $\phi(\tau) =
    I(\mathbf{m}_k+\tau\mathbf{p}_k)$. Unlike `ArmijoBacktrackingLineSearch`, this evaluates
    the directional derivative (i.e. the gradient) at every trial point, not only the loss
    -- roughly doubling the per-trial cost of this search relative to Armijo's, in exchange for a
    structural guarantee that the accepted step's gradient has genuinely reduced curvature-wise,
    not merely produced a lower loss value. This is what prevents
    `CustomLBFGSOptimizer._two_loop_recursion`'s Barzilai-Borwein-style seed scaling from ever
    seeing a correction pair with a pathologically small $\|\mathbf{y}\|^2$ relative to
    $(\mathbf{s},\mathbf{y})$: `ArmijoBacktrackingLineSearch` (sufficient decrease only, no
    curvature condition) can accept exactly such a step.

    Both phases eagerly evaluate the directional derivative at every new bracket/zoom endpoint,
    even on iterations where the algorithm would not strictly require it, so that the zoom
    phase's cubic interpolation always has $\phi$ and $\phi'$ at both of its endpoints available --
    a deliberate simplification trading a modest number of extra gradient evaluations for a
    substantially simpler, easier-to-verify implementation than tracking which endpoint's
    derivative is or isn't already known.

    The bracketing phase is capped at `settings.max_bracketing_iterations` and the zoom phase at
    `settings.max_zoom_iterations`, each raising instead of looping indefinitely if no acceptable
    step is found; this can only happen if `search_direction` is not a genuine descent direction,
    e.g. from numerical error. `nan`/`inf` loss or derivative values at a trial point are not
    special-cased: every acceptance/rejection comparison below evaluates to `False` for `nan`
    operands (a Python/NumPy guarantee), which correctly routes such a trial into the same
    "shrink and retry" path as an ordinary rejection.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, settings: StrongWolfeLineSearchSettings, logger: BaseLogger | None = None
    ) -> None:
        """Initialize the line search.

        Args:
            settings (StrongWolfeLineSearchSettings): Settings for the line search.
            logger (BaseLogger | None, optional): Logger for rejected trial steps in both the
                bracketing and zoom phases (debug level) and for the search-direction/budget
                warnings issued if no acceptable step is found (warning level). Defaults to
                `None`.
        """
        self._settings = settings
        self._logger = logger

    # ----------------------------------------------------------------------------------------------
    @override
    def find_step_size(
        self,
        current_point: np.ndarray[tuple[int], np.dtype[np.float64]],
        search_direction: np.ndarray[tuple[int], np.dtype[np.float64]],
        current_loss: float,
        current_gradient: np.ndarray[tuple[int], np.dtype[np.float64]],
        loss_function: LossFunction,
        gradient_function: GradientFunction,
        inner_product: InnerProductFunction,
    ) -> LineSearchResult:
        r"""Find a step size satisfying the strong Wolfe conditions (Algorithm 3.5).

        Raises:
            RuntimeError: If the directional derivative $\phi'(0)$ is non-negative (not a descent
                direction), or if no acceptable step size is found within
                `settings.max_bracketing_iterations` bracketing iterations or
                `settings.max_zoom_iterations` zoom iterations.
        """
        directional_derivative = inner_product(current_gradient, search_direction)
        if directional_derivative >= 0.0:
            message = (
                "Strong-Wolfe line search received a non-descent search direction (directional "
                f"derivative {directional_derivative:.3e} >= 0)."
            )
            self._log_warning(message)
            raise RuntimeError(message)

        settings = self._settings
        previous_step_size = 0.0
        previous_loss = current_loss
        previous_directional_derivative = directional_derivative
        step_size = settings.initial_step_size

        for bracketing_iteration in range(settings.max_bracketing_iterations):
            trial_point = current_point + step_size * search_direction
            trial_loss = loss_function(trial_point)
            sufficient_decrease_holds = trial_loss <= current_loss + (
                settings.sufficient_decrease_constant * step_size * directional_derivative
            )

            if not sufficient_decrease_holds or (
                bracketing_iteration > 0 and trial_loss >= previous_loss
            ):
                self._log_debug_bracketing_rejection(bracketing_iteration, step_size, trial_loss)
                trial_directional_derivative = inner_product(
                    gradient_function(trial_point), search_direction
                )
                return self._zoom(
                    current_point,
                    search_direction,
                    current_loss,
                    directional_derivative,
                    loss_function,
                    gradient_function,
                    inner_product,
                    alpha_lo=previous_step_size,
                    phi_lo=previous_loss,
                    derphi_lo=previous_directional_derivative,
                    alpha_hi=step_size,
                    phi_hi=trial_loss,
                    derphi_hi=trial_directional_derivative,
                )

            trial_directional_derivative = inner_product(
                gradient_function(trial_point), search_direction
            )
            if abs(trial_directional_derivative) <= settings.curvature_constant * abs(
                directional_derivative
            ):
                return LineSearchResult(step_size=step_size, loss=trial_loss)

            if trial_directional_derivative >= 0.0:
                self._log_debug_bracketing_rejection(bracketing_iteration, step_size, trial_loss)
                return self._zoom(
                    current_point,
                    search_direction,
                    current_loss,
                    directional_derivative,
                    loss_function,
                    gradient_function,
                    inner_product,
                    alpha_lo=step_size,
                    phi_lo=trial_loss,
                    derphi_lo=trial_directional_derivative,
                    alpha_hi=previous_step_size,
                    phi_hi=previous_loss,
                    derphi_hi=previous_directional_derivative,
                )

            previous_step_size = step_size
            previous_loss = trial_loss
            previous_directional_derivative = trial_directional_derivative
            step_size *= settings.step_growth_factor

        message = (
            "Strong-Wolfe line search's bracketing phase did not find an acceptable step size "
            f"or a valid bracket within {settings.max_bracketing_iterations} iterations. The "
            "search direction may not be a descent direction."
        )
        self._log_warning(message)
        raise RuntimeError(message)

    # ----------------------------------------------------------------------------------------------
    def _zoom(
        self,
        current_point: np.ndarray[tuple[int], np.dtype[np.float64]],
        search_direction: np.ndarray[tuple[int], np.dtype[np.float64]],
        current_loss: float,
        directional_derivative: float,
        loss_function: LossFunction,
        gradient_function: GradientFunction,
        inner_product: InnerProductFunction,
        alpha_lo: float,
        phi_lo: float,
        derphi_lo: float,
        alpha_hi: float,
        phi_hi: float,
        derphi_hi: float,
    ) -> LineSearchResult:
        r"""Zoom phase: shrink $[\tau_{\text{lo}}, \tau_{\text{hi}}]$, known to
        contain a strong-Wolfe point, via safeguarded cubic interpolation, until one is found.

        `alpha_lo`/`phi_lo`/`derphi_lo` is always the best sufficient-decrease point found so far;
        `alpha_hi`/`phi_hi`/`derphi_hi` is the other bracket endpoint. Both endpoints' loss and
        directional derivative are always fully known on entry (see the class docstring).

        Raises:
            RuntimeError: If the bracket collapses to zero width, or `settings.max_zoom_iterations`
                is exhausted, without finding an acceptable step size.
        """
        settings = self._settings

        for zoom_iteration in range(settings.max_zoom_iterations):
            interval_low = min(alpha_lo, alpha_hi)
            interval_high = max(alpha_lo, alpha_hi)
            if interval_high - interval_low <= 0.0:
                message = (
                    "Strong-Wolfe line search's zoom phase collapsed to a zero-width bracket "
                    f"after {zoom_iteration} iterations without finding an acceptable step size."
                )
                self._log_warning(message)
                raise RuntimeError(message)

            trial_step_size = self._interpolate(
                alpha_lo,
                phi_lo,
                derphi_lo,
                alpha_hi,
                phi_hi,
                derphi_hi,
                interval_low,
                interval_high,
            )
            trial_point = current_point + trial_step_size * search_direction
            trial_loss = loss_function(trial_point)
            trial_directional_derivative = inner_product(
                gradient_function(trial_point), search_direction
            )

            sufficient_decrease_holds = trial_loss <= current_loss + (
                settings.sufficient_decrease_constant * trial_step_size * directional_derivative
            )
            if not sufficient_decrease_holds or trial_loss >= phi_lo:
                self._log_debug_zoom_rejection(zoom_iteration, trial_step_size, trial_loss)
                alpha_hi = trial_step_size
                phi_hi = trial_loss
                derphi_hi = trial_directional_derivative
                continue

            if abs(trial_directional_derivative) <= settings.curvature_constant * abs(
                directional_derivative
            ):
                return LineSearchResult(step_size=trial_step_size, loss=trial_loss)

            if trial_directional_derivative * (alpha_hi - alpha_lo) >= 0.0:
                alpha_hi, phi_hi, derphi_hi = alpha_lo, phi_lo, derphi_lo
            alpha_lo = trial_step_size
            phi_lo = trial_loss
            derphi_lo = trial_directional_derivative
            self._log_debug_zoom_rejection(zoom_iteration, trial_step_size, trial_loss)

        message = (
            "Strong-Wolfe line search's zoom phase did not find an acceptable step size within "
            f"{settings.max_zoom_iterations} iterations."
        )
        self._log_warning(message)
        raise RuntimeError(message)

    # ----------------------------------------------------------------------------------------------
    def _interpolate(
        self,
        alpha_lo: float,
        phi_lo: float,
        derphi_lo: float,
        alpha_hi: float,
        phi_hi: float,
        derphi_hi: float,
        interval_low: float,
        interval_high: float,
    ) -> float:
        r"""Safeguarded cubic interpolation of a trial step within
        $[\tau_{\text{lo}}, \tau_{\text{hi}}]$ (in whichever order), falling back to
        bisection if the cubic estimate is degenerate or too close to either endpoint."""
        margin = self._settings.interpolation_safeguard_fraction * (interval_high - interval_low)
        bisection_step_size = 0.5 * (interval_low + interval_high)

        d1 = derphi_lo + derphi_hi - 3.0 * (phi_lo - phi_hi) / (alpha_lo - alpha_hi)
        discriminant = d1**2 - derphi_lo * derphi_hi
        if discriminant < 0.0:
            return bisection_step_size
        d2 = math.copysign(math.sqrt(discriminant), alpha_hi - alpha_lo)
        denominator = derphi_hi - derphi_lo + 2.0 * d2
        if denominator == 0.0:
            return bisection_step_size

        trial_step_size = alpha_hi - (alpha_hi - alpha_lo) * (derphi_hi + d2 - d1) / denominator
        if trial_step_size < interval_low + margin or trial_step_size > interval_high - margin:
            return bisection_step_size
        return trial_step_size

    # ----------------------------------------------------------------------------------------------
    def _log_debug_bracketing_rejection(
        self, iteration: int, step_size: float, loss: float
    ) -> None:
        """Log a rejected bracketing-phase trial step, if a logger is attached."""
        if self._logger is not None:
            self._logger.debug(
                f"Bracketing iteration {iteration}: step size {step_size:.3e} rejected "
                f"(loss {loss:.6e})."
            )

    # ----------------------------------------------------------------------------------------------
    def _log_debug_zoom_rejection(self, iteration: int, step_size: float, loss: float) -> None:
        """Log a rejected zoom-phase trial step, if a logger is attached."""
        if self._logger is not None:
            self._logger.debug(
                f"Zoom iteration {iteration}: step size {step_size:.3e} rejected (loss {loss:.6e})."
            )

    # ----------------------------------------------------------------------------------------------
    def _log_warning(self, message: str) -> None:
        """Log a warning, if a logger is attached."""
        if self._logger is not None:
            self._logger.warning(message)
