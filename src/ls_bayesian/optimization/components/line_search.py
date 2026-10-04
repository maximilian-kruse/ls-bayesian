r"""Step-size selection strategies for metric-generic optimization backends.

Classes:
    LineSearchProblem: A line search's one-dimensional view of the objective along a search
        direction.
    LineSearchResult: Outcome of a line search: the accepted step size and the loss already
        evaluated at the resulting point.
    LineSearchStrategy: ABC interface for step-size selection strategies.
    ArmijoBacktrackingLineSearchSettings: Settings for `ArmijoBacktrackingLineSearch`.
    ArmijoBacktrackingLineSearch: Backtracking line search on the Armijo sufficient-decrease
        condition.
    StrongWolfeLineSearchSettings: Settings for `StrongWolfeLineSearch`.
    StrongWolfeLineSearch: Strong-Wolfe line search via bracketing and zoom.
    _BracketEndpoint: Step size with the loss and directional derivative there (internal).
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
class LineSearchProblem:
    r"""One-dimensional view of the objective along a search direction, as seen by a line search.

    Describes $\phi(\tau) = I(\mathbf{m}_k + \tau\mathbf{p}_k)$ and its derivative
    $\phi'(\tau) = (\nabla I(\mathbf{m}_k + \tau\mathbf{p}_k), \mathbf{p}_k)$, where
    $(\cdot,\cdot)$ is the inner product in which the search direction was computed, so that
    directional derivatives are always taken in that inner product.

    Attributes:
        current_point (np.ndarray[tuple[int], np.dtype[np.float64]]): Current iterate
            $\mathbf{m}_k$.
        search_direction (np.ndarray[tuple[int], np.dtype[np.float64]]): Search direction
            $\mathbf{p}_k$.
        current_loss (float): Loss at `current_point`, $\phi(0) = I(\mathbf{m}_k)$.
        current_gradient (np.ndarray[tuple[int], np.dtype[np.float64]]): Gradient
            $\mathbf{g}_k = \nabla I(\mathbf{m}_k)$ at `current_point`, so strategies need not
            re-evaluate it.
        loss_function (LossFunction): Callable evaluating the loss at a point.
        gradient_function (GradientFunction): Callable evaluating the gradient $\nabla I$ at a
            point.
        inner_product (InnerProductFunction): Inner product $(\cdot, \cdot)$ in which the
            search direction was computed.
    """

    current_point: np.ndarray[tuple[int], np.dtype[np.float64]]
    search_direction: np.ndarray[tuple[int], np.dtype[np.float64]]
    current_loss: float
    current_gradient: np.ndarray[tuple[int], np.dtype[np.float64]]
    loss_function: LossFunction
    gradient_function: GradientFunction
    inner_product: InnerProductFunction

    # ----------------------------------------------------------------------------------------------
    @property
    def directional_derivative(self) -> float:
        r"""Directional derivative $\phi'(0) = (\mathbf{g}_k, \mathbf{p}_k)$ at the current
        point; negative for a descent direction."""
        return self.inner_product(self.current_gradient, self.search_direction)

    # ----------------------------------------------------------------------------------------------
    def evaluate_loss(self, step_size: float) -> float:
        r"""Evaluate $\phi(\tau) = I(\mathbf{m}_k + \tau\mathbf{p}_k)$.

        Args:
            step_size (float): Step size $\tau$.

        Returns:
            float: Loss at the point reached with `step_size`.
        """
        return self.loss_function(self.current_point + step_size * self.search_direction)

    # ----------------------------------------------------------------------------------------------
    def evaluate_directional_derivative(self, step_size: float) -> float:
        r"""Evaluate $\phi'(\tau) = (\nabla I(\mathbf{m}_k + \tau\mathbf{p}_k), \mathbf{p}_k)$.

        Args:
            step_size (float): Step size $\tau$.

        Returns:
            float: Directional derivative at the point reached with `step_size`.
        """
        gradient = self.gradient_function(self.current_point + step_size * self.search_direction)
        return self.inner_product(gradient, self.search_direction)

    # ----------------------------------------------------------------------------------------------
    def create_failed_result(self) -> LineSearchResult:
        """Create the result for a search that found no acceptable step size: `success=False`,
        `step_size=nan` and the unchanged current loss."""
        return LineSearchResult(step_size=math.nan, loss=self.current_loss, success=False)


# ==================================================================================================
@dataclass(frozen=True)
class LineSearchResult:
    r"""Outcome of a line search.

    Bundles the accepted step size with the loss value the search already evaluated at the
    resulting point ($I(\mathbf{m}_k + \tau\mathbf{p}_k)$), so callers do not need to re-evaluate
    the (potentially expensive) loss function at that point themselves.

    If no acceptable step size was found, `success` is `False`, `step_size` is `nan` (no valid step
    exists, so it must not be used) and `loss` is the unchanged current loss. Usual causes
    are a search direction that is not a descent direction (e.g. from rounding error or an
    inaccurate gradient) or a loss that cannot decrease further in floating point.

    Attributes:
        step_size (float): Accepted step size $\tau$, or `nan` on failure.
        loss (float): Loss at $\mathbf{m}_k + \tau\mathbf{p}_k$ (the current loss on failure).
        success (bool): Whether an acceptable step size was found.
    """

    step_size: float
    loss: float
    success: bool


# ==================================================================================================
class LineSearchStrategy(ABC):
    """ABC interface for step-size selection strategies.

    Methods:
        find_step_size: Find a step size along a search direction satisfying the strategy's
            acceptance condition, or report that none was found.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def find_step_size(self, problem: LineSearchProblem) -> LineSearchResult:
        r"""Find a step size $\tau$ along `problem.search_direction` satisfying the strategy's
        acceptance condition.

        Args:
            problem (LineSearchProblem): The objective along the search direction, including the
                loss/gradient callables and the inner product in which directional derivatives
                are taken.

        Returns:
            LineSearchResult: Accepted step size and the loss already evaluated at the resulting
                point, or a result with `success=False` if no acceptable step size was found.
                Implementations never raise for this reason.
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
        max_backtracking_steps (int): Maximum number of backtracking steps before giving up.
            Defaults to `50`: with the defaults above, 50 backtracking steps shrink the step size
            to 2^-50 (~1e-15) before giving up, comfortably below double-precision step sizes that
            could still be meaningful. Beyond that, a further decrease cannot be represented
            reliably, indicating the search direction is likely not a descent direction (e.g. due
            to numerical error), so the search returns an unsuccessful `LineSearchResult` instead of
            looping indefinitely.
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
    The backtracking loop is capped at `settings.max_backtracking_steps`, returning an
    unsuccessful `LineSearchResult` instead of looping indefinitely if no acceptable step is
    found; this typically happens if `search_direction` is not a genuine descent direction, e.g.
    from numerical error, or if the loss cannot decrease further in floating point.
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
    def find_step_size(self, problem: LineSearchProblem) -> LineSearchResult:
        r"""Find a step size satisfying the Armijo sufficient-decrease condition.

        Only `problem.directional_derivative` and `problem.evaluate_loss` are used; the gradient
        is never evaluated at trial points.

        Returns:
            LineSearchResult: Accepted step, or a failure if no acceptable
                step size is found within `settings.max_backtracking_steps` backtracking steps.
        """
        directional_derivative = problem.directional_derivative
        step_size = self._settings.initial_step_size
        for backtracking_step in range(self._settings.max_backtracking_steps + 1):
            candidate_loss = problem.evaluate_loss(step_size)
            if candidate_loss <= problem.current_loss + (
                self._settings.sufficient_decrease_constant * step_size * directional_derivative
            ):
                return LineSearchResult(step_size=step_size, loss=candidate_loss, success=True)
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
        return problem.create_failed_result()


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
            iterations before giving up. Defaults to `20`: with `step_growth_factor=2.0`, 20
            iterations grow the step size to $2^{20}\approx10^6$ times `initial_step_size`,
            comfortably covering any well-scaled quasi-Newton step; beyond that, the search
            direction is likely not a genuine descent direction.
        max_zoom_iterations (int): Maximum number of zoom-phase (Algorithm 3.6) iterations before
            giving up. Defaults to `30`, a generous cap for cubic-safeguarded zoom to shrink a
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
@dataclass(frozen=True)
class _BracketEndpoint:
    r"""Step size $\tau$ with the loss $\phi(\tau)$ and directional derivative $\phi'(\tau)$
    there: one endpoint of a strong-Wolfe bracket."""

    step_size: float
    loss: float
    directional_derivative: float


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
    `settings.max_zoom_iterations`, each returning an unsuccessful
    `LineSearchResult` instead of looping indefinitely if no acceptable step is found; this
    typically happens if `search_direction` is not a genuine descent direction, e.g. from
    numerical error. A non-descent direction
    ($\phi'(0) \geq 0$) is reported the same way, before any trial point is evaluated.

    `nan`/`inf` loss or derivative values at a trial point are not special-cased: every
    acceptance/rejection comparison below evaluates to `False` for `nan` operands (a Python/NumPy
    guarantee), which correctly routes such a trial into the same "shrink and retry" path as an
    ordinary rejection.
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
    def find_step_size(self, problem: LineSearchProblem) -> LineSearchResult:
        r"""Find a step size satisfying the strong Wolfe conditions (Algorithm 3.5).

        Returns:
            LineSearchResult: Accepted step, or a failure if the directional
                derivative $\phi'(0)$ is non-negative (not a descent direction), or if no
                acceptable step size is found within `settings.max_bracketing_iterations`
                bracketing iterations or `settings.max_zoom_iterations` zoom iterations.
        """
        directional_derivative = problem.directional_derivative
        if directional_derivative >= 0.0:
            message = (
                "Strong-Wolfe line search received a non-descent search direction (directional "
                f"derivative {directional_derivative:.3e} >= 0)."
            )
            self._log_warning(message)
            return problem.create_failed_result()

        settings = self._settings
        previous = _BracketEndpoint(0.0, problem.current_loss, directional_derivative)
        step_size = settings.initial_step_size

        for bracketing_iteration in range(settings.max_bracketing_iterations):
            trial = _BracketEndpoint(
                step_size,
                problem.evaluate_loss(step_size),
                problem.evaluate_directional_derivative(step_size),
            )
            sufficient_decrease_holds = trial.loss <= problem.current_loss + (
                settings.sufficient_decrease_constant * step_size * directional_derivative
            )

            if not sufficient_decrease_holds or (
                bracketing_iteration > 0 and trial.loss >= previous.loss
            ):
                self._log_debug_bracketing_rejection(bracketing_iteration, step_size, trial.loss)
                return self._zoom(problem, endpoint_lo=previous, endpoint_hi=trial)

            if abs(trial.directional_derivative) <= settings.curvature_constant * abs(
                directional_derivative
            ):
                return LineSearchResult(step_size=step_size, loss=trial.loss, success=True)

            if trial.directional_derivative >= 0.0:
                self._log_debug_bracketing_rejection(bracketing_iteration, step_size, trial.loss)
                return self._zoom(problem, endpoint_lo=trial, endpoint_hi=previous)

            previous = trial
            step_size *= settings.step_growth_factor

        message = (
            "Strong-Wolfe line search's bracketing phase did not find an acceptable step size "
            f"or a valid bracket within {settings.max_bracketing_iterations} iterations. The "
            "search direction may not be a descent direction."
        )
        self._log_warning(message)
        return problem.create_failed_result()

    # ----------------------------------------------------------------------------------------------
    def _zoom(
        self,
        problem: LineSearchProblem,
        endpoint_lo: _BracketEndpoint,
        endpoint_hi: _BracketEndpoint,
    ) -> LineSearchResult:
        r"""Zoom phase: shrink the bracket between `endpoint_lo` and `endpoint_hi`, known to
        contain a strong-Wolfe point, via safeguarded cubic interpolation, until one is found.

        `endpoint_lo` is always the best sufficient-decrease point found so far (not necessarily
        the smaller step size); `endpoint_hi` is the other bracket endpoint. Both endpoints' loss
        and directional derivative are always fully known on entry (see the class docstring).

        Returns:
            LineSearchResult: Accepted step, or a failure if the bracket collapses to zero width,
                or `settings.max_zoom_iterations` is exhausted, without finding an acceptable
                step size.
        """
        settings = self._settings
        directional_derivative = problem.directional_derivative

        for zoom_iteration in range(settings.max_zoom_iterations):
            if endpoint_lo.step_size == endpoint_hi.step_size:
                message = (
                    "Strong-Wolfe line search's zoom phase collapsed to a zero-width bracket "
                    f"after {zoom_iteration} iterations without finding an acceptable step size."
                )
                self._log_warning(message)
                return problem.create_failed_result()

            trial_step_size = self._interpolate(endpoint_lo, endpoint_hi)
            trial = _BracketEndpoint(
                trial_step_size,
                problem.evaluate_loss(trial_step_size),
                problem.evaluate_directional_derivative(trial_step_size),
            )

            sufficient_decrease_holds = trial.loss <= problem.current_loss + (
                settings.sufficient_decrease_constant * trial_step_size * directional_derivative
            )
            if not sufficient_decrease_holds or trial.loss >= endpoint_lo.loss:
                self._log_debug_zoom_rejection(zoom_iteration, trial_step_size, trial.loss)
                endpoint_hi = trial
                continue

            if abs(trial.directional_derivative) <= settings.curvature_constant * abs(
                directional_derivative
            ):
                return LineSearchResult(step_size=trial_step_size, loss=trial.loss, success=True)

            if (
                trial.directional_derivative * (endpoint_hi.step_size - endpoint_lo.step_size)
                >= 0.0
            ):
                endpoint_hi = endpoint_lo
            endpoint_lo = trial
            self._log_debug_zoom_rejection(zoom_iteration, trial_step_size, trial.loss)

        message = (
            "Strong-Wolfe line search's zoom phase did not find an acceptable step size within "
            f"{settings.max_zoom_iterations} iterations."
        )
        self._log_warning(message)
        return problem.create_failed_result()

    # ----------------------------------------------------------------------------------------------
    def _interpolate(self, endpoint_lo: _BracketEndpoint, endpoint_hi: _BracketEndpoint) -> float:
        r"""Safeguarded cubic interpolation of a trial step between `endpoint_lo` and
        `endpoint_hi` (in whichever order), falling back to bisection if the cubic estimate is
        degenerate or too close to either endpoint."""
        interval_low = min(endpoint_lo.step_size, endpoint_hi.step_size)
        interval_high = max(endpoint_lo.step_size, endpoint_hi.step_size)
        margin = self._settings.interpolation_safeguard_fraction * (interval_high - interval_low)
        bisection_step_size = 0.5 * (interval_low + interval_high)

        alpha_lo, phi_lo, derphi_lo = (
            endpoint_lo.step_size,
            endpoint_lo.loss,
            endpoint_lo.directional_derivative,
        )
        alpha_hi, phi_hi, derphi_hi = (
            endpoint_hi.step_size,
            endpoint_hi.loss,
            endpoint_hi.directional_derivative,
        )

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
