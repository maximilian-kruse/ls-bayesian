r"""Step-size selection strategies for metric-generic optimization backends.

Classes:
    LineSearchResult: Outcome of a line search: the accepted step size and the loss already
        evaluated at the resulting point.
    LineSearch: ABC interface for step-size selection strategies.
    ArmijoBacktrackingLineSearchSettings: Settings for `ArmijoBacktrackingLineSearch`.
    ArmijoBacktrackingLineSearch: Backtracking line search on the Armijo sufficient-decrease
        condition.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Annotated, override

import numpy as np
from beartype.vale import Is

from ls_bayesian.common.logging import BaseLogger


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
        directional_derivative: float,
        loss_function: Callable[[np.ndarray[tuple[int], np.dtype[np.float64]]], float],
    ) -> LineSearchResult:
        r"""Find a step size $\tau$ along `search_direction` satisfying the strategy's acceptance
        condition.

        Args:
            current_point (np.ndarray[tuple[int], np.dtype[np.float64]]): Current iterate
                $\mathbf{m}_k$.
            search_direction (np.ndarray[tuple[int], np.dtype[np.float64]]): Search direction
                $\mathbf{p}_k$.
            current_loss (float): Loss at `current_point`, $I(\mathbf{m}_k)$.
            directional_derivative (float): Directional derivative
                $(\mathbf{g}_k, \mathbf{p}_k)$, precomputed by the caller in the inner product the
                search direction was computed in.
            loss_function (Callable[[np.ndarray[tuple[int], np.dtype[np.float64]]], float]):
                Callable evaluating the loss at a point.

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
        directional_derivative: float,
        loss_function: Callable[[np.ndarray[tuple[int], np.dtype[np.float64]]], float],
    ) -> LineSearchResult:
        r"""Find a step size satisfying the Armijo sufficient-decrease condition.

        Raises:
            RuntimeError: If no acceptable step size is found within
                `settings.max_backtracking_steps` backtracking steps.
        """
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
