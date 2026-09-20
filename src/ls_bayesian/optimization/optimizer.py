"""Template-method driver for gradient-based optimization backends.

Classes:
    OptimizationResult: Outcome of an optimization run, backend-agnostic.
    OptimizationHistory: Records loss and gradient-norm values as an optimization progresses.
    BaseOptimizer: ABC template-method driver for optimization backends.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.optimization.model import OptimizationModel

# Invoked once per accepted iteration as callback(loss, gradient_norm); `BaseOptimizer.run` binds
# it to a closure that records both values into the run's `OptimizationHistory` and logs the
# iteration. `_run_impl` implementations must pass values already computed as part of the
# iteration, not re-evaluate the model for this purpose.
type IterationCallback = Callable[[float, float], None]

# Column widths for the progress table logged by `_log_header`/`_log_iteration`. Wide enough to fit
# the longest header ("Grad. norm") and typical numeric values in scientific notation (e.g.
# "1.234568e+00") with one space of padding.
_ITERATION_COLUMN_WIDTH = 10
_TIME_COLUMN_WIDTH = 12
_LOSS_COLUMN_WIDTH = 14
_GRADIENT_NORM_COLUMN_WIDTH = 14


# ==================================================================================================
@dataclass
class OptimizationResult:
    """Outcome of an optimization run.

    Attributes:
        result (np.ndarray[tuple[int], np.dtype[np.float64]]): Minimizer found, or the best
            iterate if the run did not converge.
        loss_history (np.ndarray[tuple[int], np.dtype[np.float64]]): Loss value at every accepted
            iterate, in iteration order (one value per iteration, not per internal model
            evaluation, e.g. line-search trials are not included).
        gradient_norm_history (np.ndarray[tuple[int], np.dtype[np.float64]]): Norm of the
            gradient at every accepted iterate, in iteration order, evaluated via the model's
            `evaluate_norm`.
        num_iterations (int): Number of iterations performed by the backend.
        success (bool): Whether the backend reports convergence.
        status_message (str): Human-readable termination message from the backend.
    """

    result: np.ndarray[tuple[int], np.dtype[np.float64]]
    loss_history: np.ndarray[tuple[int], np.dtype[np.float64]]
    gradient_norm_history: np.ndarray[tuple[int], np.dtype[np.float64]]
    num_iterations: int
    success: bool
    status_message: str


# ==================================================================================================
class OptimizationHistory:
    """Records loss and gradient-norm values evaluated during an optimization run.

    Values are appended in the order they are recorded. The class does not compute anything and
    does not know when it is called from; callers (the optimization driver, via the
    per-iteration `IterationCallback`) decide what to record and when. This mirrors
    [`EvaluationCache`][ls_bayesian.posterior.cache.EvaluationCache]'s design: recording is an
    explicit, single-purpose side effect isolated in its own component, rather than a hidden
    mutation of the driver's own state.

    Methods:
        record_loss: Append a loss value.
        record_gradient_norm: Append a gradient-norm value.

    Attributes:
        loss_history (np.ndarray[tuple[int], np.dtype[np.float64]]): Recorded loss values, in call
            order.
        gradient_norm_history (np.ndarray[tuple[int], np.dtype[np.float64]]): Recorded
            gradient-norm values, in call order.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        """Initialize an empty history."""
        self._loss_values: list[float] = []
        self._gradient_norm_values: list[float] = []

    # ----------------------------------------------------------------------------------------------
    def record_loss(self, value: float) -> None:
        """Append a loss value.

        Args:
            value (float): Loss value to record.
        """
        self._loss_values.append(value)

    # ----------------------------------------------------------------------------------------------
    def record_gradient_norm(self, value: float) -> None:
        """Append a gradient-norm value.

        Args:
            value (float): Gradient-norm value to record.
        """
        self._gradient_norm_values.append(value)

    # ----------------------------------------------------------------------------------------------
    @property
    def loss_history(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Return the recorded loss values, in call order, as a new array."""
        return np.array(self._loss_values, dtype=np.float64)

    # ----------------------------------------------------------------------------------------------
    @property
    def gradient_norm_history(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Return the recorded gradient-norm values, in call order, as a new array."""
        return np.array(self._gradient_norm_values, dtype=np.float64)


# ==================================================================================================
class BaseOptimizer(ABC):
    """ABC template-method driver for gradient-based optimization backends.

    Subclasses implement [`_run_impl`][ls_bayesian.optimization.optimizer.BaseOptimizer._run_impl]
    (the actual minimization call) and
    [`_create_optimization_result`][ls_bayesian.optimization.optimizer.BaseOptimizer._create_optimization_result]
    (mapping the backend's raw result onto
    [`OptimizationResult`][ls_bayesian.optimization.optimizer.OptimizationResult]). This base
    class owns everything that is common across backends: input validation, progress
    instrumentation via
    [`OptimizationHistory`][ls_bayesian.optimization.optimizer.OptimizationHistory],
    iteration-by-iteration reporting via `_log_header`/`_log_iteration`, and final-outcome
    reporting via `_log_final_outcome` (`info` on convergence, `warning` otherwise), all to an
    optional [`BaseLogger`][ls_bayesian.common.logging.BaseLogger]. `_run_impl` receives the model
    unwrapped: evaluating `model.evaluate_cost`/`evaluate_gradient` never records anything.
    Instead, `_run_impl` must call the `IterationCallback` it is given exactly once per accepted
    iteration, with the loss and gradient norm at the new iterate (typically values it already
    computed as part of the iteration, not obtained by evaluating the model again). `run` binds
    that callback to a closure which records both values into the history and logs the
    iteration.

    Methods:
        run: Run the optimizer from an initial guess.

    Attributes:
        requires_hessian (bool): Whether this backend requires a model with Hessian-vector
            products. Informational only: `OptimizationModel.evaluate_hessian_vector_product` is
            always callable, and raises `NotImplementedError` itself if a model does not support
            it, so subclasses needing it do not have to check this flag before calling it.
            Subclasses override the default. Defaults to `False`.
    """

    requires_hessian: bool = False

    # ----------------------------------------------------------------------------------------------
    def __init__(self, logger: BaseLogger | None = None) -> None:
        """Initialize the driver.

        Args:
            logger (BaseLogger | None, optional): Logger for iteration-by-iteration progress
                reports. Nothing is logged if `None`. The caller owns the logger's lifetime
                (construction, closing); the optimizer never constructs its own logger, so the
                same logger can be shared with other components, e.g. a
                [`LogPosterior`][ls_bayesian.posterior.posterior.LogPosterior]. Defaults to `None`.
        """
        self._logger = logger

    # ----------------------------------------------------------------------------------------------
    def run(
        self,
        initial_guess: np.ndarray[tuple[int], np.dtype[np.float64]],
        model: OptimizationModel,
    ) -> OptimizationResult:
        """Run the optimizer from an initial guess.

        Args:
            initial_guess (np.ndarray[tuple[int], np.dtype[np.float64]]): Starting point,
                one-dimensional.
            model (OptimizationModel): Objective to minimize, together with the inner-product
                space its gradient/Hessian-vector product are expressed in.

        Raises:
            ValueError: If `initial_guess` is not one-dimensional or contains non-finite values.

        Returns:
            OptimizationResult: Outcome of the optimization run.
        """
        initial_guess = np.asarray(initial_guess, dtype=np.float64)
        self._validate_initial_guess(initial_guess)

        history = OptimizationHistory()
        start_time = time.monotonic()
        iteration_count = 0
        self._log_header()

        def callback(loss: float, gradient_norm: float) -> None:
            nonlocal iteration_count
            iteration_count += 1
            history.record_loss(loss)
            history.record_gradient_norm(gradient_norm)
            elapsed_time_seconds = time.monotonic() - start_time
            self._log_iteration(iteration_count, elapsed_time_seconds, loss, gradient_norm)

        raw_result = self._run_impl(initial_guess, model, callback)
        result = self._create_optimization_result(raw_result, history)
        self._log_final_outcome(result)
        return result

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def _run_impl(
        self,
        initial_guess: np.ndarray[tuple[int], np.dtype[np.float64]],
        model: OptimizationModel,
        callback: IterationCallback,
    ) -> Any:
        """Run the backend-specific minimization, returning its raw, backend-specific result.

        Args:
            initial_guess (np.ndarray[tuple[int], np.dtype[np.float64]]): Starting point.
            model (OptimizationModel): Objective to minimize. Evaluating
                `evaluate_cost`/`evaluate_gradient` never records anything by itself.
            callback (IterationCallback): Callback to invoke exactly once per accepted iteration,
                as `callback(loss, gradient_norm)`; records both values into the run's history and
                logs progress. Implementations must pass values already computed as part of the
                iteration rather than evaluating the model again for this purpose.

        Returns:
            Any: Backend-specific raw result, consumed by `_create_optimization_result`.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def _create_optimization_result(
        self, raw_result: Any, history: OptimizationHistory
    ) -> OptimizationResult:
        """Map the backend's raw result and the recorded history onto `OptimizationResult`.

        Args:
            raw_result (Any): Backend-specific raw result, as returned by `_run_impl`.
            history (OptimizationHistory): History recorded during the run.

        Returns:
            OptimizationResult: Outcome of the optimization run.
        """

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _validate_initial_guess(
        initial_guess: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Raise if `initial_guess` is not one-dimensional and finite."""
        if initial_guess.ndim != 1:
            raise ValueError(
                f"initial_guess must be one-dimensional, got shape {initial_guess.shape}."
            )
        if not np.all(np.isfinite(initial_guess)):
            raise ValueError("initial_guess must contain only finite values.")

    # ----------------------------------------------------------------------------------------------
    def _log_header(self) -> None:
        """Log the column header for iteration-by-iteration progress reports.

        This method, together with `_log_iteration` and `_log_final_outcome`, is the only I/O
        performed by the `optimization` subpackage, kept separate from the actual optimization
        computation. No-op if `self._logger` is `None`.
        """
        if self._logger is None:
            return
        header = (
            f"{'Iteration':>{_ITERATION_COLUMN_WIDTH}} "
            f"{'Time [s]':>{_TIME_COLUMN_WIDTH}} "
            f"{'Loss':>{_LOSS_COLUMN_WIDTH}} "
            f"{'Grad. norm':>{_GRADIENT_NORM_COLUMN_WIDTH}}"
        )
        self._logger.info(header)
        self._logger.info("-" * len(header))

    # ----------------------------------------------------------------------------------------------
    def _log_iteration(
        self,
        iteration: int,
        elapsed_time_seconds: float,
        loss: float,
        gradient_norm: float,
    ) -> None:
        """Log one row of iteration-by-iteration progress: iteration, elapsed time, loss, grad
        norm. No-op if `self._logger` is `None`.

        Args:
            iteration (int): Iteration number.
            elapsed_time_seconds (float): Elapsed time since the start of the optimization run, in
                seconds.
            loss (float): Loss value at this iteration.
            gradient_norm (float): Gradient norm at this iteration.
        """
        if self._logger is None:
            return
        row = (
            f"{iteration:>{_ITERATION_COLUMN_WIDTH}d} "
            f"{elapsed_time_seconds:>{_TIME_COLUMN_WIDTH}.3f} "
            f"{loss:>{_LOSS_COLUMN_WIDTH}.6e} "
            f"{gradient_norm:>{_GRADIENT_NORM_COLUMN_WIDTH}.6e}"
        )
        self._logger.info(row)

    # ----------------------------------------------------------------------------------------------
    def _log_final_outcome(self, result: OptimizationResult) -> None:
        """Log the final convergence status and termination message of a completed run. No-op if
        `self._logger` is `None`.

        Args:
            result (OptimizationResult): Outcome of the run.
        """
        if self._logger is None:
            return
        outcome = "Converged" if result.success else "Did not converge"
        message = f"{outcome} after {result.num_iterations} iterations: {result.status_message}"
        if result.success:
            self._logger.info(message)
        else:
            self._logger.warning(message)
