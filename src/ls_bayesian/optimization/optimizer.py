"""Template-method driver for gradient-based optimization backends.

Classes:
    OptimizationResult: Outcome of an optimization run, backend-agnostic.
    OptimizationHistory: Records loss and gradient-norm values as an optimization progresses.
    BaseOptimizer: ABC template-method driver for optimization backends.

Functions:
    log_header: Log the column header for iteration progress reports.
    log_iteration: Log one row of iteration progress.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, override

import numpy as np

from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.optimization.model import OptimizationModel

# Column widths for the progress table logged by `log_header`/`log_iteration`. Wide enough to fit
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
        loss_history (np.ndarray[tuple[int], np.dtype[np.float64]]): Loss value at every
            evaluation of the loss function, in call order.
        gradient_norm_history (np.ndarray[tuple[int], np.dtype[np.float64]]): Norm of the
            gradient at every evaluation of the gradient function, in call order, evaluated via
            the model's `evaluate_norm`.
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
    does not know when it is called from; callers (the optimization driver, via
    `_ModelWithRecords`) decide what to record and when. This mirrors
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
def log_header(logger: BaseLogger | None) -> None:
    """Log the column header for iteration-by-iteration progress reports.

    This function, together with `log_iteration`, is the only I/O performed by the `optimization`
    subpackage, kept separate from the actual optimization computation.

    Args:
        logger (BaseLogger | None): Logger to report to. No-op if `None`.
    """
    if logger is None:
        return
    header = (
        f"{'Iteration':>{_ITERATION_COLUMN_WIDTH}} "
        f"{'Time [s]':>{_TIME_COLUMN_WIDTH}} "
        f"{'Loss':>{_LOSS_COLUMN_WIDTH}} "
        f"{'Grad. norm':>{_GRADIENT_NORM_COLUMN_WIDTH}}"
    )
    logger.info(header)
    logger.info("-" * len(header))


# ==================================================================================================
def log_iteration(
    logger: BaseLogger | None,
    iteration: int,
    elapsed_time_seconds: float,
    loss: float,
    gradient_norm: float,
) -> None:
    """Log one row of iteration-by-iteration progress: iteration, elapsed time, loss, grad norm.

    Args:
        logger (BaseLogger | None): Logger to report to. No-op if `None`.
        iteration (int): Iteration number.
        elapsed_time_seconds (float): Elapsed time since the start of the optimization run, in
            seconds.
        loss (float): Loss value at this iteration.
        gradient_norm (float): Gradient norm at this iteration.
    """
    if logger is None:
        return
    row = (
        f"{iteration:>{_ITERATION_COLUMN_WIDTH}d} "
        f"{elapsed_time_seconds:>{_TIME_COLUMN_WIDTH}.3f} "
        f"{loss:>{_LOSS_COLUMN_WIDTH}.6e} "
        f"{gradient_norm:>{_GRADIENT_NORM_COLUMN_WIDTH}.6e}"
    )
    logger.info(row)


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
    [`OptimizationHistory`][ls_bayesian.optimization.optimizer.OptimizationHistory], and
    iteration-by-iteration reporting via `log_header`/`log_iteration` to an optional
    [`BaseLogger`][ls_bayesian.common.logging.BaseLogger]. `_run_impl` receives a
    `_ModelWithRecords`-wrapped model, so every `evaluate_cost`/`evaluate_gradient` call a
    backend makes on it is transparently recorded, with gradient norms always taken via the
    model's own `evaluate_norm`.

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
        model_with_records = _ModelWithRecords(model, history)

        start_time = time.monotonic()
        iteration_count = 0
        log_header(self._logger)

        def callback(*_: Any) -> None:
            nonlocal iteration_count
            iteration_count += 1
            elapsed_time_seconds = time.monotonic() - start_time
            latest_loss = history.loss_history[-1] if history.loss_history.size > 0 else np.nan
            latest_gradient_norm = (
                history.gradient_norm_history[-1]
                if history.gradient_norm_history.size > 0
                else np.nan
            )
            log_iteration(
                self._logger,
                iteration_count,
                elapsed_time_seconds,
                latest_loss,
                latest_gradient_norm,
            )

        raw_result = self._run_impl(initial_guess, model_with_records, callback)
        return self._create_optimization_result(raw_result, history)

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def _run_impl(
        self,
        initial_guess: np.ndarray[tuple[int], np.dtype[np.float64]],
        model: OptimizationModel,
        callback: Callable[..., None],
    ) -> Any:
        """Run the backend-specific minimization, returning its raw, backend-specific result.

        Args:
            initial_guess (np.ndarray[tuple[int], np.dtype[np.float64]]): Starting point.
            model (OptimizationModel): Objective to minimize; `evaluate_cost`/`evaluate_gradient`
                calls are recorded into the run's history.
            callback (Callable[..., None]): Callback to invoke once per iteration, with any
                arguments the backend passes it; records timing and logs progress.

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


# ==================================================================================================
class _ModelWithRecords(OptimizationModel):
    """Wraps a model so each `evaluate_cost`/`evaluate_gradient` call is recorded into a history.

    `evaluate_hessian_vector_product`, `evaluate_inner_product` and `evaluate_norm` are delegated
    to the wrapped model unchanged.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, model: OptimizationModel, history: OptimizationHistory) -> None:
        """Initialize the wrapper.

        Args:
            model (OptimizationModel): Model to wrap.
            history (OptimizationHistory): History to record evaluations into.
        """
        self._model = model
        self._history = history

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_cost(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        value = self._model.evaluate_cost(parameter_vector)
        self._history.record_loss(float(value))
        return value

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_gradient(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        gradient = self._model.evaluate_gradient(parameter_vector)
        self._history.record_gradient_norm(self._model.evaluate_norm(gradient))
        return gradient

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_hessian_vector_product(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._model.evaluate_hessian_vector_product(parameter_vector, direction_vector)

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_inner_product(
        self,
        first_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        second_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> float:
        return self._model.evaluate_inner_product(first_vector, second_vector)

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_norm(self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        return self._model.evaluate_norm(vector)
