r"""L-BFGS with cautious updating in an arbitrary inner-product space.

Classes:
    CorrectionPair: One correction pair $(s_i, y_i, \rho_i)$ of the limited-memory Hessian
        approximation.
    CorrectionPairStore: Bounded FIFO memory of the last $M$ correction pairs.
    CustomLBFGSSettings: Settings for `CustomLBFGSOptimizer`.
    CustomLBFGSOptimizer: L-BFGS with cautious updating, generalized to an arbitrary inner
        product.
"""

from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from numbers import Real
from typing import Annotated, override

import numpy as np
from beartype.vale import Is

from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.optimization.components.cautious_update import CorrectionPairAcceptanceStrategy
from ls_bayesian.optimization.components.line_search import LineSearchStrategy
from ls_bayesian.optimization.model import OptimizationModel
from ls_bayesian.optimization.optimizer import (
    BaseOptimizer,
    IterationCallback,
    OptimizationHistory,
    OptimizationResult,
)

# Initial operator to be used as action of Hessian approximation in double loop recursion
type SeedOperator = Callable[
    [np.ndarray[tuple[int], np.dtype[np.float64]]], np.ndarray[tuple[int], np.dtype[np.float64]]
]


# ==================================================================================================
@dataclass(frozen=True)
class CorrectionPair:
    r"""One correction pair $(s_i, y_i, \rho_i)$ of the limited-memory Hessian approximation.

    Attributes:
        state_difference (np.ndarray[tuple[int], np.dtype[np.float64]]): Iterate displacement
            $s_i = m_{i+1} - m_i$.
        gradient_difference (np.ndarray[tuple[int], np.dtype[np.float64]]): Gradient displacement
            $y_i = \nabla I(m_{i+1}) - \nabla I(m_i)$.
        inner_product_reciprocal (float): $\rho_i = 1 / (s_i, y_i)$, computed once at insertion.
    """

    state_difference: np.ndarray[tuple[int], np.dtype[np.float64]]
    gradient_difference: np.ndarray[tuple[int], np.dtype[np.float64]]
    inner_product_reciprocal: float


# ==================================================================================================
class CorrectionPairStore:
    """Bounded FIFO memory of the last $M$ correction pairs.

    Backed by a `collections.deque(maxlen=memory_size)`, which evicts the oldest pair
    automatically once full, so no explicit eviction logic is needed here.

    Methods:
        add: Add a new correction pair, evicting the oldest one if the store is full.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, memory_size: int) -> None:
        """Initialize an empty store.

        Args:
            memory_size (int): Maximum number of correction pairs to keep, $M$.
        """
        self._pairs: deque[CorrectionPair] = deque(maxlen=memory_size)

    # ----------------------------------------------------------------------------------------------
    def add(
        self,
        state_difference: np.ndarray[tuple[int], np.dtype[np.float64]],
        gradient_difference: np.ndarray[tuple[int], np.dtype[np.float64]],
        inner_product_reciprocal: float,
    ) -> None:
        r"""Add a new correction pair, evicting the oldest one if the store is full.

        Args:
            state_difference (np.ndarray[tuple[int], np.dtype[np.float64]]): Iterate displacement
                $s_i$.
            gradient_difference (np.ndarray[tuple[int], np.dtype[np.float64]]): Gradient
                displacement $y_i$.
            inner_product_reciprocal (float): $\rho_i = 1 / (s_i, y_i)$.
        """
        self._pairs.append(
            CorrectionPair(
                state_difference=state_difference,
                gradient_difference=gradient_difference,
                inner_product_reciprocal=inner_product_reciprocal,
            )
        )

    # ----------------------------------------------------------------------------------------------
    def __iter__(self) -> Iterator[CorrectionPair]:
        """Iterate correction pairs from oldest to newest."""
        return iter(self._pairs)

    # ----------------------------------------------------------------------------------------------
    def __reversed__(self) -> Iterator[CorrectionPair]:
        """Iterate correction pairs from newest to oldest."""
        return reversed(self._pairs)

    # ----------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the number of stored correction pairs."""
        return len(self._pairs)


# ==================================================================================================
@dataclass
class CustomLBFGSSettings:
    r"""Settings for `CustomLBFGSOptimizer`.

    The field constraints are validated on initialization.

    Attributes:
        memory_size (int): Number of correction pairs to keep, $M$. Defaults to `10`, a widely
            used, moderate choice within the standard range of 3 to 20 (Nocedal & Wright,
            "Numerical Optimization", 2006, Sec. 7.2).
        maximum_num_iterations (int): Maximum number of outer iterations. Defaults to `1000`, a
            generic safeguard against non-termination; not tied to any specific problem scale.
        gradient_norm_tolerance (Real): Convergence tolerance on the gradient norm, evaluated in
            the optimizer's inner-product space. Defaults to `1e-6`, a typical convergence
            tolerance for double-precision optimization.
    """

    memory_size: Annotated[int, Is[lambda x: x > 0]] = 10
    maximum_num_iterations: Annotated[int, Is[lambda x: x > 0]] = 1000
    gradient_norm_tolerance: Annotated[Real, Is[lambda x: x > 0]] = 1e-6


# ==================================================================================================
@dataclass(frozen=True)
class _CustomLBFGSRawResult:
    """Raw result of `CustomLBFGSOptimizer._run_impl`, mapped onto `OptimizationResult` by
    `_create_optimization_result`."""

    final_point: np.ndarray[tuple[int], np.dtype[np.float64]]
    num_iterations: int
    converged: bool
    final_gradient_norm: float


# ==================================================================================================
class CustomLBFGSOptimizer(BaseOptimizer):
    r"""L-BFGS with cautious updating, generalized to an arbitrary inner-product space.

    Combines the two-loop recursion, an Armijo backtracking line search, and cautious
    correction-pair updating, generalized from a Cameron-Martin space to whatever inner product
    the [`OptimizationModel`][ls_bayesian.optimization.model.OptimizationModel] passed to `run`
    implements via `evaluate_inner_product`/`evaluate_norm`. `optimization` never depends on where
    that inner product comes from (e.g. a Cameron-Martin inner product induced by a Bayesian
    prior's covariance, built in another subpackage).

    **`model.evaluate_gradient` must return the Riesz representer of the objective's derivative
    under `model.evaluate_inner_product`** -- e.g. the Cameron-Martin representer, not an
    L2/Euclidean discretized gradient. Converting between representers (e.g. applying a prior
    covariance operator to a raw discretized gradient) is the caller's responsibility, performed
    outside this class, when implementing the `OptimizationModel`.

    The injected [`LineSearch`][ls_bayesian.optimization.components.line_search.LineSearch] caps
    the number of backtracking steps and raises if none succeeds, since an unbounded loop could
    hang if `search_direction` is not a genuine descent direction due to numerical error. This
    class adds `maximum_num_iterations` and `gradient_norm_tolerance` (Sec.
    `CustomLBFGSSettings`) as explicit stopping criteria.

    Attributes:
        requires_hessian (bool): Always `False`; this method does not use Hessian information.
    """

    requires_hessian: bool = False

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        settings: CustomLBFGSSettings,
        line_search: LineSearchStrategy,
        acceptance_strategy: CorrectionPairAcceptanceStrategy,
        seed_operator: SeedOperator = lambda vector: vector,
        logger: BaseLogger | None = None,
    ) -> None:
        r"""Initialize the optimizer.

        Args:
            settings (CustomLBFGSSettings): Settings for the optimizer.
            line_search (LineSearch): Step-size selection strategy.
            acceptance_strategy (CorrectionPairAcceptanceStrategy): Correction-pair acceptance
                policy.
            seed_operator (SeedOperator, optional): Initial inverse-Hessian approximation
                $\mathbf{H}_k^0$ for the two-loop recursion. Defaults to the identity,
                $\mathbf{H}_k^0 = \mathbf{I}$: in a Cameron-Martin space induced by a Gaussian
                prior's covariance, the prior term's Hessian with respect to the Cameron-Martin
                inner product is exactly the identity, making this the natural "structured seed
                matrix" choice (Mannel & Rund 2024, Petra & Ghattas 2019).
            logger (BaseLogger | None, optional): Logger for iteration-by-iteration progress
                reports. Defaults to `None`.
        """
        super().__init__(logger=logger)
        self._settings = settings
        self._line_search = line_search
        self._acceptance_strategy = acceptance_strategy
        self._seed_operator = seed_operator

    # ----------------------------------------------------------------------------------------------
    def _two_loop_recursion(
        self,
        gradient: np.ndarray[tuple[int], np.dtype[np.float64]],
        correction_pairs: CorrectionPairStore,
        model: OptimizationModel,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""L-BFGS two-loop recursion, computing a search direction without forming the inverse
        Hessian approximation $\mathbf{H}_k$ explicitly.

        Implements the two-loop recursion in the form of Nocedal & Wright, "Numerical
        Optimization" (2006), Algorithm 7.4, generalized to an arbitrary inner product, via
        `model.evaluate_inner_product`. With no stored correction pairs and the identity seed
        operator, this reduces exactly to steepest descent, $\mathbf{p}_k = -\mathbf{g}_k$.

        Args:
            gradient (np.ndarray[tuple[int], np.dtype[np.float64]]): Current gradient
                $\mathbf{g}_k$, represented in `model`'s inner product.
            correction_pairs (CorrectionPairStore): Stored correction pairs
                $\{(\mathbf{s}_i, \mathbf{y}_i, \rho_i)\}$.
            model (OptimizationModel): Model whose `evaluate_inner_product` the recursion is
                carried out in.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Search direction
                $\mathbf{p}_k = -\mathbf{r}$.
        """
        inner_product = model.evaluate_inner_product
        q = gradient.copy()
        etas: list[float] = []
        for pair in reversed(correction_pairs):
            eta = pair.inner_product_reciprocal * inner_product(pair.state_difference, q)
            q = q - eta * pair.gradient_difference
            etas.append(eta)

        r = self._seed_operator(q)
        for pair, eta in zip(correction_pairs, reversed(etas), strict=True):
            zeta = pair.inner_product_reciprocal * inner_product(pair.gradient_difference, r)
            r = r + pair.state_difference * (eta - zeta)

        return -r

    # ----------------------------------------------------------------------------------------------
    @override
    def _run_impl(
        self,
        initial_guess: np.ndarray[tuple[int], np.dtype[np.float64]],
        model: OptimizationModel,
        callback: IterationCallback,
    ) -> _CustomLBFGSRawResult:
        """Run the outer iteration: two-loop recursion for the direction, line search for the
        step, cautious updating for whether to store the new correction pair, until convergence or
        `maximum_num_iterations` is reached.

        Raises:
            ValueError: If the acceptance strategy accepts a correction pair with non-positive
                curvature $(s, y) \\leq 0$, which would make the correction pair's reciprocal
                $\\rho$ ill-defined or sign-flip the limited-memory inverse-Hessian approximation.
                `CautiousUpdateStrategy` guards against this; a permissive strategy such as
                `AlwaysAcceptStrategy` does not.
        """
        loss_function = model.evaluate_cost
        gradient_function = model.evaluate_gradient
        inner_product = model.evaluate_inner_product
        correction_pairs = CorrectionPairStore(self._settings.memory_size)

        current_point = initial_guess.copy()
        current_loss = loss_function(current_point)
        current_gradient = gradient_function(current_point)
        gradient_norm = model.evaluate_norm(current_gradient)
        iteration = 0
        converged = gradient_norm <= self._settings.gradient_norm_tolerance

        while not converged and iteration < self._settings.maximum_num_iterations:
            search_direction = self._two_loop_recursion(current_gradient, correction_pairs, model)
            directional_derivative = inner_product(current_gradient, search_direction)
            line_search_result = self._line_search.find_step_size(
                current_point, search_direction, current_loss, directional_derivative, loss_function
            )
            next_point = current_point + line_search_result.step_size * search_direction
            next_loss = line_search_result.loss
            next_gradient = gradient_function(next_point)

            state_difference = next_point - current_point
            gradient_difference = next_gradient - current_gradient
            pair_accepted = self._acceptance_strategy.accept_update(
                state_difference, gradient_difference, current_gradient, model
            )
            if pair_accepted:
                curvature = inner_product(state_difference, gradient_difference)
                if curvature <= 0.0:
                    raise ValueError(
                        "Correction pair accepted by the acceptance strategy has non-positive "
                        f"curvature (s, y) = {curvature}; the limited-memory inverse-Hessian "
                        "approximation requires (s, y) > 0 to stay positive definite."
                    )
                correction_pairs.add(
                    state_difference, gradient_difference, inner_product_reciprocal=1.0 / curvature
                )

            current_point, current_loss, current_gradient = next_point, next_loss, next_gradient
            gradient_norm = model.evaluate_norm(current_gradient)
            iteration += 1
            self._log_debug_iteration_detail(iteration, line_search_result.step_size, pair_accepted)
            callback(current_loss, gradient_norm)
            converged = gradient_norm <= self._settings.gradient_norm_tolerance

        return _CustomLBFGSRawResult(
            final_point=current_point,
            num_iterations=iteration,
            converged=converged,
            final_gradient_norm=gradient_norm,
        )

    # ----------------------------------------------------------------------------------------------
    def _log_debug_iteration_detail(
        self, iteration: int, step_size: float, pair_accepted: bool
    ) -> None:
        """Log the step size taken and whether the resulting correction pair was accepted into the
        limited-memory store, if a logger is attached."""
        if self._logger is not None:
            acceptance = "accepted" if pair_accepted else "rejected"
            self._logger.debug(
                f"Iteration {iteration}: step size {step_size:.3e}, correction pair {acceptance}."
            )

    # ----------------------------------------------------------------------------------------------
    @override
    def _create_optimization_result(
        self, raw_result: _CustomLBFGSRawResult, history: OptimizationHistory
    ) -> OptimizationResult:
        """Map the raw iteration result and the recorded history onto `OptimizationResult`."""
        if raw_result.converged:
            status_message = (
                f"Converged: gradient norm {raw_result.final_gradient_norm:.3e} <= tolerance "
                f"{self._settings.gradient_norm_tolerance:.3e}."
            )
        else:
            status_message = (
                f"Did not converge within {self._settings.maximum_num_iterations} iterations."
            )
        return OptimizationResult(
            result=raw_result.final_point,
            loss_history=history.loss_history,
            gradient_norm_history=history.gradient_norm_history,
            num_iterations=raw_result.num_iterations,
            success=raw_result.converged,
            status_message=status_message,
        )
