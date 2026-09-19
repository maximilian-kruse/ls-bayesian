r"""L-BFGS with cautious updating in an arbitrary inner-product space.

Classes:
    CorrectionPair: One correction pair $(s_i, y_i, \rho_i)$ of the limited-memory Hessian
        approximation.
    CorrectionPairStore: Bounded FIFO memory of the last $M$ correction pairs.
    MetricLBFGSSettings: Settings for `MetricLBFGSOptimizer`.
    MetricLBFGSOptimizer: L-BFGS with cautious updating, generalized to an arbitrary inner
        product.

Functions:
    identity_seed_operator: The identity seed operator $\mathbf{H}_k^0 = \mathbf{I}$.
    two_loop_recursion: L-BFGS two-loop recursion, computing a search direction from the current
        gradient and stored correction pairs, without forming the inverse-Hessian approximation
        explicitly.
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
    OptimizationHistory,
    OptimizationResult,
)

# Standard L-BFGS memory size (Nocedal & Wright, "Numerical Optimization", 2006, Sec. 7.2):
# commonly chosen between 3 and 20; 10 is a widely used, moderate default.
DEFAULT_MEMORY_SIZE = 10
# Generic safeguard against non-termination; not tied to any specific problem scale.
DEFAULT_MAXIMUM_NUM_ITERATIONS = 1000
# A typical convergence tolerance on the gradient norm for double-precision optimization.
DEFAULT_GRADIENT_NORM_TOLERANCE = 1e-6

type SeedOperator = Callable[
    [np.ndarray[tuple[int], np.dtype[np.float64]]], np.ndarray[tuple[int], np.dtype[np.float64]]
]


# ==================================================================================================
def identity_seed_operator(
    vector: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    r"""The identity seed operator, $\mathbf{H}_k^0 = \mathbf{I}$.

    The default initial inverse-Hessian approximation for the two-loop recursion. In a
    Cameron-Martin space induced by a Gaussian prior's covariance, the prior term's Hessian with
    respect to the Cameron-Martin inner product is exactly the identity, making this the natural
    "structured seed matrix" choice (Mannel & Rund 2024, Petra & Ghattas 2019); other seed
    operators remain pluggable via `MetricLBFGSOptimizer`'s `seed_operator` argument.

    Args:
        vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Vector to apply the seed operator
            to.

    Returns:
        np.ndarray[tuple[int], np.dtype[np.float64]]: `vector`, unchanged.
    """
    return vector


# ==================================================================================================
@dataclass(frozen=True)
class CorrectionPair:
    r"""One correction pair $(s_i, y_i, \rho_i)$ of the limited-memory Hessian approximation.

    Attributes:
        s (np.ndarray[tuple[int], np.dtype[np.float64]]): Iterate displacement
            $s_i = m_{i+1} - m_i$.
        y (np.ndarray[tuple[int], np.dtype[np.float64]]): Gradient displacement
            $y_i = \nabla I(m_{i+1}) - \nabla I(m_i)$.
        rho (float): $\rho_i = 1 / (s_i, y_i)$, computed once at insertion.
    """

    s: np.ndarray[tuple[int], np.dtype[np.float64]]
    y: np.ndarray[tuple[int], np.dtype[np.float64]]
    rho: float


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
        s: np.ndarray[tuple[int], np.dtype[np.float64]],
        y: np.ndarray[tuple[int], np.dtype[np.float64]],
        rho: float,
    ) -> None:
        r"""Add a new correction pair, evicting the oldest one if the store is full.

        Args:
            s (np.ndarray[tuple[int], np.dtype[np.float64]]): Iterate displacement $s_i$.
            y (np.ndarray[tuple[int], np.dtype[np.float64]]): Gradient displacement $y_i$.
            rho (float): $\rho_i = 1 / (s_i, y_i)$.
        """
        self._pairs.append(CorrectionPair(s=s, y=y, rho=rho))

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
def two_loop_recursion(
    gradient: np.ndarray[tuple[int], np.dtype[np.float64]],
    correction_pairs: CorrectionPairStore,
    model: OptimizationModel,
    seed_operator: SeedOperator,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    r"""L-BFGS two-loop recursion, computing a search direction without forming the inverse
    Hessian approximation $\mathbf{H}_k$ explicitly.

    Implements the two-loop recursion in the form of Nocedal & Wright, "Numerical Optimization"
    (2006), Algorithm 7.4, generalized to an arbitrary inner product, via
    `model.evaluate_inner_product`. With no stored correction pairs and the identity seed
    operator, this reduces exactly to steepest descent, $\mathbf{p}_k = -\mathbf{g}_k$.

    Deviation from `optimization.tex`: the notes' first loop (lines 120-127) compute
    $\eta_i = \rho_i(\mathbf{y}_i, \mathbf{q})$, dotting $\mathbf{q}$ with $\mathbf{y}_i$. The
    standard recursion (and the only form consistent with the explicit BFGS inverse-Hessian update
    the same notes state in eq. 29/43 — verified numerically: reconstructing $\mathbf{H}_1$
    explicitly from that update and applying it to a gradient does not match the notes' two-loop
    pseudocode for a single correction pair) instead dots $\mathbf{q}$ with $\mathbf{s}_i$,
    i.e. $\eta_i = \rho_i(\mathbf{s}_i, \mathbf{q})$; this is what is implemented here. This looks
    like a transcription slip in the notes (the second loop's $\zeta_i = \rho_i(\mathbf{y}_i,
    \mathbf{r})$ already matches the standard form).

    Args:
        gradient (np.ndarray[tuple[int], np.dtype[np.float64]]): Current gradient
            $\mathbf{g}_k$, represented in `model`'s inner product.
        correction_pairs (CorrectionPairStore): Stored correction pairs
            $\{(\mathbf{s}_i, \mathbf{y}_i, \rho_i)\}$.
        model (OptimizationModel): Model whose `evaluate_inner_product` the recursion is carried
            out in.
        seed_operator (SeedOperator): Initial inverse-Hessian approximation $\mathbf{H}_k^0$.

    Returns:
        np.ndarray[tuple[int], np.dtype[np.float64]]: Search direction $\mathbf{p}_k = -\mathbf{r}$.
    """
    inner_product = model.evaluate_inner_product
    q = gradient.copy()
    etas: list[float] = []
    for pair in reversed(correction_pairs):
        eta = pair.rho * inner_product(pair.s, q)
        q = q - eta * pair.y
        etas.append(eta)

    r = seed_operator(q)
    for pair, eta in zip(correction_pairs, reversed(etas), strict=True):
        zeta = pair.rho * inner_product(pair.y, r)
        r = r + pair.s * (eta - zeta)

    return -r


# ==================================================================================================
@dataclass
class MetricLBFGSSettings:
    r"""Settings for `MetricLBFGSOptimizer`.

    The field constraints are validated on initialization.

    Attributes:
        memory_size (int): Number of correction pairs to keep, $M$. Defaults to
            `DEFAULT_MEMORY_SIZE`.
        maximum_num_iterations (int): Maximum number of outer iterations. Defaults to
            `DEFAULT_MAXIMUM_NUM_ITERATIONS`.
        gradient_norm_tolerance (Real): Convergence tolerance on the gradient norm, evaluated in
            the optimizer's inner-product space. Defaults to `DEFAULT_GRADIENT_NORM_TOLERANCE`.
    """

    memory_size: Annotated[int, Is[lambda x: x > 0]] = DEFAULT_MEMORY_SIZE
    maximum_num_iterations: Annotated[int, Is[lambda x: x > 0]] = DEFAULT_MAXIMUM_NUM_ITERATIONS
    gradient_norm_tolerance: Annotated[Real, Is[lambda x: x > 0]] = DEFAULT_GRADIENT_NORM_TOLERANCE


# ==================================================================================================
@dataclass(frozen=True)
class _MetricLBFGSRawResult:
    """Raw result of `MetricLBFGSOptimizer._run_impl`, mapped onto `OptimizationResult` by
    `_create_optimization_result`."""

    final_point: np.ndarray[tuple[int], np.dtype[np.float64]]
    num_iterations: int
    converged: bool
    final_gradient_norm: float


# ==================================================================================================
class MetricLBFGSOptimizer(BaseOptimizer):
    r"""L-BFGS with cautious updating, generalized to an arbitrary inner-product space.

    Implements the algorithm of `optimization.tex` (Algorithm 2: two-loop recursion, Armijo
    backtracking line search, cautious correction-pair updating), generalized from the
    Cameron-Martin space of that derivation to whatever inner product the
    [`OptimizationModel`][ls_bayesian.optimization.model.OptimizationModel] passed to `run`
    implements via `evaluate_inner_product`/`evaluate_norm`. `optimization` never depends on where
    that inner product comes from (e.g. a Cameron-Martin inner product induced by a Bayesian
    prior's covariance, built in another subpackage).

    **`model.evaluate_gradient` must return the Riesz representer of the objective's derivative
    under `model.evaluate_inner_product`** -- e.g. the Cameron-Martin representer, not an
    L2/Euclidean discretized gradient. Converting between representers (e.g. applying a prior
    covariance operator to a raw discretized gradient) is the caller's responsibility, performed
    outside this class, when implementing the `OptimizationModel`.

    Deviating from the notes' unbounded backtracking `while` loop (line 142-150), the injected
    [`LineSearch`][ls_bayesian.optimization.components.line_search.LineSearch] caps the number of
    backtracking steps and raises if none succeeds, since an unbounded loop could hang if
    `search_direction` is not a genuine descent direction due to numerical error. Termination is
    not specified in the notes; this class adds `maximum_num_iterations` and
    `gradient_norm_tolerance` (Sec. `MetricLBFGSSettings`) as explicit stopping criteria.

    Attributes:
        requires_hessian (bool): Always `False`; this method does not use Hessian information.
    """

    requires_hessian: bool = False

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        settings: MetricLBFGSSettings,
        line_search: LineSearchStrategy,
        acceptance_strategy: CorrectionPairAcceptanceStrategy,
        seed_operator: SeedOperator = identity_seed_operator,
        logger: BaseLogger | None = None,
    ) -> None:
        r"""Initialize the optimizer.

        Args:
            settings (MetricLBFGSSettings): Settings for the optimizer.
            line_search (LineSearch): Step-size selection strategy.
            acceptance_strategy (CorrectionPairAcceptanceStrategy): Correction-pair acceptance
                policy.
            seed_operator (SeedOperator, optional): Initial inverse-Hessian approximation
                $\mathbf{H}_k^0$ for the two-loop recursion. Defaults to
                `identity_seed_operator`.
            logger (BaseLogger | None, optional): Logger for iteration-by-iteration progress
                reports. Defaults to `None`.
        """
        super().__init__(logger=logger)
        self._settings = settings
        self._line_search = line_search
        self._acceptance_strategy = acceptance_strategy
        self._seed_operator = seed_operator

    # ----------------------------------------------------------------------------------------------
    @override
    def _run_impl(
        self,
        initial_guess: np.ndarray[tuple[int], np.dtype[np.float64]],
        model: OptimizationModel,
        callback: Callable[..., None],
    ) -> _MetricLBFGSRawResult:
        """Run the outer iteration: two-loop recursion for the direction, line search for the
        step, cautious updating for whether to store the new correction pair, until convergence or
        `maximum_num_iterations` is reached.

        `model.evaluate_hessian_vector_product` is never called, since this method does not use
        Hessian information.
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
            search_direction = two_loop_recursion(
                current_gradient, correction_pairs, model, self._seed_operator
            )
            directional_derivative = inner_product(current_gradient, search_direction)
            line_search_result = self._line_search.find_step_size(
                current_point, search_direction, current_loss, directional_derivative, loss_function
            )
            next_point = current_point + line_search_result.step_size * search_direction
            next_loss = line_search_result.loss
            next_gradient = gradient_function(next_point)

            s = next_point - current_point
            y = next_gradient - current_gradient
            if self._acceptance_strategy.accept_update(s, y, current_gradient, model):
                correction_pairs.add(s, y, rho=1.0 / inner_product(s, y))

            current_point, current_loss, current_gradient = next_point, next_loss, next_gradient
            gradient_norm = model.evaluate_norm(current_gradient)
            iteration += 1
            callback(current_point)
            converged = gradient_norm <= self._settings.gradient_norm_tolerance

        return _MetricLBFGSRawResult(
            final_point=current_point,
            num_iterations=iteration,
            converged=converged,
            final_gradient_norm=gradient_norm,
        )

    # ----------------------------------------------------------------------------------------------
    @override
    def _create_optimization_result(
        self, raw_result: _MetricLBFGSRawResult, history: OptimizationHistory
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
