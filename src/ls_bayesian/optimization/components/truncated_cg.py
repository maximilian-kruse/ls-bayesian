r"""Truncated conjugate gradients for the Newton system in an arbitrary inner product.

Classes:
    TruncationReason: Why a truncated CG solve stopped.
    TruncatedCGResult: Outcome of a truncated CG solve.
    TruncatedConjugateGradient: Truncated CG for $\mathbf{H}\mathbf{p} = -\mathbf{g}$.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from numbers import Real
from typing import Annotated

import numpy as np
from beartype.vale import Is

type VectorFunction = Callable[
    [np.ndarray[tuple[int], np.dtype[np.float64]]], np.ndarray[tuple[int], np.dtype[np.float64]]
]
type InnerProductFunction = Callable[
    [np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]],
    float,
]


# ==================================================================================================
class TruncationReason(Enum):
    """Why a truncated CG solve stopped."""

    CONVERGED = auto()
    MAXIMUM_ITERATIONS = auto()
    NON_POSITIVE_CURVATURE = auto()


# ==================================================================================================
@dataclass(frozen=True)
class TruncatedCGResult:
    """Outcome of a truncated CG solve.

    Attributes:
        direction (np.ndarray[tuple[int], np.dtype[np.float64]]): Approximate Newton direction
            $\\mathbf{p}$; a descent direction for the given gradient.
        num_iterations (int): Number of CG iterations performed (Hessian-vector products).
        reason (TruncationReason): Why the iteration stopped.
        residual_norm (float): Norm of the final residual $\\mathbf{H}\\mathbf{p} + \\mathbf{g}$ in
            the given inner product.
    """

    direction: np.ndarray[tuple[int], np.dtype[np.float64]]
    num_iterations: int
    reason: TruncationReason
    residual_norm: float


# ==================================================================================================
@dataclass
class TruncatedCGSettings:
    """Settings for `TruncatedConjugateGradient`.

    The field constraints are validated on initialization.

    Attributes:
        maximum_num_iterations (int): Maximum number of CG iterations per solve. Defaults to
            `250`, a generous cap: with a well-chosen inner product CG needs far fewer
            iterations, the cap only guards against stagnation.
    """

    maximum_num_iterations: Annotated[int, Is[lambda x: x > 0]] = 250


# ==================================================================================================
class TruncatedConjugateGradient:
    r"""Truncated conjugate gradients for $\mathbf{H}\mathbf{p} = -\mathbf{g}$.

    Algorithm 7.1 of Nocedal & Wright, "Numerical Optimization" (2006), in an arbitrary inner
    product $(\cdot,\cdot)$: $\mathbf{H}$ is a self-adjoint operator and $\mathbf{g}$ a Riesz
    representer under that inner product. The iteration starts at $\mathbf{p} = \mathbf{0}$ and
    stops when the residual norm drops below `relative_tolerance` times $\|\mathbf{g}\|$, when the
    iteration cap is reached, or when it meets a direction of non-positive curvature
    $(\mathbf{d}, \mathbf{H}\mathbf{d}) \le 0$. In the last case it returns the current iterate,
    or $-\mathbf{g}$ if that happens in the first iteration, so the result is always a descent
    direction. There is no trust-region radius; the caller globalizes with a line search.

    Methods:
        solve: Solve the Newton system approximately.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: TruncatedCGSettings) -> None:
        """Initialize the solver.

        Args:
            settings (TruncatedCGSettings): Settings for the solver.
        """
        self._settings = settings

    # ----------------------------------------------------------------------------------------------
    def solve(
        self,
        hessian_action: VectorFunction,
        gradient: np.ndarray[tuple[int], np.dtype[np.float64]],
        inner_product: InnerProductFunction,
        relative_tolerance: Real,
    ) -> TruncatedCGResult:
        r"""Solve $\mathbf{H}\mathbf{p} = -\mathbf{g}$ approximately.

        Args:
            hessian_action (VectorFunction): Action $\mathbf{d} \mapsto \mathbf{H}\mathbf{d}$,
                returning the Riesz representer under `inner_product`.
            gradient (np.ndarray[tuple[int], np.dtype[np.float64]]): Gradient $\mathbf{g}$ as a
                Riesz representer under `inner_product`.
            inner_product (InnerProductFunction): Inner product the iteration is carried out in.
            relative_tolerance (Real): Stop when $\|\mathbf{r}\| \le$ `relative_tolerance`
                $\cdot \|\mathbf{g}\|$.

        Returns:
            TruncatedCGResult: Approximate Newton direction and the reason the iteration stopped.
        """
        gradient_norm = float(np.sqrt(max(inner_product(gradient, gradient), 0.0)))
        threshold = float(relative_tolerance) * gradient_norm
        direction = np.zeros_like(gradient)
        residual = gradient.copy()  # r = H p + g with p = 0
        conjugate = -residual
        residual_square = inner_product(residual, residual)
        residual_norm = float(np.sqrt(max(residual_square, 0.0)))
        if residual_norm <= threshold:
            return TruncatedCGResult(
                direction=-gradient,
                num_iterations=0,
                reason=TruncationReason.CONVERGED,
                residual_norm=residual_norm,
            )

        for iteration in range(self._settings.maximum_num_iterations):
            hessian_conjugate = hessian_action(conjugate)
            curvature = inner_product(conjugate, hessian_conjugate)
            if curvature <= 0.0:
                result_direction = -gradient if iteration == 0 else direction
                return TruncatedCGResult(
                    direction=result_direction,
                    num_iterations=iteration + 1,
                    reason=TruncationReason.NON_POSITIVE_CURVATURE,
                    residual_norm=residual_norm,
                )
            step = residual_square / curvature
            direction = direction + step * conjugate
            residual = residual + step * hessian_conjugate
            next_residual_square = inner_product(residual, residual)
            residual_norm = float(np.sqrt(max(next_residual_square, 0.0)))
            if residual_norm <= threshold:
                return TruncatedCGResult(
                    direction=direction,
                    num_iterations=iteration + 1,
                    reason=TruncationReason.CONVERGED,
                    residual_norm=residual_norm,
                )
            conjugate = -residual + (next_residual_square / residual_square) * conjugate
            residual_square = next_residual_square

        return TruncatedCGResult(
            direction=direction,
            num_iterations=self._settings.maximum_num_iterations,
            reason=TruncationReason.MAXIMUM_ITERATIONS,
            residual_norm=residual_norm,
        )
