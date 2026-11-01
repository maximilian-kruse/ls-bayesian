r"""Truncated conjugate gradients for the Newton system in an arbitrary inner product.

Classes:
    TruncationReason: Why a truncated CG solve stopped.
    TruncatedCGResult: Outcome of a truncated CG solve.
    TruncatedConjugateGradient: Truncated CG for $\mathbf{H}\mathbf{p} = -\mathbf{g}$.
"""

import math
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
    TRUST_REGION_BOUNDARY = auto()


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
        residual (np.ndarray[tuple[int], np.dtype[np.float64]]): The final residual
            $\\mathbf{r} = \\mathbf{H}\\mathbf{p} + \\mathbf{g}$ itself, which gives the value of
            the quadratic model without another Hessian-vector product,
            $(\\mathbf{g}, \\mathbf{p}) + \\frac{1}{2}(\\mathbf{p}, \\mathbf{H}\\mathbf{p}) =
            \\frac{1}{2}\\{(\\mathbf{g}, \\mathbf{p}) + (\\mathbf{p}, \\mathbf{r})\\}$. `None` if
            the direction is the steepest-descent fallback $-\\mathbf{g}$ of a solve without trust
            region, whose residual is not known.
    """

    direction: np.ndarray[tuple[int], np.dtype[np.float64]]
    num_iterations: int
    reason: TruncationReason
    residual_norm: float
    residual: np.ndarray[tuple[int], np.dtype[np.float64]] | None


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
    $(\mathbf{d}, \mathbf{H}\mathbf{d}) \le 0$. Without a trust region, the last case returns the
    current iterate, or $-\mathbf{g}$ if that happens in the first iteration, so the result is
    always a descent direction; the caller globalizes with a line search.

    With a trust-region radius $\Delta$ this is the Steihaug-Toint method (Steihaug, 1983;
    Nocedal & Wright, Algorithm 7.2): as soon as an iterate would leave the region
    $\|\mathbf{p}\| \le \Delta$, or a direction of non-positive curvature is met, the iteration
    stops at the boundary point $\mathbf{p} + \tau\mathbf{d}$ with $\|\mathbf{p} +
    \tau\mathbf{d}\| = \Delta$, $\tau \ge 0$. The norm is the one induced by `inner_product`.
    The first CG iterates are steepest-descent-like, so a small radius gives a short step in a
    useful direction.

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
        trust_region_radius: Real | None = None,
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
            trust_region_radius (Real | None, optional): Radius $\Delta > 0$ of the trust region in
                the norm of `inner_product`. `None` (default) means no trust region.

        Raises:
            ValueError: If `trust_region_radius` is not positive.

        Returns:
            TruncatedCGResult: Approximate Newton direction and the reason the iteration stopped.
        """
        if trust_region_radius is not None and not trust_region_radius > 0.0:
            raise ValueError(f"trust_region_radius must be positive, got {trust_region_radius}.")
        radius_square = None if trust_region_radius is None else float(trust_region_radius) ** 2

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
                residual=None,
            )

        # (p, p), (p, d) and (d, d), tracked only for the trust-region test
        direction_square = 0.0
        direction_conjugate = 0.0
        conjugate_square = residual_square
        for iteration in range(self._settings.maximum_num_iterations):
            hessian_conjugate = hessian_action(conjugate)
            curvature = inner_product(conjugate, hessian_conjugate)
            step = math.nan if curvature <= 0.0 else residual_square / curvature

            if radius_square is not None and (
                curvature <= 0.0
                or direction_square + 2.0 * step * direction_conjugate + step**2 * conjugate_square
                >= radius_square
            ):
                boundary_step = self._boundary_step(
                    direction_square, direction_conjugate, conjugate_square, radius_square
                )
                return TruncatedCGResult(
                    direction=direction + boundary_step * conjugate,
                    num_iterations=iteration + 1,
                    reason=(
                        TruncationReason.NON_POSITIVE_CURVATURE
                        if curvature <= 0.0
                        else TruncationReason.TRUST_REGION_BOUNDARY
                    ),
                    residual_norm=residual_norm,
                    residual=residual + boundary_step * hessian_conjugate,
                )
            if curvature <= 0.0:
                return TruncatedCGResult(
                    direction=-gradient if iteration == 0 else direction,
                    num_iterations=iteration + 1,
                    reason=TruncationReason.NON_POSITIVE_CURVATURE,
                    residual_norm=residual_norm,
                    residual=None if iteration == 0 else residual,
                )

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
                    residual=residual,
                )
            conjugate = -residual + (next_residual_square / residual_square) * conjugate
            residual_square = next_residual_square
            if radius_square is not None:
                direction_square = inner_product(direction, direction)
                direction_conjugate = inner_product(direction, conjugate)
                conjugate_square = inner_product(conjugate, conjugate)

        return TruncatedCGResult(
            direction=direction,
            num_iterations=self._settings.maximum_num_iterations,
            reason=TruncationReason.MAXIMUM_ITERATIONS,
            residual_norm=residual_norm,
            residual=residual,
        )

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _boundary_step(
        direction_square: float,
        direction_conjugate: float,
        conjugate_square: float,
        radius_square: float,
    ) -> float:
        r"""Positive root $\tau$ of $\|\mathbf{p} + \tau\mathbf{d}\|^2 = \Delta^2$.

        Args:
            direction_square (float): $(\mathbf{p}, \mathbf{p})$, at most $\Delta^2$.
            direction_conjugate (float): $(\mathbf{p}, \mathbf{d})$.
            conjugate_square (float): $(\mathbf{d}, \mathbf{d})$.
            radius_square (float): $\Delta^2$.
        """
        discriminant = direction_conjugate**2 + conjugate_square * (
            radius_square - direction_square
        )
        return (-direction_conjugate + math.sqrt(max(discriminant, 0.0))) / conjugate_square
