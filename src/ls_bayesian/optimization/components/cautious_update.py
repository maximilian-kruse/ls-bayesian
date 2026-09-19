r"""Correction-pair acceptance strategies for metric-generic L-BFGS.

Classes:
    CorrectionPairAcceptanceStrategy: ABC interface for correction-pair acceptance policies.
    AlwaysAcceptStrategy: Trivial strategy accepting every correction pair.
    CautiousUpdateSettings: Settings for `CautiousUpdateStrategy`.
    CautiousUpdateStrategy: Cautious updating, the strengthened curvature condition of Li &
        Fukushima (2001).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import Annotated, override

import numpy as np
from beartype.vale import Is

from ls_bayesian.optimization.model import OptimizationModel

# A small epsilon, consistent with the cautious-update literature (Li & Fukushima, "A globally
# convergent BFGS method for nonconvex minimization without line search", 2001), keeps the
# condition close to the standard curvature condition ((y, s) > 0) while still guarding against
# near-degenerate pairs; alpha=1 is the simplest exponent used in that condition. These are
# reasonable starting defaults, not universal optima; problem-specific tuning may be warranted.
DEFAULT_EPSILON = 1e-6
DEFAULT_ALPHA = 1.0


# ==================================================================================================
class CorrectionPairAcceptanceStrategy(ABC):
    """ABC interface for correction-pair acceptance policies.

    Methods:
        accept_update: Decide whether a correction pair should be stored.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def accept_update(
        self,
        s: np.ndarray[tuple[int], np.dtype[np.float64]],
        y: np.ndarray[tuple[int], np.dtype[np.float64]],
        gradient: np.ndarray[tuple[int], np.dtype[np.float64]],
        model: OptimizationModel,
    ) -> bool:
        r"""Decide whether a correction pair should be stored.

        Args:
            s (np.ndarray[tuple[int], np.dtype[np.float64]]): Iterate displacement $s$.
            y (np.ndarray[tuple[int], np.dtype[np.float64]]): Gradient displacement $y$.
            gradient (np.ndarray[tuple[int], np.dtype[np.float64]]): Gradient $g$ the displacement
                was taken from.
            model (OptimizationModel): Model whose `evaluate_inner_product`/`evaluate_norm` the
                policy is evaluated in.

        Returns:
            bool: Whether the pair should be stored.
        """


# ==================================================================================================
class AlwaysAcceptStrategy(CorrectionPairAcceptanceStrategy):
    """Trivial strategy accepting every correction pair, unconditionally."""

    # ----------------------------------------------------------------------------------------------
    @override
    def accept_update(
        self,
        s: np.ndarray[tuple[int], np.dtype[np.float64]],
        y: np.ndarray[tuple[int], np.dtype[np.float64]],
        gradient: np.ndarray[tuple[int], np.dtype[np.float64]],
        model: OptimizationModel,
    ) -> bool:
        """Always accept."""
        return True


# ==================================================================================================
@dataclass
class CautiousUpdateSettings:
    r"""Settings for `CautiousUpdateStrategy`.

    The field constraints are validated on initialization.

    Attributes:
        epsilon (Real): Threshold scale $\epsilon \geq 0$. `epsilon = 0` reduces the condition to
            the standard curvature condition $(y, s) > 0$. Defaults to `DEFAULT_EPSILON`.
        alpha (Real): Exponent $\alpha > 0$ on the gradient norm. Defaults to `DEFAULT_ALPHA`.
    """

    epsilon: Annotated[Real, Is[lambda x: x >= 0]] = DEFAULT_EPSILON
    alpha: Annotated[Real, Is[lambda x: x > 0]] = DEFAULT_ALPHA


# ==================================================================================================
class CautiousUpdateStrategy(CorrectionPairAcceptanceStrategy):
    r"""Cautious updating, the strengthened curvature condition of Li & Fukushima (2001).

    Implements `optimization.tex`, eq. 81 / lines 162-168:

    $$
    \frac{(y, s)}{\|s\|^2} \geq \epsilon \|g\|^\alpha,
    $$

    where $g$ is the gradient at the point the pair's displacement was taken from. Accepting a
    pair only when this (strengthened) curvature condition holds keeps the limited-memory inverse
    Hessian approximation positive definite (Li & Fukushima, 2001). At `epsilon = 0`, this reduces
    exactly to the standard curvature condition $(y, s) > 0$.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: CautiousUpdateSettings) -> None:
        """Initialize the strategy.

        Args:
            settings (CautiousUpdateSettings): Settings for the condition.
        """
        self._settings = settings

    # ----------------------------------------------------------------------------------------------
    @override
    def accept_update(
        self,
        s: np.ndarray[tuple[int], np.dtype[np.float64]],
        y: np.ndarray[tuple[int], np.dtype[np.float64]],
        gradient: np.ndarray[tuple[int], np.dtype[np.float64]],
        model: OptimizationModel,
    ) -> bool:
        """Evaluate the cautious update condition for a correction pair."""
        inner_product = model.evaluate_inner_product
        s_norm_squared = inner_product(s, s)
        if s_norm_squared == 0.0:
            return False
        gradient_norm = model.evaluate_norm(gradient)
        curvature_ratio = inner_product(y, s) / s_norm_squared
        return curvature_ratio >= self._settings.epsilon * gradient_norm**self._settings.alpha
