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
        state_difference: np.ndarray[tuple[int], np.dtype[np.float64]],
        gradient_difference: np.ndarray[tuple[int], np.dtype[np.float64]],
        gradient: np.ndarray[tuple[int], np.dtype[np.float64]],
        model: OptimizationModel,
    ) -> bool:
        r"""Decide whether a correction pair should be stored.

        Args:
            state_difference (np.ndarray[tuple[int], np.dtype[np.float64]]): Iterate displacement
                $s$.
            gradient_difference (np.ndarray[tuple[int], np.dtype[np.float64]]): Gradient
                displacement $y$.
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
        state_difference: np.ndarray[tuple[int], np.dtype[np.float64]],
        gradient_difference: np.ndarray[tuple[int], np.dtype[np.float64]],
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
            the standard curvature condition $(y, s) > 0$. Defaults to `1e-6`, consistent with the
            cautious-update literature (Li & Fukushima, "A globally convergent BFGS method for
            nonconvex minimization without line search", 2001): small enough to keep the condition
            close to the standard curvature condition while still guarding against near-degenerate
            pairs. A reasonable starting default, not a universal optimum; problem-specific tuning
            may be warranted.
        alpha (Real): Exponent $\alpha > 0$ on the gradient norm. Defaults to `1.0`, the simplest
            exponent used in the cautious-update condition (Li & Fukushima, 2001). A reasonable
            starting default, not a universal optimum; problem-specific tuning may be warranted.
    """

    epsilon: Annotated[Real, Is[lambda x: x >= 0]] = 1e-6
    alpha: Annotated[Real, Is[lambda x: x > 0]] = 1.0


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
        state_difference: np.ndarray[tuple[int], np.dtype[np.float64]],
        gradient_difference: np.ndarray[tuple[int], np.dtype[np.float64]],
        gradient: np.ndarray[tuple[int], np.dtype[np.float64]],
        model: OptimizationModel,
    ) -> bool:
        """Evaluate the cautious update condition for a correction pair."""
        inner_product = model.evaluate_inner_product
        state_difference_norm_squared = inner_product(state_difference, state_difference)
        if state_difference_norm_squared == 0.0:
            return False
        gradient_norm = model.evaluate_norm(gradient)
        curvature_ratio = (
            inner_product(gradient_difference, state_difference) / state_difference_norm_squared
        )
        return curvature_ratio >= self._settings.epsilon * gradient_norm**self._settings.alpha
