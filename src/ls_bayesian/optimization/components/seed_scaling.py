r"""Seed-scaling strategies for the L-BFGS two-loop recursion's initial inverse-Hessian
approximation.

Classes:
    SeedScalingStrategy: ABC interface for seed-scaling policies.
    NoSeedScaling: Trivial strategy applying no scaling; the factor is always `1.0`.
    BarzilaiBorweinSeedScalingSettings: Settings for `BarzilaiBorweinSeedScaling`.
    BarzilaiBorweinSeedScaling: Barzilai-Borwein-style scaling, clamped to a configurable range.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import Annotated, override

import numpy as np
from beartype.vale import Is

from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.optimization.model import OptimizationModel


# ==================================================================================================
class SeedScalingStrategy(ABC):
    r"""ABC interface for scaling the L-BFGS two-loop recursion's seed-operator output.

    `CustomLBFGSOptimizer._two_loop_recursion` computes $\mathbf{r} = \gamma_k
    \mathbf{H}_k^0\mathbf{q}$, i.e. the injected `SeedOperator`'s output, scaled by whatever this
    strategy returns.

    Methods:
        compute_scale_factor: Compute the scalar multiplying the seed operator's output.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def compute_scale_factor(
        self,
        newest_state_difference: np.ndarray[tuple[int], np.dtype[np.float64]] | None,
        newest_gradient_difference: np.ndarray[tuple[int], np.dtype[np.float64]] | None,
        model: OptimizationModel,
    ) -> float:
        r"""Compute the scale factor $\gamma_k$.

        Args:
            newest_state_difference (np.ndarray[tuple[int], np.dtype[np.float64]] | None): Most
                recently stored correction pair's iterate displacement $\mathbf{s}_{k-1}$, or
                `None` if no correction pair is stored yet.
            newest_gradient_difference (np.ndarray[tuple[int], np.dtype[np.float64]] | None): The
                same pair's gradient displacement $\mathbf{y}_{k-1}$, or `None` if no correction
                pair is stored yet.
            model (OptimizationModel): Model whose `evaluate_inner_product` the factor is computed
                in.

        Returns:
            float: Scale factor $\gamma_k$.
        """


# ==================================================================================================
class NoSeedScaling(SeedScalingStrategy):
    """Trivial strategy applying no scaling: the factor is always `1.0`, so the two-loop
    recursion's seed step is exactly `SeedOperator`'s own output, unmodified."""

    # ----------------------------------------------------------------------------------------------
    @override
    def compute_scale_factor(
        self,
        newest_state_difference: np.ndarray[tuple[int], np.dtype[np.float64]] | None,
        newest_gradient_difference: np.ndarray[tuple[int], np.dtype[np.float64]] | None,
        model: OptimizationModel,
    ) -> float:
        """Always return `1.0`."""
        return 1.0


# ==================================================================================================
@dataclass
class BarzilaiBorweinSeedScalingSettings:
    r"""Settings for `BarzilaiBorweinSeedScaling`.

    The field constraints are validated on initialization.

    Attributes:
        gamma_min (Real): Lower bound the seed-scaling factor $\gamma_k$ is clamped to. Defaults
            to `1e-2`, two orders of magnitude below the "neutral" value 1. A reasonable starting
            default, not a universal optimum; problem-specific tuning may be warranted.
        gamma_max (Real): Upper bound $\gamma_k$ is clamped to. Defaults to `1e2`, two orders of
            magnitude above 1, for the same reason as `gamma_min`. Guards against the scenario
            where a correction pair passes the acceptance strategy's curvature-ratio check (which
            bounds $(y,s)/\|s\|^2$) while still having a near-degenerate $\|y\|^2$, which would
            otherwise blow up $\gamma_k=(s,y)/\|y\|^2$ and the resulting trial step size.
    """

    gamma_min: Annotated[Real, Is[lambda x: x > 0]] = 1e-2
    gamma_max: Annotated[Real, Is[lambda x: x > 0]] = 1e2

    def __post_init__(self) -> None:
        """Check `gamma_min < gamma_max`"""
        if self.gamma_min >= self.gamma_max:
            raise ValueError(
                f"gamma_min ({self.gamma_min}) must be strictly less than gamma_max "
                f"({self.gamma_max})."
            )


# ==================================================================================================
class BarzilaiBorweinSeedScaling(SeedScalingStrategy):
    r"""Barzilai-Borwein-style seed scaling, clamped to `[settings.gamma_min, settings.gamma_max]`.

    Computes $\gamma_k = (\mathbf{s}_{k-1}, \mathbf{y}_{k-1}) / (\mathbf{y}_{k-1},
    \mathbf{y}_{k-1})$ from the most recently stored correction pair (`1.0`, i.e. no scaling, if
    none is stored yet): the standard way L-BFGS keeps its very first trial step per iteration
    close to the right scale, composing with a structured `SeedOperator` rather than replacing it.
    Without it, an un-scaled seed step can require many backtracking halvings every iteration on
    problems whose gradient magnitude is far from unit scale in `model`'s inner product, since a
    line search always starts from the same fixed initial step size regardless of how the objective
    is scaled.

    The raw ratio is clamped to guard against a different failure mode: a correction pair can
    satisfy an acceptance strategy's curvature-ratio condition (e.g. `CautiousUpdateStrategy`,
    which bounds $(y,s)/\|s\|^2$) while still having a near-degenerate $\|y\|^2$, which would
    otherwise blow up $\gamma_k$ and, with it, the very first trial step size every subsequent
    iteration -- forcing the line search into ever more backtracking steps for ever less progress.
    A genuine curvature (Wolfe) condition in the line search prevents such a pair from being
    accepted in the first place (see `StrongWolfeLineSearch`); the clamp here is a cheap safety net
    independent of which line search is in use.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, settings: BarzilaiBorweinSeedScalingSettings, logger: BaseLogger | None = None
    ) -> None:
        """Initialize the strategy.

        Args:
            settings (BarzilaiBorweinSeedScalingSettings): Settings for the clamp bounds.
            logger (BaseLogger | None, optional): Logger for clamped scale factors, reported at
                debug level. Defaults to `None`.
        """
        self._settings = settings
        self._logger = logger

    # ----------------------------------------------------------------------------------------------
    @override
    def compute_scale_factor(
        self,
        newest_state_difference: np.ndarray[tuple[int], np.dtype[np.float64]] | None,
        newest_gradient_difference: np.ndarray[tuple[int], np.dtype[np.float64]] | None,
        model: OptimizationModel,
    ) -> float:
        """Compute the clamped Barzilai-Borwein-style scale factor, or `1.0` if no correction pair
        is stored yet."""
        if newest_state_difference is None or newest_gradient_difference is None:
            return 1.0
        curvature = model.evaluate_inner_product(
            newest_state_difference, newest_gradient_difference
        )
        gradient_difference_squared_norm = model.evaluate_inner_product(
            newest_gradient_difference, newest_gradient_difference
        )
        gamma = curvature / gradient_difference_squared_norm
        clamped_gamma = min(max(gamma, self._settings.gamma_min), self._settings.gamma_max)
        if clamped_gamma != gamma:
            self._log_debug_gamma_clamped(gamma, clamped_gamma)
        return clamped_gamma

    # ----------------------------------------------------------------------------------------------
    def _log_debug_gamma_clamped(self, gamma: float, clamped_gamma: float) -> None:
        """Log that the seed-scaling factor was clamped, if a logger is attached."""
        if self._logger is not None:
            self._logger.debug(
                f"Seed-scaling factor gamma={gamma:.3e} clamped to {clamped_gamma:.3e} "
                f"(bounds [{self._settings.gamma_min:.3e}, {self._settings.gamma_max:.3e}])."
            )
