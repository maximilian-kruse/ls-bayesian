r"""ABC interfaces of the Gaussian measure and target potential used by function-space MCMC
algorithms. Measures are components for an `MCMCModel`.

Function-space MCMC algorithms (Cotter, Roberts, Stuart, White, 2013) sample from a target
measure $\mu$ defined via its density with respect to a fixed reference measure $\mu_0$ -- a
generic Gaussian measure,

$$
\frac{d\mu}{d\mu_0} \propto \exp(-\Phi(u)).
$$

[`TargetMeasure`][ls_bayesian.mcmc.measures.TargetMeasure] represents $\Phi$ (relative to the
implicit, fixed $\mu_0$); [`DifferentiableTargetMeasure`]
[ls_bayesian.mcmc.measures.DifferentiableTargetMeasure] additionally exposes $\nabla\Phi$, for
algorithms that use it. [`GaussianMeasure`][ls_bayesian.mcmc.measures.GaussianMeasure] represents
one Gaussian measure $\mathcal N(\bar u, C)$ -- its mean, its covariance/precision actions, a
covariance factorization for sampling, and its own potential -- with no assumption about which
role it plays: the same interface represents $\mu_0$ itself, a cheaper or better-fitting Gaussian
approximation $\nu$ that [`PCNAlgorithm`][ls_bayesian.mcmc.algorithms.pcn.PCNAlgorithm] proposes
from instead (Pinski, Simpson, Stuart, Weber, 2015), or a fixed preconditioner $\overline K$ that
[`PMALAAlgorithm`][ls_bayesian.mcmc.algorithms.pmala.PMALAAlgorithm] uses in place of $\mu_0$'s
own covariance (Beskos, Girolami, Lan, Farrell, Stuart, 2017); which role a given
`GaussianMeasure` is asked to play, and which of its methods that role actually reads, is
determined solely by where it is placed in an
[`MCMCModel`][ls_bayesian.mcmc.model.MCMCModel] and which algorithm consumes that model, not by
the interface itself.

Pinski et al.'s proposal from $\nu \neq \mu_0$ additionally needs the Radon-Nikodym correction
$\rho(u) = \log\frac{d\mu_0}{d\nu}(u)$ between $\mu_0$ and $\nu$, vanishing exactly when $\nu =
\mu_0$. For two Gaussians this correction is closed-form -- $\rho(u) = \frac{1}{2}(u-\bar u_\nu)^T
C_\nu^{-1}(u-\bar u_\nu) - \frac{1}{2} u^T C^{-1} u$, up to an additive constant -- and needs only
$\mu_0$'s and $\nu$'s own potentials
([`GaussianMeasure.evaluate_cost`][ls_bayesian.mcmc.measures.GaussianMeasure.evaluate_cost]), so
it is computed directly by [`PCNAlgorithm`][ls_bayesian.mcmc.algorithms.pcn.PCNAlgorithm] itself
rather than by a separate interface implementations would otherwise have to re-derive.

Classes:
    TargetMeasure: ABC interface for the target's potential $\Phi(u)$, relative to $\mu_0$.
    DifferentiableTargetMeasure: `TargetMeasure` additionally exposing the gradient
        $\nabla\Phi(u)$.
    GaussianMeasure: ABC interface for one Gaussian measure $\mathcal N(\bar u, C)$: its mean,
        covariance/precision actions, a covariance factorization for sampling, and its own
        potential.
"""

from abc import ABC, abstractmethod

import numpy as np


# ==================================================================================================
class TargetMeasure(ABC):
    r"""ABC interface for the actual target $\mu$'s potential $\Phi(u)$, relative to the
    (implicit, fixed) reference measure $\mu_0$, $\frac{d\mu}{d\mu_0} \propto \exp(-\Phi(u))$.

    Methods:
        evaluate_potential: Evaluate $\Phi(u)$.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_potential(self, state: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        r"""Evaluate the potential $\Phi(u)$.

        Args:
            state (np.ndarray[tuple[int], np.dtype[np.float64]]): State $u$.

        Returns:
            float: Potential value, up to an additive constant.
        """


# --------------------------------------------------------------------------------------------------
class DifferentiableTargetMeasure(TargetMeasure):
    r"""`TargetMeasure` additionally exposing the gradient $\nabla\Phi(u)$, for algorithms that use
    it (e.g. [`MALAAlgorithm`][ls_bayesian.mcmc.algorithms.mala.MALAAlgorithm]).

    Methods:
        evaluate_potential: Evaluate $\Phi(u)$.
        evaluate_gradient: Evaluate $\nabla\Phi(u)$.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_gradient(
        self, state: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Evaluate the gradient $\nabla\Phi(u)$.

        Args:
            state (np.ndarray[tuple[int], np.dtype[np.float64]]): State $u$.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Gradient, same shape as `state`.
        """


# ==================================================================================================
class GaussianMeasure(ABC):
    r"""ABC interface for one Gaussian measure $\mathcal N(\bar u, C)$: its mean, its covariance and
    precision actions, a covariance factorization for drawing samples, and its own potential
    $\frac{1}{2}(u-\bar u)^T C^{-1}(u-\bar u)$.

    Which role a given `GaussianMeasure` plays -- the target's actual reference measure $\mu_0$, an
    alternative Gaussian $\nu$ proposed from, or a fixed preconditioner $\overline K$ -- is
    determined by an [`MCMCModel`][ls_bayesian.mcmc.model.MCMCModel] and the algorithm consuming
    it, not by this interface: a given algorithm may read only a subset of these methods (e.g.
    [`PMALAAlgorithm`][ls_bayesian.mcmc.algorithms.pmala.PMALAAlgorithm] never reads its
    preconditioner's `mean`, and only
    [`PCNAlgorithm`][ls_bayesian.mcmc.algorithms.pcn.PCNAlgorithm] ever reads `evaluate_cost`)
    An implementation must still provide all of them.

    Methods:
        apply_covariance_factorization: Apply the covariance factorization $\widehat{C}$.
        apply_covariance_operator: Apply the covariance operator $C$.
        apply_precision_operator: Apply the precision operator $C^{-1}$.
        evaluate_cost: Evaluate this measure's own potential $\frac{1}{2}(u-\bar u)^T C^{-1}(u-\bar
            u)$, up to an additive constant.

    Attributes:
        mean (np.ndarray[tuple[int], np.dtype[np.float64]]): Mean $\bar u$.
        random_vector_size (int): Size of the i.i.d. standard normal vector required by
            `apply_covariance_factorization`.
    """

    # ----------------------------------------------------------------------------------------------
    @property
    @abstractmethod
    def mean(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Return the mean $\bar u$ of the measure."""

    # ----------------------------------------------------------------------------------------------
    @property
    @abstractmethod
    def random_vector_size(self) -> int:
        r"""Return the size of the i.i.d. standard normal vector required by
        `apply_covariance_factorization`."""

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def apply_covariance_factorization(
        self, random_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply the covariance factorization $\widehat{C}$, $\widehat{C}\widehat{C}^T = C$.

        Args:
            random_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): I.i.d. standard normal
                vector, shape `(random_vector_size,)`.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Result, same shape as `mean`.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def apply_covariance_operator(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply the covariance operator $C$.

        Args:
            vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Vector to apply the covariance
                operator to, same shape as `mean`.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Result, same shape as `mean`.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def apply_precision_operator(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply the precision operator $C^{-1}$.

        Args:
            vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Vector to apply the precision
                operator to, same shape as `mean`.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Result, same shape as `mean`.
        """

    # ----------------------------------------------------------------------------------------------
    def evaluate_cost(self, state: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        r"""Evaluate this measure's own potential $\frac{1}{2}(u-\bar u)^T C^{-1}(u-\bar u)$ at
        `state`, up to an additive constant -- i.e. its negative log-density. Concrete in terms of
        `mean`/`apply_precision_operator`, so implementations need not repeat this formula
        themselves.

        Args:
            state (np.ndarray[tuple[int], np.dtype[np.float64]]): State $u$.

        Returns:
            float: Potential value, up to an additive constant.
        """
        difference = state - self.mean
        return float(0.5 * np.dot(difference, self.apply_precision_operator(difference)))
