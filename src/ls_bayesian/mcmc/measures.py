r"""ABC interfaces for what function-space MCMC algorithms sample against.

Function-space MCMC algorithms (Cotter, Roberts, Stuart, White, 2013) sample from a target
measure $\mu$ defined via its density with respect to a fixed reference measure $\mu_0$ (the
target's actual prior),

$$
\frac{d\mu}{d\mu_0} \propto \exp(-\Phi(u)).
$$

Pinski, Simpson, Stuart, Weber (2015) show that a pCN-type proposal can instead be drawn from any
Gaussian measure $\nu$ equivalent to $\mu_0$ -- e.g. a cheaper or better-fitting Gaussian
approximation of $\mu$ -- at the cost of an additional term in the acceptance probability that
accounts for the mismatch between $\nu$ and $\mu_0$, and that vanishes exactly when $\nu = \mu_0$.

$\mu_0$ itself is never represented as an object in this module: every potential below is already
expressed relative to it implicitly. The only Gaussian measure that is ever instantiated is $\nu$,
the one actually proposed from -- whether it happens to equal $\mu_0$ (as in
[`MALAAlgorithm`][ls_bayesian.mcmc.algorithms.mala.MALAAlgorithm], and in
[`PCNAlgorithm`][ls_bayesian.mcmc.algorithms.pcn.PCNAlgorithm]'s trivial fallback) or not (pCN's
general case). [`ProposalMeasure`][ls_bayesian.mcmc.measures.ProposalMeasure] represents exactly
that object: a Gaussian measure to sample from, together with its potential relative to $\mu_0$.
[`TargetMeasure`][ls_bayesian.mcmc.measures.TargetMeasure] represents the actual target $\mu$,
which is only ever evaluated, never sampled from.

Implementations from other subpackages are not coupled to this module directly; they are
connected through thin adapters in the application layer, mirroring
[`ls_bayesian.posterior.interfaces`][ls_bayesian.posterior.interfaces]. In particular, an
implementation that cannot supply a gradient or a nonzero correction potential may implement the
corresponding method by raising `NotImplementedError`, mirroring
[`LogPosterior.evaluate_hessian_vector_product`][ls_bayesian.posterior.posterior.LogPosterior.evaluate_hessian_vector_product];
whether that is acceptable depends on which algorithm the implementation is used with.

Classes:
    TargetMeasure: ABC interface for the target's potential $\Phi(u)$ and gradient
        $\nabla\Phi(u)$, relative to the reference measure $\mu_0$.
    ProposalMeasure: ABC interface for a Gaussian measure $\nu$ to propose from, together with its
        potential relative to $\mu_0$.

References:
    Cotter, Roberts, Stuart, White (2013). *MCMC Methods for Functions: Modifying Old Algorithms
    to Make Them Faster.* Statistical Science 28(3).

    Pinski, Simpson, Stuart, Weber (2015). *Algorithms for Kullback-Leibler Approximation of
    Probability Measures in Infinite Dimensions.* SIAM Journal on Scientific Computing 37(6),
    A2733-A2757.
"""

from abc import ABC, abstractmethod

import numpy as np


# ==================================================================================================
class TargetMeasure(ABC):
    r"""ABC interface for the actual target $\mu$: its potential $\Phi(u)$ and gradient
    $\nabla\Phi(u)$, relative to the (implicit, fixed) reference measure $\mu_0$,
    $\frac{d\mu}{d\mu_0} \propto \exp(-\Phi(u))$.

    Both methods are required unconditionally, mirroring
    [`Likelihood`][ls_bayesian.posterior.interfaces.Likelihood] and
    [`GaussianPrior`][ls_bayesian.posterior.interfaces.GaussianPrior] in the `posterior`
    subpackage. An implementation used only with a derivative-free algorithm (e.g.
    [`PCNAlgorithm`][ls_bayesian.mcmc.algorithms.pcn.PCNAlgorithm]) may implement
    `evaluate_gradient` by raising `NotImplementedError`.

    Methods:
        evaluate_cost: Evaluate $\Phi(u)$.
        evaluate_gradient: Evaluate $\nabla\Phi(u)$.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_cost(self, state: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        r"""Evaluate the potential $\Phi(u)$.

        Args:
            state (np.ndarray[tuple[int], np.dtype[np.float64]]): State $u$.

        Returns:
            float: Potential value, up to an additive constant.
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
class ProposalMeasure(ABC):
    r"""ABC interface for a Gaussian measure $\nu = \mathcal{N}(\bar u, C)$ to propose from,
    together with its potential $\Phi_\nu(u)$ relative to the reference measure $\mu_0$,
    $\frac{d\nu}{d\mu_0} \propto \exp(-\Phi_\nu(u))$.

    A single object therefore plays both roles that
    [`PCNAlgorithm`][ls_bayesian.mcmc.algorithms.pcn.PCNAlgorithm] needs from $\nu$: the Gaussian
    to draw the proposal from, and the correction term for using $\nu$ instead of $\mu_0$ in the
    acceptance probability (Pinski et al., 2015). They are not independent degrees of freedom --
    the correction is defined as the density of this same $\nu$ -- so they are combined in one
    interface rather than composed from two, mirroring how
    [`GaussianPrior`][ls_bayesian.posterior.interfaces.GaussianPrior] bundles its density with its
    sampling and covariance operations.

    [`MALAAlgorithm`][ls_bayesian.mcmc.algorithms.mala.MALAAlgorithm] uses the same interface for
    its `proposal_measure`, but requires $\nu = \mu_0$ exactly (`evaluate_cost` identically $0$):
    a generalized-approximation acceptance formula for MALA, analogous to Pinski et al. (2015) for
    pCN, has not been established. For pCN, choosing $\nu = \mu_0$ (`evaluate_cost` identically
    $0$) is the fallback that recovers the classical pCN sampler; how such a concrete measure is
    assembled from a prior or other reference measure is left to an adapter in the application
    layer, not fixed here.

    $\nu$ is one fixed Gaussian measure, computed once before sampling starts -- e.g. a Laplace
    approximation around the MAP estimate -- not a measure that adapts to the chain's current
    state. A proposal that instead uses a state-dependent, local Gaussian approximation (e.g. a
    Riemannian-manifold or Gauss-Newton-informed proposal) is a distinct, geometric-MCMC concern,
    out of scope for this interface.

    Methods:
        evaluate_cost: Evaluate $\Phi_\nu(u)$.
        apply_covariance_factorization: Apply the covariance factorization $\widehat{C}$.
        apply_covariance_operator: Apply the covariance operator $C$.

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
    def evaluate_cost(self, state: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        r"""Evaluate the potential $\Phi_\nu(u)$ relative to the reference measure $\mu_0$.

        Args:
            state (np.ndarray[tuple[int], np.dtype[np.float64]]): State $u$.

        Returns:
            float: Potential value, up to an additive constant. Identically $0$ if $\nu = \mu_0$.
        """

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
