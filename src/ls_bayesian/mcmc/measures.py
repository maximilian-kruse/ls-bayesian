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

For [`PCNAlgorithm`][ls_bayesian.mcmc.algorithms.pcn.PCNAlgorithm] and the plain
[`MALAAlgorithm`][ls_bayesian.mcmc.algorithms.mala.MALAAlgorithm], $\mu_0$ itself is never
represented as an object: every potential above is already expressed relative to it implicitly,
and the only Gaussian measure ever instantiated is $\nu$, the one actually proposed from --
whether it happens to equal $\mu_0$ (`MALAAlgorithm`, and `PCNAlgorithm`'s trivial fallback) or
not (pCN's general case). [`ProposalMeasure`][ls_bayesian.mcmc.measures.ProposalMeasure]
represents exactly that object: a Gaussian measure to sample from, together with its potential
relative to $\mu_0$. [`TargetMeasure`][ls_bayesian.mcmc.measures.TargetMeasure] represents the
actual target $\mu$, which is only ever evaluated, never sampled from.

[`PMALAAlgorithm`][ls_bayesian.mcmc.algorithms.pmala.PMALAAlgorithm] implements a different
generalization (Beskos, Girolami, Lan, Farrell, Stuart, 2017): rather than
proposing from a Gaussian $\nu \neq \mu_0$ and correcting for the mismatch, it keeps $\Phi$
relative to the true $\mu_0$ throughout and instead uses a fixed Gaussian $\overline K \neq C$
purely as a *preconditioner* for the Langevin dynamics that generate the proposal. It reuses
[`ProposalMeasure`][ls_bayesian.mcmc.measures.ProposalMeasure] for $\overline K$ -- the same
covariance-sampling role `ProposalMeasure` already plays for pCN and plain MALA -- except that
`mean` and `evaluate_cost` are not read in this role (Beskos et al.'s drift provably does not
depend on $\overline K$'s mean, and there is no Pinski-style correction here since $\Phi$ is never
re-expressed relative to $\overline K$); only its covariance/precision actions matter. Because the
acceptance probability must compare $\overline K$ against $\mu_0$'s own covariance $C$
explicitly, this generalization needs $\mu_0$ represented explicitly too (unlike the pCN/plain-MALA
case above): [`ReferenceMeasure`][ls_bayesian.mcmc.measures.ReferenceMeasure] represents $\mu_0$'s
precision operator for this purpose. Beskos et al.'s formulas are stated for a centered $\mu_0 =
\mathcal N(0, C)$, so `ReferenceMeasure` has no mean, unlike `ProposalMeasure`.

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
    ReferenceMeasure: ABC interface for the target's actual (centered) reference measure $\mu_0$'s
        precision operator, for algorithms that use a preconditioner other than $\mu_0$'s own
        covariance.

References:
    Cotter, Roberts, Stuart, White (2013). *MCMC Methods for Functions: Modifying Old Algorithms
    to Make Them Faster.* Statistical Science 28(3).

    Pinski, Simpson, Stuart, Weber (2015). *Algorithms for Kullback-Leibler Approximation of
    Probability Measures in Infinite Dimensions.* SIAM Journal on Scientific Computing 37(6),
    A2733-A2757.

    Beskos, Girolami, Lan, Farrell, Stuart (2017). *Geometric MCMC for Infinite-Dimensional
    Inverse Problems.* Journal of Computational Physics 335.
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
    a generalized-approximation acceptance formula for MALA in the Pinski-et-al. sense (correcting
    for $\nu \neq \mu_0$ while keeping $\Phi$ relative to $\mu_0$) would additionally need the
    gradient of the correction potential, which this interface does not expose. For pCN, choosing
    $\nu = \mu_0$ (`evaluate_cost` identically $0$) is the fallback that recovers the classical
    pCN sampler; how such a concrete measure is assembled from a prior or other reference measure
    is left to an adapter in the application layer, not fixed here.

    [`PMALAAlgorithm`][ls_bayesian.mcmc.algorithms.pmala.PMALAAlgorithm]
    reuses this same interface for a third, distinct role: a fixed Gaussian $\overline K$ used
    purely to *precondition* Langevin dynamics (Beskos et al., 2017), while $\Phi$ stays relative
    to the true $\mu_0$ throughout (via a separate
    [`ReferenceMeasure`][ls_bayesian.mcmc.measures.ReferenceMeasure]). In that role, `mean` and
    `evaluate_cost` are never read -- the proposal provably does not depend on $\overline K$'s
    mean, and there is no correction potential to speak of since $\Phi$ is never re-expressed
    relative to $\overline K$ -- but `apply_precision_operator` is, since the acceptance
    probability must compare $\overline K$ against $C$ explicitly.

    $\nu$ is one fixed Gaussian measure, computed once before sampling starts -- e.g. a Laplace
    approximation around the MAP estimate -- not a measure that adapts to the chain's current
    state. A proposal that instead uses a state-dependent, local Gaussian approximation (e.g. a
    Riemannian-manifold or Gauss-Newton-informed proposal) is a distinct, geometric-MCMC concern,
    out of scope for this interface.

    Methods:
        evaluate_cost: Evaluate $\Phi_\nu(u)$.
        apply_covariance_factorization: Apply the covariance factorization $\widehat{C}$.
        apply_covariance_operator: Apply the covariance operator $C$.
        apply_precision_operator: Apply the precision operator $C^{-1}$. Only required in the
            preconditioner role.

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

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def apply_precision_operator(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply the precision operator $C^{-1}$.

        Only called when this measure plays the preconditioner role in
        [`PMALAAlgorithm`][ls_bayesian.mcmc.algorithms.pmala.PMALAAlgorithm];
        pCN and plain MALA never call it. An implementation used only for those algorithms may
        implement this method by raising `NotImplementedError`.

        Args:
            vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Vector to apply the precision
                operator to, same shape as `mean`.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Result, same shape as `mean`.
        """


# ==================================================================================================
class ReferenceMeasure(ABC):
    r"""ABC interface for the target's actual, centered reference measure $\mu_0 = \mathcal N(0,
    C)$: its precision operator.

    Used only by [`PMALAAlgorithm`]
    [ls_bayesian.mcmc.algorithms.pmala.PMALAAlgorithm], which
    preconditions Langevin dynamics with a fixed Gaussian $\overline K \neq C$ while keeping
    `target_model`'s potential relative to $\mu_0$ itself (Beskos, Girolami, Lan, Farrell, Stuart,
    2017). Beskos et al.'s formulas are stated for $\mu_0 = \mathcal N(0, C)$, i.e. a centered
    reference measure; this interface follows them directly rather than generalizing to a nonzero
    mean, so it exposes only the precision operator, not a mean. Its proposal and acceptance
    formulas need $\mu_0$'s score $C^{-1}u$ explicitly -- unlike
    [`ProposalMeasure`][ls_bayesian.mcmc.measures.ProposalMeasure], which is only ever *sampled*
    from and so never needs its own precision operator. Conversely, this interface has no
    covariance-sampling methods: $\mu_0$ is never itself sampled from here, only evaluated.

    Methods:
        apply_precision_operator: Apply the precision operator $C^{-1}$.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def apply_precision_operator(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply the precision operator $C^{-1}$.

        Args:
            vector (np.ndarray[tuple[int], np.dtype[np.float64]]): State-space vector to apply
                the precision operator to.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Result, same shape as `vector`.
        """
