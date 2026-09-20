r"""Preconditioned Crank-Nicolson (pCN) sampler, generalized to an arbitrary Gaussian
approximation measure to propose from.

Classes:
    PCNAlgorithm: Generalized pCN sampler.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Annotated, override

import numpy as np
from beartype.vale import Is

from ls_bayesian.mcmc.algorithm import MCMCAlgorithm
from ls_bayesian.mcmc.measures import ProposalMeasure, TargetMeasure


# ==================================================================================================
@dataclass(frozen=True)
class _PCNStateCache:
    r"""Potential $\Phi_\nu$ at a cached state, so `compute_step` does not evaluate the
    (typically expensive) target potential twice for the same state.

    Attributes:
        generalized_potential (float): $\Phi_\nu(u) = \Phi(u) - \Phi_\nu^\text{approx}(u)$ at the
            cached state.
    """

    generalized_potential: float


# ==================================================================================================
class PCNAlgorithm(MCMCAlgorithm):
    r"""Preconditioned Crank-Nicolson (pCN) sampler, generalized to an arbitrary Gaussian
    approximation measure to propose from (Pinski, Simpson, Stuart, Weber, 2015).

    The classical pCN sampler (Cotter et al., 2013) proposes from the target's own reference
    measure $\mu_0 = \mathcal{N}(\bar u, C)$. Pinski et al. (2015) show that the proposal can
    instead be drawn from any Gaussian measure $\nu = \mathcal{N}(\bar u_\nu, C_\nu)$ equivalent
    to $\mu_0$, at the cost of correcting the acceptance probability for the mismatch between
    $\nu$ and $\mu_0$. Both `target_model` (for $\mu$) and `proposal_measure` (for $\nu$)
    express their potential *relative to the same, implicit $\mu_0$*,

    $$
    \frac{d\mu}{d\mu_0} \propto \exp(-\Phi(u)), \qquad \frac{d\nu}{d\mu_0} \propto
    \exp(-\Phi_\nu^\text{approx}(u)),
    $$

    so that the potential of $\mu$ *relative to $\nu$* -- what the acceptance probability of a
    proposal drawn from $\nu$ actually needs -- follows from the chain rule for Radon-Nikodym
    derivatives without either model needing to know $\mu_0$ itself:

    $$
    \frac{d\mu}{d\nu} = \frac{d\mu/d\mu_0}{d\nu/d\mu_0} \propto \exp\big(-(\Phi(u) -
    \Phi_\nu^\text{approx}(u))\big).
    $$

    Proposal (given current state $u$, proposal mean $\bar u_\nu$):

    $$
    v = \bar u_\nu + \sqrt{1-\delta^2}\,(u - \bar u_\nu) + \delta\, w, \qquad w \sim \mathcal
    N(0, C_\nu),
    $$

    with acceptance probability

    $$
    \alpha(u,v) = 1 \wedge \exp\big(\Phi_\nu(u) - \Phi_\nu(v)\big), \qquad \Phi_\nu(u) = \Phi(u) -
    \Phi_\nu^\text{approx}(u).
    $$

    With `proposal_measure` a trivial wrapper around $\mu_0$ itself (i.e. $\nu = \mu_0$),
    $\Phi_\nu^\text{approx} \equiv 0$ and $\Phi_\nu = \Phi$, recovering the classical pCN sampler
    exactly.

    Methods:
        compute_step: Compute one step of MCMC.

    References:
        Cotter, Roberts, Stuart, White (2013). *MCMC Methods for Functions: Modifying Old
        Algorithms to Make Them Faster.* Statistical Science 28(3).

        Pinski, Simpson, Stuart, Weber (2015). *Algorithms for Kullback-Leibler Approximation of
        Probability Measures in Infinite Dimensions.* SIAM Journal on Scientific Computing 37(6),
        A2733-A2757.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        target_model: TargetMeasure,
        proposal_measure: ProposalMeasure,
        step_width: Annotated[Real, Is[lambda x: 0 < x < 1]],
    ) -> None:
        r"""Initialize the pCN algorithm.

        Args:
            target_model (TargetMeasure): Potential $\Phi$ of the actual target $\mu$, relative to
                the reference measure $\mu_0$.
            proposal_measure (ProposalMeasure): Gaussian measure $\nu$ the proposal is drawn from,
                together with its own potential relative to $\mu_0$. Use a measure wrapping
                $\mu_0$ itself, with `evaluate_cost` identically $0$, to recover classical pCN.
            step_width (Real): Proposal scaling $\delta \in (0,1)$. Smaller values give higher
                acceptance but slower exploration; larger values explore faster until acceptance
                deteriorates.
        """
        super().__init__(step_width)
        self._target_model = target_model
        self._proposal_measure = proposal_measure
        self._current_cache: _PCNStateCache | None = None
        self._pending_proposal_cache: _PCNStateCache | None = None

    # ----------------------------------------------------------------------------------------------
    @override
    def _create_proposal(
        self,
        state: np.ndarray[tuple[int], np.dtype[np.float64]],
        rng: np.random.Generator,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        random_increment = rng.normal(size=self._proposal_measure.random_vector_size)
        random_increment = self._proposal_measure.apply_covariance_factorization(random_increment)
        mean = self._proposal_measure.mean
        proposal = (
            mean
            + np.sqrt(1 - self._step_width**2) * (state - mean)
            + self._step_width * random_increment
        )
        return proposal

    # ----------------------------------------------------------------------------------------------
    @override
    def _evaluate_acceptance_probability(
        self,
        current_state: np.ndarray[tuple[int], np.dtype[np.float64]],
        proposal: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> float:
        if self._current_cache is None:
            self._current_cache = _PCNStateCache(
                generalized_potential=self._evaluate_generalized_potential(current_state)
            )
        potential_proposal = self._evaluate_generalized_potential(proposal)
        self._pending_proposal_cache = _PCNStateCache(generalized_potential=potential_proposal)
        return float(
            min(1.0, np.exp(self._current_cache.generalized_potential - potential_proposal))
        )

    # ----------------------------------------------------------------------------------------------
    @override
    def _update_cache(self, accepted: bool) -> None:
        if accepted:
            self._current_cache = self._pending_proposal_cache
        self._pending_proposal_cache = None

    # ----------------------------------------------------------------------------------------------
    def _evaluate_generalized_potential(
        self, state: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        r"""Evaluate $\Phi_\nu(u) = \Phi(u) - \Phi_\nu^\text{approx}(u)$."""
        return self._target_model.evaluate_cost(state) - self._proposal_measure.evaluate_cost(state)
