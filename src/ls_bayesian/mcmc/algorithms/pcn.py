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

from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.mcmc.algorithm import MCMCAlgorithm
from ls_bayesian.mcmc.model import MCMCModel


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
    measure $\mu_0 = \mathcal{N}(0, C)$. Pinski et al. (2015) show that the proposal can instead be
    drawn from any Gaussian measure $\nu = \mathcal{N}(\bar u_\nu, C_\nu)$ equivalent to $\mu_0$, at
    the cost of correcting the acceptance probability for the mismatch between $\nu$ and $\mu_0$,
    $\rho(u) = \log\frac{d\mu_0}{d\nu}(u)$. For two Gaussians this correction is closed-form,

    $$
    \rho(u) = \frac{1}{2}(u-\bar u_\nu)^T C_\nu^{-1} (u-\bar u_\nu) - \frac{1}{2} u^T C^{-1} u,
    $$

    up to an additive constant that cancels in the acceptance probability -- exactly the difference
    between `model.approximation`'s and `model.reference`'s own potentials
    ([`GaussianMeasure.evaluate_cost`][ls_bayesian.mcmc.measures.GaussianMeasure.evaluate_cost]),
    so this class computes it directly from the two measures rather than requiring a separate
    correction object. The potential of $\mu$
    *relative to $\nu$* -- what the acceptance probability of a proposal drawn from $\nu$ actually
    needs -- then follows from the chain rule for Radon-Nikodym derivatives,

    $$
    \frac{d\mu}{d\nu} = \frac{d\mu/d\mu_0}{d\nu/d\mu_0} \propto \exp\big(-(\Phi(u) - \rho(u))\big).
    $$

    Proposal (given current state $u$, proposal mean $\bar u_\nu$):

    $$
    v = \bar u_\nu + \sqrt{1-\delta^2}\,(u - \bar u_\nu) + \delta\, w, \qquad w \sim \mathcal
    N(0, C_\nu),
    $$

    with acceptance probability

    $$
    \alpha(u,v) = 1 \wedge \exp\big(\Phi_\nu(u) - \Phi_\nu(v)\big), \qquad \Phi_\nu(u) = \Phi(u) -
    \rho(u).
    $$

    With `model.reference`/`model.approximation` equal, or `model.approximation` not given (i.e.
    $\nu = \mu_0$), $\rho \equiv 0$ and $\Phi_\nu = \Phi$, recovering the classical pCN sampler
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
        model: MCMCModel,
        step_width: Annotated[Real, Is[lambda x: 0 < x < 1]],
        logger: BaseLogger | None = None,
    ) -> None:
        r"""Initialize the pCN algorithm.

        Args:
            model (MCMCModel): Target and the reference measure $\mu_0$, plus optionally an
                alternative Gaussian $\nu$ to propose from instead. If `model.approximation` is
                given, the proposal is drawn from it, with the correction $\rho$ computed from
                `model.reference`'s and `model.approximation`'s own potentials. Leave it out (or
                pass `model.reference` itself) to recover classical pCN exactly.
            step_width (Real): Proposal scaling $\delta \in (0,1)$. Smaller values give higher
                acceptance but slower exploration; larger values explore faster until acceptance
                deteriorates.
            logger (BaseLogger | None, optional): Logger for a one-time info message noting that
                the proposal is drawn from `model.approximation` when given. Nothing is logged if
                `None`. Defaults to `None`.
        """
        super().__init__(step_width)
        self._target_model = model.target
        self._reference_measure = model.reference
        self._proposal_measure = (
            model.approximation if model.approximation is not None else model.reference
        )
        if model.approximation is not None and logger is not None:
            logger.info(
                "model.approximation given: proposing from it, correcting relative to "
                "model.reference."
            )
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
        r"""Evaluate $\Phi_\nu(u) = \Phi(u) - \rho(u)$."""
        return self._target_model.evaluate_potential(state) - self._evaluate_reference_correction(
            state
        )

    # ----------------------------------------------------------------------------------------------
    def _evaluate_reference_correction(
        self, state: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        r"""Evaluate the correction $\rho(u) = \log\frac{d\mu_0}{d\nu}(u)$, up to an additive
        constant. This is `_proposal_measure.evaluate_cost(state) -
        _reference_measure.evaluate_cost(state)`, identically $0$ (without evaluating either
        measure's cost) when the two are the same object, in particular when
        `model.approximation` was not given, i.e. $\nu = \mu_0$."""
        if self._reference_measure is self._proposal_measure:
            return 0.0
        return self._proposal_measure.evaluate_cost(state) - self._reference_measure.evaluate_cost(
            state
        )
