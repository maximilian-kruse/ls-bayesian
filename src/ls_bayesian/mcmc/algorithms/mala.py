r"""Preconditioned Crank-Nicolson Langevin (pCNL / MALA) sampler.

Classes:
    MALAAlgorithm: pCNL / MALA sampler.
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
class _MALAStateCache:
    r"""Quantities at a cached state, so `compute_step` does not re-evaluate the potential,
    gradient, and covariance action for the same state.

    Attributes:
        potential (float): $\Phi(u)$ at the cached state.
        gradient (np.ndarray[tuple[int], np.dtype[np.float64]]): $\nabla\Phi(u)$ at the cached
            state.
        covariance_gradient_action (np.ndarray[tuple[int], np.dtype[np.float64]]): $C\nabla\Phi(u)$
            at the cached state.
    """

    potential: float
    gradient: np.ndarray[tuple[int], np.dtype[np.float64]]
    covariance_gradient_action: np.ndarray[tuple[int], np.dtype[np.float64]]


# ==================================================================================================
class MALAAlgorithm(MCMCAlgorithm):
    r"""Preconditioned Crank-Nicolson Langevin (pCNL / MALA) sampler.

    Implements the function-space MALA variant (Cotter et al., 2013) using a Gaussian reference
    measure $\mathcal{N}(\bar u, C)$.

    Proposal (given current state $u$):

    $$
    v = \bar u + \frac{2-\delta}{2+\delta}(u - \bar u) - \frac{2\delta}{2+\delta} C\nabla\Phi(u)
        + \frac{\sqrt{8\delta}}{2+\delta} w, \qquad w \sim \mathcal{N}(0, C).
    $$

    with acceptance probability $\alpha(u,v) = 1 \wedge \exp(\varrho(u,v) - \varrho(v,u))$, where

    $$
    \varrho(u,v) = \Phi(u) + \frac{1}{2}(v-u,\nabla\Phi(u)) + \frac{\delta}{4}(u+v-2\bar
        u,\nabla\Phi(u)) + \frac{\delta}{4}(\nabla\Phi(u), C\nabla\Phi(u)).
    $$

    Inner products above are the ordinary Euclidean dot product of coefficient vectors (Cotter et
    al., 2013, Sec. 6): `proposal_measure` and `target_model`'s gradient must therefore already
    be expressed on the same coefficient space (e.g. any finite-element mass-matrix weighting has
    to have been absorbed into both beforehand).

    Unlike [`PCNAlgorithm`][ls_bayesian.mcmc.algorithms.pcn.PCNAlgorithm], this class has no
    generalized-approximation counterpart: `target_model`'s potential is always relative to
    `proposal_measure` itself, i.e. `proposal_measure` must be the target's actual reference
    measure $\mu_0$ (`evaluate_cost` identically $0$). A generalized-approximation acceptance
    formula for MALA, analogous to Pinski et al. (2015) for pCN, has not been established.

    Methods:
        compute_step: Compute one step of MCMC.

    References:
        Cotter, Roberts, Stuart, White (2013). *MCMC Methods for Functions: Modifying Old
        Algorithms to Make Them Faster.* Statistical Science 28(3).
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        target_model: TargetMeasure,
        proposal_measure: ProposalMeasure,
        step_width: Annotated[Real, Is[lambda x: x > 0]],
    ) -> None:
        r"""Initialize the MALA algorithm.

        Args:
            target_model (TargetMeasure): Potential $\Phi$ of the actual target and its gradient,
                relative to `proposal_measure`.
            proposal_measure (ProposalMeasure): The target's actual reference measure $\mu_0 =
                \mathcal N(\bar u, C)$ (`evaluate_cost` identically $0$), expressed on the same
                coefficient space as `target_model`'s gradient.
            step_width (Real): Step size $\delta > 0$. Smaller values increase acceptance; larger
                values explore faster until stability or acceptance deteriorates.
        """
        super().__init__(step_width)
        self._target_model = target_model
        self._proposal_measure = proposal_measure
        self._current_cache: _MALAStateCache | None = None
        self._pending_proposal_cache: _MALAStateCache | None = None

    # ----------------------------------------------------------------------------------------------
    @override
    def _create_proposal(
        self,
        state: np.ndarray[tuple[int], np.dtype[np.float64]],
        rng: np.random.Generator,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        if self._current_cache is None:
            self._current_cache = self._evaluate_state_cache(state)
        random_increment = rng.normal(size=self._proposal_measure.random_vector_size)
        random_increment = self._proposal_measure.apply_covariance_factorization(random_increment)
        mean = self._proposal_measure.mean
        proposal = mean + (
            (2 - self._step_width) * (state - mean)
            - 2 * self._step_width * self._current_cache.covariance_gradient_action
            + np.sqrt(8 * self._step_width) * random_increment
        ) / (2 + self._step_width)
        return proposal

    # ----------------------------------------------------------------------------------------------
    @override
    def _evaluate_acceptance_probability(
        self,
        current_state: np.ndarray[tuple[int], np.dtype[np.float64]],
        proposal: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> float:
        if self._current_cache is None:
            self._current_cache = self._evaluate_state_cache(current_state)
        proposal_cache = self._evaluate_state_cache(proposal)
        self._pending_proposal_cache = proposal_cache
        mean = self._proposal_measure.mean
        log_transition_current_to_proposal = self._evaluate_log_transition_density(
            current_state, proposal, self._current_cache, mean
        )
        log_transition_proposal_to_current = self._evaluate_log_transition_density(
            proposal, current_state, proposal_cache, mean
        )
        exponent = log_transition_current_to_proposal - log_transition_proposal_to_current
        return float(min(1.0, np.exp(exponent)))

    # ----------------------------------------------------------------------------------------------
    @override
    def _update_cache(self, accepted: bool) -> None:
        if accepted:
            self._current_cache = self._pending_proposal_cache
        self._pending_proposal_cache = None

    # ----------------------------------------------------------------------------------------------
    def _evaluate_state_cache(
        self, state: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> _MALAStateCache:
        r"""Evaluate $\Phi(u)$, $\nabla\Phi(u)$, and $C\nabla\Phi(u)$ at `state`."""
        gradient = self._target_model.evaluate_gradient(state)
        return _MALAStateCache(
            potential=self._target_model.evaluate_cost(state),
            gradient=gradient,
            covariance_gradient_action=self._proposal_measure.apply_covariance_operator(gradient),
        )

    # ----------------------------------------------------------------------------------------------
    def _evaluate_log_transition_density(
        self,
        state: np.ndarray[tuple[int], np.dtype[np.float64]],
        other_state: np.ndarray[tuple[int], np.dtype[np.float64]],
        state_cache: _MALAStateCache,
        mean: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> float:
        r"""Evaluate $\varrho(u,v) = \Phi(u) + \frac{1}{2}(v-u,\nabla\Phi(u)) +
        \frac{\delta}{4}(u+v-2\bar u,\nabla\Phi(u)) + \frac{\delta}{4}(\nabla\Phi(u),
        C\nabla\Phi(u))$, with `state` playing the role of $u$ and `other_state` of $v$."""
        return (
            state_cache.potential
            + 0.5 * np.dot(other_state - state, state_cache.gradient)
            + self._step_width / 4 * np.dot(state + other_state - 2 * mean, state_cache.gradient)
            + self._step_width
            / 4
            * np.dot(state_cache.gradient, state_cache.covariance_gradient_action)
        )
