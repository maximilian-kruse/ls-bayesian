r"""Preconditioned Crank-Nicolson Langevin (pCNL / MALA) sampler.

Classes:
    MALAAlgorithm: pCNL / MALA sampler.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Annotated, override

import numpy as np
from beartype.vale import Is

from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.mcmc.algorithm import MCMCAlgorithm
from ls_bayesian.mcmc.measures import DifferentiableTargetMeasure
from ls_bayesian.mcmc.model import MCMCModel


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
    al., 2013, Sec. 6): `model.reference` and `model.target`'s gradient must therefore already
    be expressed on the same coefficient space (e.g. any finite-element mass-matrix weighting has
    to have been absorbed into both beforehand).

    Unlike [`PCNAlgorithm`][ls_bayesian.mcmc.algorithms.pcn.PCNAlgorithm], this class has no
    Pinski-et-al.-style generalized-approximation counterpart: `model.target`'s potential is
    always relative to `model.reference` itself, i.e. this class requires `model.reference` and
    ignores `model.approximation` if also given -- re-expressing $\Phi$ relative to a different
    Gaussian $\nu \neq \mu_0$, as `PCNAlgorithm` does, would additionally require the *gradient* of
    $\nu$'s correction potential $\rho$, which the acceptance probability below never computes,
    since pCN's does not need it. A *different* generalization to an alternative Gaussian does
    exist and is implemented separately: see
    [`PMALAAlgorithm`][ls_bayesian.mcmc.algorithms.pmala.PMALAAlgorithm],
    which keeps $\Phi$ relative to $\mu_0$ and instead uses the alternative Gaussian purely as a
    preconditioner (Beskos, Girolami, Lan, Farrell, Stuart, 2017).

    Methods:
        compute_step: Compute one step of MCMC.

    References:
        Cotter, Roberts, Stuart, White (2013). *MCMC Methods for Functions: Modifying Old
        Algorithms to Make Them Faster.* Statistical Science 28(3).
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        model: MCMCModel,
        step_width: Annotated[Real, Is[lambda x: x > 0]],
        logger: BaseLogger | None = None,
    ) -> None:
        r"""Initialize the MALA algorithm.

        Args:
            model (MCMCModel): Target (must be a `DifferentiableTargetMeasure`) and reference
                measure $\mu_0 = \mathcal N(\bar u, C)$, expressed on the same coefficient space as
                `model.target`'s gradient. `model.approximation`, if given, is ignored.
            step_width (Real): Step size $\delta > 0$. Smaller values increase acceptance; larger
                values explore faster until stability or acceptance deteriorates.
            logger (BaseLogger | None, optional): Logger for a one-time info message noting that
                `model.approximation` is ignored, when given alongside `model.reference`. Nothing
                is logged if `None`. Defaults to `None`.

        Raises:
            ValueError: If `model.target` is not a `DifferentiableTargetMeasure`.
        """
        if model.approximation is not None and logger is not None:
            logger.info(
                "model.approximation is given but ignored by MALAAlgorithm; using "
                "model.reference. Use PCNAlgorithm/PMALAAlgorithm to make use of both."
            )
        if not isinstance(model.target, DifferentiableTargetMeasure):
            raise ValueError(
                "MALAAlgorithm requires model.target to be a DifferentiableTargetMeasure."
            )
        super().__init__(step_width)
        self._target_model = model.target
        self._proposal_measure = model.reference
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
            self._current_cache = self._evaluate_state_and_cache(state)
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
            self._current_cache = self._evaluate_state_and_cache(current_state)
        proposal_cache = self._evaluate_state_and_cache(proposal)
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
    def _evaluate_state_and_cache(
        self, state: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> _MALAStateCache:
        r"""Evaluate $\Phi(u)$, $\nabla\Phi(u)$, and $C\nabla\Phi(u)$ at `state`."""
        gradient = self._target_model.evaluate_gradient(state)
        return _MALAStateCache(
            potential=self._target_model.evaluate_potential(state),
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
