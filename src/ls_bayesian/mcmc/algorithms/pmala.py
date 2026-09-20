r"""Preconditioned MALA sampler with a fixed, non-prior Gaussian preconditioner.

Classes:
    PMALAAlgorithm: MALA sampler preconditioned by a fixed Gaussian
        $\overline K \neq C$.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Annotated, override

import numpy as np
from beartype.vale import Is

from ls_bayesian.mcmc.algorithm import MCMCAlgorithm
from ls_bayesian.mcmc.measures import ProposalMeasure, ReferenceMeasure, TargetMeasure


# ==================================================================================================
@dataclass(frozen=True)
class _PMALAStateCache:
    r"""Quantities at a cached state, so `compute_step` does not re-evaluate the potential,
    gradient, and preconditioner action for the same state.

    Attributes:
        potential (float): $\Phi(u)$ at the cached state.
        score (np.ndarray[tuple[int], np.dtype[np.float64]]): $s(u) = \nabla\Phi(u) + C^{-1}u$ at
            the cached state.
        preconditioned_score (np.ndarray[tuple[int], np.dtype[np.float64]]): $\overline K s(u)$
            at the cached state.
    """

    potential: float
    score: np.ndarray[tuple[int], np.dtype[np.float64]]
    preconditioned_score: np.ndarray[tuple[int], np.dtype[np.float64]]


# ==================================================================================================
class PMALAAlgorithm(MCMCAlgorithm):
    r"""MALA sampler preconditioned by a fixed Gaussian $\overline K \neq C$.

    Implements the state-independent special case of $\infty$-mMALA (Beskos, Girolami, Lan,
    Farrell, Stuart, 2017, Sec. 3.2-3.3, Algorithm 3.7): the general method there preconditions
    Langevin dynamics with a location-specific Gaussian $K(u)$ while keeping the target's
    potential $\Phi$ relative to the true, centered reference measure $\mu_0 = \mathcal N(0, C)$.
    Fixing $K(u) \equiv \overline K$ (e.g. a Laplace approximation computed once at the MAP,
    rather than re-linearized at every state) makes the Carleman-Fredholm determinant that
    otherwise appears in the acceptance probability (their Theorem 3.5) state-independent, so it
    cancels exactly in the Metropolis-Hastings ratio. `target_model`'s potential and gradient stay
    relative to the true $\mu_0$ throughout -- unlike
    [`MALAAlgorithm`][ls_bayesian.mcmc.algorithms.mala.MALAAlgorithm] with a non-prior
    `proposal_measure`, this does not require `target_model` to be re-expressed relative to
    $\overline K$, since $\overline K$ never plays the role of a measure proposed from and
    corrected for, only that of a preconditioner.

    Proposal (given current state $u$, score $s(u) = \nabla\Phi(u) + C^{-1}u$):

    $$
    v = u - \frac{2\delta}{2+\delta} \overline K s(u) + \frac{\sqrt{8\delta}}{2+\delta} w, \qquad
        w \sim \mathcal N(0, \overline K).
    $$

    $\overline K$'s own mean plays no role: it cancels identically out of the drift for any
    choice, which is why `preconditioner` uses [`ProposalMeasure`]
    [ls_bayesian.mcmc.measures.ProposalMeasure] purely for its covariance/precision actions
    (`mean` and `evaluate_cost` are never read here).

    With $\rho = \frac{2-\delta}{2+\delta}$, $g(u) = u - \overline K s(u)$, and innovation
    $w(u,v) = (v - \rho u) / \sqrt{1-\rho^2}$, the acceptance probability is $\alpha(u,v) = 1
    \wedge \exp(\varrho(u,v) - \varrho(v,u))$, where

    $$
    \varrho(u,v) = \Phi(u) + \frac{1}{2}\Big\langle \sqrt{\delta/2}\, g(u) - w(u,v),\
        \overline K^{-1}\big(\sqrt{\delta/2}\, g(u) - w(u,v)\big) \Big\rangle - \frac{1}{2}
        \big\langle w(u,v), C^{-1} w(u,v) \big\rangle.
    $$

    Inner products above are the ordinary Euclidean dot product of coefficient vectors (as in
    [`MALAAlgorithm`][ls_bayesian.mcmc.algorithms.mala.MALAAlgorithm]): `target_model`'s gradient,
    `reference_measure`, and `preconditioner` must all be expressed on the same coefficient space.

    When `preconditioner` wraps $C$ itself (`apply_covariance_operator`/
    `apply_covariance_factorization`/`apply_precision_operator` all acting as $\mu_0$'s own
    covariance/precision), $\varrho$ reduces exactly to `MALAAlgorithm`'s $\varrho$ and the two
    algorithms coincide (Beskos et al.'s Remark 3.8, "$\infty$-MALA and $\infty$-mMALA coincide
    when $K(u) \equiv C$"); this reduction was checked analytically when deriving the formula
    above, and is exercised directly by this class's tests. For the fully location-specific
    $K(u)$, or for a Riemannian-manifold/Gauss-Newton preconditioner, see $\infty$-mMALA in the
    reference below -- out of scope here, since its acceptance probability needs the
    (state-dependent, generally intractable) Carleman-Fredholm determinant that this class's fixed
    $\overline K$ lets it avoid.

    Methods:
        compute_step: Compute one step of MCMC.

    References:
        Beskos, Girolami, Lan, Farrell, Stuart (2017). *Geometric MCMC for Infinite-Dimensional
        Inverse Problems.* Journal of Computational Physics 335, 327-351.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        target_model: TargetMeasure,
        reference_measure: ReferenceMeasure,
        preconditioner: ProposalMeasure,
        step_width: Annotated[Real, Is[lambda x: x > 0]],
    ) -> None:
        r"""Initialize the preconditioned MALA algorithm.

        Args:
            target_model (TargetMeasure): Potential $\Phi$ of the actual target and its gradient,
                relative to the true, centered reference measure $\mu_0$.
            reference_measure (ReferenceMeasure): The target's actual, centered reference measure
                $\mu_0 = \mathcal N(0, C)$, expressed on the same coefficient space as
                `target_model`'s gradient.
            preconditioner (ProposalMeasure): The fixed Gaussian $\overline K$ used to precondition
                the proposal, expressed on the same coefficient space. Only its covariance and
                precision actions are used; `mean` and `evaluate_cost` are ignored. Use a
                preconditioner wrapping $C$ itself to recover plain MALA (prefer
                [`MALAAlgorithm`][ls_bayesian.mcmc.algorithms.mala.MALAAlgorithm] directly in that
                case, which needs neither `reference_measure` nor a precision operator).
            step_width (Real): Step size $\delta > 0$. Smaller values increase acceptance; larger
                values explore faster until stability or acceptance deteriorates.
        """
        super().__init__(step_width)
        self._target_model = target_model
        self._reference_measure = reference_measure
        self._preconditioner = preconditioner
        self._current_cache: _PMALAStateCache | None = None
        self._pending_proposal_cache: _PMALAStateCache | None = None

    # ----------------------------------------------------------------------------------------------
    @override
    def _create_proposal(
        self,
        state: np.ndarray[tuple[int], np.dtype[np.float64]],
        rng: np.random.Generator,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        if self._current_cache is None:
            self._current_cache = self._evaluate_state_cache(state)
        random_increment = rng.normal(size=self._preconditioner.random_vector_size)
        random_increment = self._preconditioner.apply_covariance_factorization(random_increment)
        proposal = (
            state
            - (2 * self._step_width)
            / (2 + self._step_width)
            * self._current_cache.preconditioned_score
            + np.sqrt(8 * self._step_width) / (2 + self._step_width) * random_increment
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
            self._current_cache = self._evaluate_state_cache(current_state)
        proposal_cache = self._evaluate_state_cache(proposal)
        self._pending_proposal_cache = proposal_cache
        log_transition_current_to_proposal = self._evaluate_log_transition_potential(
            current_state, proposal, self._current_cache
        )
        log_transition_proposal_to_current = self._evaluate_log_transition_potential(
            proposal, current_state, proposal_cache
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
    ) -> _PMALAStateCache:
        r"""Evaluate $\Phi(u)$, $s(u) = \nabla\Phi(u) + C^{-1}u$, and $\overline K s(u)$ at
        `state`."""
        gradient = self._target_model.evaluate_gradient(state)
        precision_weighted_state = self._reference_measure.apply_precision_operator(state)
        score = gradient + precision_weighted_state
        return _PMALAStateCache(
            potential=self._target_model.evaluate_cost(state),
            score=score,
            preconditioned_score=self._preconditioner.apply_covariance_operator(score),
        )

    # ----------------------------------------------------------------------------------------------
    def _evaluate_log_transition_potential(
        self,
        state: np.ndarray[tuple[int], np.dtype[np.float64]],
        other_state: np.ndarray[tuple[int], np.dtype[np.float64]],
        state_cache: _PMALAStateCache,
    ) -> float:
        r"""Evaluate $\varrho(u,v) = \Phi(u) + \frac{1}{2}\langle \sqrt{\delta/2}\, g(u) - w,
        \overline K^{-1}(\sqrt{\delta/2}\, g(u) - w)\rangle - \frac{1}{2}\langle w, C^{-1}
        w\rangle$, with `state` playing the role of $u$ and `other_state` of $v$."""
        g = state - state_cache.preconditioned_score
        rho = (2 - self._step_width) / (2 + self._step_width)
        sqrt_one_minus_rho_squared = np.sqrt(8 * self._step_width) / (2 + self._step_width)
        innovation = (other_state - rho * state) / sqrt_one_minus_rho_squared
        correction = np.sqrt(self._step_width / 2) * g - innovation
        return (
            state_cache.potential
            + 0.5 * np.dot(correction, self._preconditioner.apply_precision_operator(correction))
            - 0.5 * np.dot(innovation, self._reference_measure.apply_precision_operator(innovation))
        )
