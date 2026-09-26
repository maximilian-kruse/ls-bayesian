r"""Template-method driver for a single Metropolis-Hastings transition on function space.

Classes:
    MCMCAlgorithm: ABC template-method driver for one Metropolis-Hastings step.
"""

from abc import ABC, abstractmethod
from numbers import Real
from typing import Annotated

import numpy as np
from beartype.vale import Is


# ==================================================================================================
class MCMCAlgorithm(ABC):
    r"""ABC template-method driver for one Metropolis-Hastings step on function space.

    Subclasses implement proposal generation and the acceptance probability; this class owns the
    generic propose/accept-reject step shared by every Metropolis-Hastings algorithm, together
    with `_update_cache`, a hook subclasses use to carry expensive per-state quantities (e.g. a
    potential or gradient evaluation) from one call to `compute_step` to the next instead of
    recomputing them.

    Methods:
        compute_step: Advance the chain by one Metropolis-Hastings step.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, step_width: Annotated[Real, Is[lambda x: x > 0]]) -> None:
        r"""Initialize the algorithm.

        Args:
            step_width (Real): Proposal step size $\delta > 0$; the precise admissible range and
                interpretation depend on the concrete algorithm.
        """
        self._step_width = step_width

    # ----------------------------------------------------------------------------------------------
    def compute_step(
        self,
        current_state: np.ndarray[tuple[int], np.dtype[np.float64]],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.float64]], bool]:
        """Advance the Markov chain by one Metropolis-Hastings step.

        Args:
            current_state (np.ndarray[tuple[int], np.dtype[np.float64]]): Current chain state.
            rng (np.random.Generator): Random number generator.

        Returns:
            tuple[np.ndarray[tuple[int], np.dtype[np.float64]], bool]: New state (the proposal if
                accepted, `current_state` otherwise) and whether the proposal was accepted.
        """
        proposal = self._create_proposal(current_state, rng)
        acceptance_probability = self._evaluate_acceptance_probability(current_state, proposal)
        accepted = bool(rng.uniform() < acceptance_probability)
        new_state = proposal if accepted else current_state
        self._update_cache(accepted)
        return new_state, accepted

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def _create_proposal(
        self,
        state: np.ndarray[tuple[int], np.dtype[np.float64]],
        rng: np.random.Generator,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Create a proposal given the current state and RNG.

        Args:
            state (np.ndarray[tuple[int], np.dtype[np.float64]]): Current state.
            rng (np.random.Generator): Random number generator.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Proposed next state, same shape as
                `state`.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def _evaluate_acceptance_probability(
        self,
        current_state: np.ndarray[tuple[int], np.dtype[np.float64]],
        proposal: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> float:
        """Evaluate the Metropolis-Hastings acceptance probability for a proposal, in $[0, 1]$.

        Args:
            current_state (np.ndarray[tuple[int], np.dtype[np.float64]]): Current state.
            proposal (np.ndarray[tuple[int], np.dtype[np.float64]]): Proposed state.

        Returns:
            float: Acceptance probability.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def _update_cache(self, accepted: bool) -> None:
        """Update any per-state quantities cached across calls to `compute_step`, e.g. promoting
        proposal-state quantities to current-state quantities on acceptance.

        Args:
            accepted (bool): Whether the proposal evaluated in the step just completed was
                accepted.
        """
