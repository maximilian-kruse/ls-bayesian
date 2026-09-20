from typing import override

import numpy as np
import pytest

from tests.mcmc import helpers

pytestmark = pytest.mark.unit


# ==================================================================================================
def test_compute_step_accepts_when_uniform_draw_below_probability() -> None:
    algorithm_under_test = helpers.FakeMCMCAlgorithm(acceptance_probability=0.5)

    class FixedUniformGenerator(np.random.Generator):
        def __init__(self) -> None:
            super().__init__(np.random.PCG64(0))

        @override
        def uniform(self, *args: object, **kwargs: object) -> float:
            return 0.3

    _, accepted = algorithm_under_test.compute_step(np.zeros(2), FixedUniformGenerator())

    assert accepted


# --------------------------------------------------------------------------------------------------
def test_compute_step_rejects_when_acceptance_probability_zero() -> None:
    algorithm_under_test = helpers.FakeMCMCAlgorithm(acceptance_probability=0.0)
    rng = np.random.default_rng(0)

    _, accepted = algorithm_under_test.compute_step(np.zeros(2), rng)

    assert not accepted


# ==================================================================================================
def test_compute_step_returns_proposal_object_on_accept() -> None:
    class RecordingAlgorithm(helpers.FakeMCMCAlgorithm):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.last_proposal: np.ndarray | None = None

        @override
        def _create_proposal(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            self.last_proposal = super()._create_proposal(state, rng)
            return self.last_proposal

    algorithm_under_test = RecordingAlgorithm(acceptance_probability=1.0)
    rng = np.random.default_rng(0)

    new_state, accepted = algorithm_under_test.compute_step(np.zeros(2), rng)

    assert accepted
    assert new_state is algorithm_under_test.last_proposal


# --------------------------------------------------------------------------------------------------
def test_compute_step_returns_current_state_object_on_reject() -> None:
    algorithm_under_test = helpers.FakeMCMCAlgorithm(acceptance_probability=0.0)
    rng = np.random.default_rng(0)
    current_state = np.zeros(2)

    new_state, accepted = algorithm_under_test.compute_step(current_state, rng)

    assert not accepted
    assert new_state is current_state


# ==================================================================================================
def test_compute_step_calls_update_cache_exactly_once_with_correct_flag() -> None:
    algorithm_under_test = helpers.FakeMCMCAlgorithm(acceptance_probability=1.0)
    rng = np.random.default_rng(0)

    algorithm_under_test.compute_step(np.zeros(2), rng)

    assert algorithm_under_test.update_cache_calls == [True]


# --------------------------------------------------------------------------------------------------
def test_compute_step_calls_update_cache_with_false_on_reject() -> None:
    algorithm_under_test = helpers.FakeMCMCAlgorithm(acceptance_probability=0.0)
    rng = np.random.default_rng(0)

    algorithm_under_test.compute_step(np.zeros(2), rng)

    assert algorithm_under_test.update_cache_calls == [False]


# ==================================================================================================
def test_compute_step_calls_hooks_in_documented_order() -> None:
    call_order: list[str] = []

    class RecordingAlgorithm(helpers.FakeMCMCAlgorithm):
        @override
        def _create_proposal(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            call_order.append("_create_proposal")
            return super()._create_proposal(state, rng)

        @override
        def _evaluate_acceptance_probability(
            self, current_state: np.ndarray, proposal: np.ndarray
        ) -> float:
            call_order.append("_evaluate_acceptance_probability")
            return super()._evaluate_acceptance_probability(current_state, proposal)

        @override
        def _update_cache(self, accepted: bool) -> None:
            call_order.append("_update_cache")
            super()._update_cache(accepted)

    algorithm_under_test = RecordingAlgorithm(acceptance_probability=1.0)
    rng = np.random.default_rng(0)

    algorithm_under_test.compute_step(np.zeros(2), rng)

    assert call_order == [
        "_create_proposal",
        "_evaluate_acceptance_probability",
        "_update_cache",
    ]
