from collections.abc import Callable

import numpy as np
import pytest

from ls_bayesian.mcmc.algorithm import MCMCAlgorithm
from ls_bayesian.mcmc.algorithms.mala import MALAAlgorithm
from ls_bayesian.mcmc.algorithms.pcn import PCNAlgorithm
from ls_bayesian.mcmc.algorithms.pmala import PMALAAlgorithm
from ls_bayesian.mcmc.output import AcceptanceQoI, RunningMeanStatistic, build
from ls_bayesian.mcmc.sampler import Sampler, SamplerSettings
from ls_bayesian.mcmc.storage import NumpyStorage
from tests.mcmc import helpers

pytestmark = pytest.mark.integration


# ==================================================================================================
def _build_pcn(setup: helpers.QuadraticGaussianSetup) -> MCMCAlgorithm:
    return PCNAlgorithm(setup.to_model(), step_width=0.2)


# --------------------------------------------------------------------------------------------------
def _build_mala(setup: helpers.QuadraticGaussianSetup) -> MCMCAlgorithm:
    return MALAAlgorithm(setup.to_model(), step_width=0.2)


# --------------------------------------------------------------------------------------------------
def _build_pmala(setup: helpers.QuadraticGaussianSetup) -> MCMCAlgorithm:
    return PMALAAlgorithm(setup.to_model(setup.reference), step_width=0.2)


# ==================================================================================================
@pytest.mark.parametrize(
    "build_algorithm", [_build_pcn, _build_mala, _build_pmala], ids=["pcn", "mala", "pmala"]
)
def test_sampler_runs_algorithm_end_to_end_with_numpy_storage_and_outputs(
    build_algorithm: Callable[[helpers.QuadraticGaussianSetup], MCMCAlgorithm],
) -> None:
    """`Sampler` only ever calls `algorithm.compute_step`, so it is largely algorithm-agnostic --
    but it is the only test exercising the real `Sampler`/`MCMCOutput`/`NumpyStorage` wiring
    end-to-end (as opposed to hand-driving `compute_step` directly, as the unit/algorithms and
    stationarity tests do), so it is worth running for each algorithm rather than just one."""
    setup = helpers.create_quadratic_gaussian_setup(seed=100)
    algorithm = build_algorithm(setup)
    storage = NumpyStorage()
    acceptance_output = build(AcceptanceQoI(), RunningMeanStatistic())
    settings = SamplerSettings(num_samples=200, store_interval=1, log_interval=200)
    sampler_under_test = Sampler(algorithm, storage=storage, outputs=[acceptance_output])
    initial_state = np.random.default_rng(101).standard_normal(helpers.STATE_DIM)

    sampler_under_test.run(initial_state, settings, seed=102)

    assert storage.values.shape == (200, helpers.STATE_DIM)
    assert 0.0 <= acceptance_output.value <= 1.0
