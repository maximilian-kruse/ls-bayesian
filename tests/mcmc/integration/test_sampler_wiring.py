import numpy as np
import pytest

from ls_bayesian.mcmc.algorithms.mala import MALAAlgorithm
from ls_bayesian.mcmc.output import AcceptanceQoI, RunningMeanStatistic, build
from ls_bayesian.mcmc.sampler import Sampler, SamplerSettings
from ls_bayesian.mcmc.storage import NumpyStorage
from tests.mcmc import helpers

pytestmark = pytest.mark.integration


# ==================================================================================================
def test_sampler_runs_mala_algorithm_end_to_end_with_numpy_storage_and_outputs() -> None:
    setup = helpers.create_quadratic_gaussian_setup(seed=100)
    algorithm = MALAAlgorithm(setup.target, setup.reference, step_width=0.2)
    storage = NumpyStorage()
    acceptance_output = build(AcceptanceQoI(), RunningMeanStatistic())
    settings = SamplerSettings(num_samples=200, store_interval=1, log_interval=200)
    sampler_under_test = Sampler(algorithm, storage=storage, outputs=[acceptance_output])
    initial_state = np.random.default_rng(101).standard_normal(helpers.STATE_DIM)

    sampler_under_test.run(initial_state, settings, seed=102)

    assert storage.values.shape == (200, helpers.STATE_DIM)
    assert 0.0 <= acceptance_output.value <= 1.0
