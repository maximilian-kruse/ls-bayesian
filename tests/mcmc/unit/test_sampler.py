from pathlib import Path

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation

from ls_bayesian.common.logging import BaseLogger, LoggerSettings
from ls_bayesian.mcmc.output import AcceptanceQoI, IdentityStatistic, MCMCOutput
from ls_bayesian.mcmc.sampler import Sampler, SamplerSettings
from ls_bayesian.mcmc.storage import NumpyStorage
from tests.mcmc import helpers

pytestmark = pytest.mark.unit


# ==================================================================================================
def test_run_stores_and_logs_initial_state_regardless_of_intervals() -> None:
    algorithm_under_test = helpers.FakeMCMCAlgorithm(acceptance_probability=1.0)
    storage = NumpyStorage()
    settings = SamplerSettings(num_samples=5, store_interval=5, log_interval=5)
    sampler_under_test = Sampler(algorithm_under_test, storage=storage)

    sampler_under_test.run(np.zeros(1), settings)

    # store_interval == num_samples: only iteration 0 (0 % 5 == 0) and the last-iteration
    # coincidence would land at iteration 5, which never occurs since the loop stops at 4.
    assert storage.values.shape[0] == 1
    np.testing.assert_allclose(storage.values[0], np.zeros(1))


# ==================================================================================================
def test_run_calls_storage_store_only_at_store_interval() -> None:
    algorithm_under_test = helpers.FakeMCMCAlgorithm(
        acceptance_probability=1.0, proposal_increment=1.0
    )
    storage = NumpyStorage()
    settings = SamplerSettings(num_samples=5, store_interval=2, log_interval=5)
    sampler_under_test = Sampler(algorithm_under_test, storage=storage)

    sampler_under_test.run(np.zeros(1), settings)

    # Always-accepted, increment-by-1 states: iteration i has state == i. Stored at 0, 2, 4.
    np.testing.assert_allclose(storage.values, [[0.0], [2.0], [4.0]])


# ==================================================================================================
def test_run_updates_every_output_every_iteration_regardless_of_intervals() -> None:
    algorithm_under_test = helpers.FakeMCMCAlgorithm(acceptance_probability=1.0)
    output_under_test = MCMCOutput(AcceptanceQoI(), IdentityStatistic())
    settings = SamplerSettings(num_samples=5, store_interval=5, log_interval=5)
    sampler_under_test = Sampler(algorithm_under_test, outputs=[output_under_test])

    sampler_under_test.run(np.zeros(1), settings)

    assert output_under_test.all_values.shape[0] == 5


# ==================================================================================================
def test_run_logs_header_exactly_once_at_first_logged_iteration(tmp_path: Path) -> None:
    algorithm_under_test = helpers.FakeMCMCAlgorithm(acceptance_probability=1.0)
    logfile_path = tmp_path / "sampler.log"
    logger_settings = LoggerSettings(print_to_console=False, logfile_path=logfile_path)
    settings = SamplerSettings(num_samples=5, store_interval=5, log_interval=2)

    with BaseLogger(logger_settings, prefix="sampler") as logger:
        sampler_under_test = Sampler(algorithm_under_test, logger=logger)
        sampler_under_test.run(np.zeros(1), settings)

    log_content = logfile_path.read_text()
    assert log_content.count("Iteration") == 1


# ==================================================================================================
def test_run_flushes_storage_on_exception() -> None:
    algorithm_under_test = helpers.FailingAfterNStepsAlgorithm(fail_at_call=2)
    storage = helpers.RecordingStorage(NumpyStorage())
    settings = SamplerSettings(num_samples=5, store_interval=1, log_interval=1)
    sampler_under_test = Sampler(algorithm_under_test, storage=storage)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        sampler_under_test.run(np.zeros(1), settings)

    assert storage.flush_call_count == 1


# ==================================================================================================
def test_run_is_reproducible_for_same_seed_and_diverges_for_different_seed() -> None:
    settings = SamplerSettings(num_samples=20, store_interval=1, log_interval=20)

    def run_with_seed(seed: int) -> np.ndarray:
        algorithm = helpers.FakeMCMCAlgorithm(acceptance_probability=0.5)
        storage = NumpyStorage()
        Sampler(algorithm, storage=storage).run(np.zeros(1), settings, seed=seed)
        return storage.values

    first_run = run_with_seed(seed=0)
    second_run = run_with_seed(seed=0)
    third_run = run_with_seed(seed=1)

    np.testing.assert_array_equal(first_run, second_run)
    assert not np.array_equal(first_run, third_run)


# ==================================================================================================
def test_run_with_num_samples_one_never_calls_compute_step() -> None:
    algorithm_under_test = helpers.FailingAfterNStepsAlgorithm(fail_at_call=1)
    settings = SamplerSettings(num_samples=1)
    sampler_under_test = Sampler(algorithm_under_test)

    sampler_under_test.run(np.zeros(1), settings)

    assert algorithm_under_test.num_create_proposal_calls == 0


# ==================================================================================================
def test_run_rejects_store_interval_exceeding_num_samples() -> None:
    algorithm_under_test = helpers.FakeMCMCAlgorithm(acceptance_probability=1.0)
    settings = SamplerSettings(num_samples=3, store_interval=4)
    sampler_under_test = Sampler(algorithm_under_test)

    with pytest.raises(ValueError, match="store_interval"):
        sampler_under_test.run(np.zeros(1), settings)


# --------------------------------------------------------------------------------------------------
def test_run_rejects_log_interval_exceeding_num_samples() -> None:
    algorithm_under_test = helpers.FakeMCMCAlgorithm(acceptance_probability=1.0)
    settings = SamplerSettings(num_samples=3, log_interval=4)
    sampler_under_test = Sampler(algorithm_under_test)

    with pytest.raises(ValueError, match="log_interval"):
        sampler_under_test.run(np.zeros(1), settings)


# ==================================================================================================
@pytest.mark.parametrize(
    ("num_samples", "store_interval", "log_interval"),
    [(0, 1, 1), (-1, 1, 1), (1, 0, 1), (1, -1, 1), (1, 1, 0), (1, 1, -1)],
    ids=[
        "num_samples_zero",
        "num_samples_negative",
        "store_zero",
        "store_negative",
        "log_zero",
        "log_negative",
    ],
)
def test_settings_reject_non_positive_fields(
    num_samples: int, store_interval: int, log_interval: int
) -> None:
    with pytest.raises(BeartypeCallHintViolation):
        SamplerSettings(
            num_samples=num_samples, store_interval=store_interval, log_interval=log_interval
        )
