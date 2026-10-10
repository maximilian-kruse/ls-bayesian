from pathlib import Path

import numpy as np
import pytest

from ls_bayesian.mcmc.diagnostics import (
    AutocovarianceEstimate,
    compute_autocorrelation,
    compute_autocovariance,
    compute_effective_sample_size,
    estimate_effective_sample_size,
)
from ls_bayesian.mcmc.storage import ZarrStorage, open_zarr_samples
from tests.mcmc.helpers import generate_ar1_chain, reference_effective_sample_size

pytestmark = pytest.mark.unit


# ==================================================================================================
def direct_autocovariance(chain: np.ndarray, max_lag: int) -> np.ndarray:
    centered = chain - chain.mean(axis=0)
    num_samples = chain.shape[0]
    return np.stack(
        [(centered[: num_samples - lag] * centered[lag:]).sum(axis=0) / num_samples
         for lag in range(max_lag + 1)]
    )  # fmt: skip


# ==================================================================================================
@pytest.mark.parametrize("block_size", [1, 3, 7, 50, 1000])
def test_autocovariance_matches_direct_sum_for_every_block_size(block_size: int) -> None:
    chain = np.random.default_rng(0).normal(size=(50, 3))

    estimate = compute_autocovariance(chain, max_lag=10, block_size=block_size)

    np.testing.assert_allclose(
        estimate.autocovariance, direct_autocovariance(chain, 10), atol=1e-12
    )
    np.testing.assert_allclose(estimate.mean, chain.mean(axis=0), atol=1e-14)
    assert estimate.num_samples == 50


# --------------------------------------------------------------------------------------------------
def test_autocovariance_of_a_segment_ignores_the_rest_of_the_chain() -> None:
    chain = np.random.default_rng(1).normal(size=(60, 2))

    estimate = compute_autocovariance(chain, max_lag=5, start=10, stop=45, block_size=8)

    np.testing.assert_allclose(estimate.autocovariance, direct_autocovariance(chain[10:45], 5))
    assert estimate.num_samples == 35


# --------------------------------------------------------------------------------------------------
def test_autocovariance_at_lag_zero_is_the_biased_variance() -> None:
    chain = np.random.default_rng(2).normal(size=(40, 2))

    estimate = compute_autocovariance(chain, max_lag=0, block_size=6)

    np.testing.assert_allclose(estimate.autocovariance[0], chain.var(axis=0), atol=1e-14)


# --------------------------------------------------------------------------------------------------
def test_autocovariance_of_zarr_store_equals_that_of_numpy_array(tmp_path: Path) -> None:
    chain = np.random.default_rng(3).normal(size=(120, 4))
    storage = ZarrStorage(tmp_path / "samples.zarr", chunk_size=17, buffer_size=5)
    for sample in chain:
        storage.store(sample)
    storage.flush()

    from_disk = compute_autocovariance(
        open_zarr_samples(tmp_path / "samples.zarr"), 20, block_size=9
    )
    from_memory = compute_autocovariance(chain, 20, block_size=9)

    np.testing.assert_array_equal(from_disk.autocovariance, from_memory.autocovariance)
    np.testing.assert_array_equal(from_disk.mean, from_memory.mean)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"max_lag": 10, "start": 5, "stop": 5}, "non-empty"),
        ({"max_lag": 10, "stop": 99}, "inside the chain"),
        ({"max_lag": 10, "start": 25}, "max_lag must be in"),
        ({"max_lag": -1}, "max_lag must be in"),
        ({"max_lag": 3, "block_size": 0}, "block_size must be positive"),
    ],
)
def test_autocovariance_rejects_invalid_arguments(arguments: dict, message: str) -> None:
    chain = np.zeros((30, 2))

    with pytest.raises(ValueError, match=message):
        compute_autocovariance(chain, **arguments)


# --------------------------------------------------------------------------------------------------
def test_autocovariance_rejects_a_one_dimensional_chain() -> None:
    with pytest.raises(ValueError, match="num_components"):
        compute_autocovariance(np.zeros(30), max_lag=3)


# ==================================================================================================
@pytest.mark.parametrize("autoregression_coefficient", [0.0, 0.5, -0.5, 0.9])
def test_autocorrelation_time_of_ar1_chain_matches_analytic_value(
    autoregression_coefficient: float,
) -> None:
    chain = generate_ar1_chain(autoregression_coefficient, 200_000, 2, np.random.default_rng(4))
    analytic_time = (1 + autoregression_coefficient) / (1 - autoregression_coefficient)

    result = estimate_effective_sample_size(chain, max_lag=500, block_size=10_000)

    np.testing.assert_allclose(
        result.integrated_autocorrelation_time, analytic_time, rtol=0.08, atol=0.05
    )
    np.testing.assert_allclose(
        result.effective_sample_size, 200_000 / result.integrated_autocorrelation_time
    )
    assert not result.truncated.any()


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("num_chains", [1, 2, 3])
def test_effective_sample_size_matches_loop_reference_of_arviz_algorithm(num_chains: int) -> None:
    rng = np.random.default_rng(5)
    chains = np.stack(
        [generate_ar1_chain(0.7, 400, 3, rng) + 0.05 * index for index in range(num_chains)]
    )
    estimates = [compute_autocovariance(chain, max_lag=399) for chain in chains]

    result = compute_effective_sample_size(estimates)

    # For a truncated component, the reference stops at its own loop limit `num_draws - 3`, so
    # the two are only comparable where the lag-pair sum terminates by itself.
    comparable_components = np.flatnonzero(~result.truncated)
    assert comparable_components.size >= 2
    for component in comparable_components:
        np.testing.assert_allclose(
            result.effective_sample_size[component],
            reference_effective_sample_size(chains[:, :, component]),
            rtol=1e-10,
        )


# --------------------------------------------------------------------------------------------------
def test_effective_sample_size_does_not_depend_on_block_size() -> None:
    chain = generate_ar1_chain(0.6, 3000, 3, np.random.default_rng(6))

    reference = estimate_effective_sample_size(chain, max_lag=100, block_size=3000)
    streamed = estimate_effective_sample_size(chain, max_lag=100, block_size=37)

    np.testing.assert_allclose(streamed.effective_sample_size, reference.effective_sample_size)


# --------------------------------------------------------------------------------------------------
def test_burn_in_discards_the_leading_samples() -> None:
    rng = np.random.default_rng(7)
    chain = generate_ar1_chain(0.5, 2000, 2, rng)
    chain[:200] += 50.0  # transient

    with_burn_in = estimate_effective_sample_size(chain, max_lag=50, burn_in=200)
    on_stationary_part = estimate_effective_sample_size(chain[200:], max_lag=50)

    np.testing.assert_allclose(
        with_burn_in.effective_sample_size, on_stationary_part.effective_sample_size
    )


# --------------------------------------------------------------------------------------------------
def test_slowly_mixing_chain_is_flagged_as_truncated_for_short_max_lag() -> None:
    chain = generate_ar1_chain(0.99, 20_000, 1, np.random.default_rng(8))

    short = estimate_effective_sample_size(chain, max_lag=5)
    long = estimate_effective_sample_size(chain, max_lag=2000)

    assert short.truncated.all()
    assert not long.truncated.any()
    assert short.effective_sample_size[0] > long.effective_sample_size[0]


# --------------------------------------------------------------------------------------------------
def test_constant_component_gives_nan_and_does_not_affect_the_others() -> None:
    chain = generate_ar1_chain(0.5, 1000, 2, np.random.default_rng(9))
    chain[:, 1] = 3.0

    result = estimate_effective_sample_size(chain, max_lag=20)

    assert np.isfinite(result.effective_sample_size[0])
    assert np.isnan(result.effective_sample_size[1])
    assert not result.truncated[1]


# --------------------------------------------------------------------------------------------------
def test_chains_with_different_means_have_a_lower_effective_sample_size() -> None:
    rng = np.random.default_rng(10)
    stationary = generate_ar1_chain(0.3, 1000, 1, rng)
    drifting = stationary + np.linspace(0.0, 5.0, 1000)[:, None]

    stationary_result = estimate_effective_sample_size(stationary, max_lag=50)
    drifting_result = estimate_effective_sample_size(drifting, max_lag=50)

    assert (
        drifting_result.effective_sample_size[0] < 0.1 * stationary_result.effective_sample_size[0]
    )


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"burn_in": 100}, "burn_in must be in"),
        ({"burn_in": 95}, "need at least"),
        ({"max_lag": 60}, "max_lag must be in"),
    ],
)
def test_estimate_effective_sample_size_rejects_invalid_arguments(
    arguments: dict, message: str
) -> None:
    chain = np.random.default_rng(11).normal(size=(100, 2))

    with pytest.raises(ValueError, match=message):
        estimate_effective_sample_size(chain, **{"max_lag": 10, **arguments})


# --------------------------------------------------------------------------------------------------
def test_compute_effective_sample_size_rejects_inconsistent_estimates() -> None:
    chain = np.random.default_rng(12).normal(size=(40, 2))
    short = compute_autocovariance(chain[:20], max_lag=5)
    long = compute_autocovariance(chain, max_lag=5)

    with pytest.raises(ValueError, match="same number of samples"):
        compute_effective_sample_size([short, long])
    with pytest.raises(ValueError, match="At least one"):
        compute_effective_sample_size([])
    with pytest.raises(ValueError, match="at least 4 samples"):
        compute_effective_sample_size([compute_autocovariance(chain[:3], max_lag=1)])
    with pytest.raises(ValueError, match="lags 0 and 1"):
        compute_effective_sample_size([compute_autocovariance(chain, max_lag=0)])


# --------------------------------------------------------------------------------------------------
def test_estimate_dataclass_holds_what_it_was_given() -> None:
    estimate = AutocovarianceEstimate(np.zeros(2), np.ones((3, 2)), 10)

    assert estimate.num_samples == 10
    assert estimate.autocovariance.shape == (3, 2)


# ==================================================================================================
def test_autocorrelation_is_normalized_autocovariance_and_starts_at_one() -> None:
    chain = np.random.default_rng(13).normal(size=(80, 3))

    autocorrelation = compute_autocorrelation(chain, max_lag=12, block_size=9)

    expected = direct_autocovariance(chain, 12) / chain.var(axis=0)
    np.testing.assert_allclose(autocorrelation, expected, atol=1e-12)
    np.testing.assert_allclose(autocorrelation[0], 1.0)
    assert np.all(np.abs(autocorrelation) <= 1.0 + 1e-12)


# --------------------------------------------------------------------------------------------------
def test_autocorrelation_of_ar1_chain_decays_geometrically() -> None:
    coefficient = 0.7
    chain = generate_ar1_chain(coefficient, 200_000, 1, np.random.default_rng(14))

    autocorrelation = compute_autocorrelation(chain, max_lag=5, block_size=20_000)

    np.testing.assert_allclose(autocorrelation[:, 0], coefficient ** np.arange(6), atol=0.02)


# --------------------------------------------------------------------------------------------------
def test_autocorrelation_of_constant_component_is_nan() -> None:
    chain = np.random.default_rng(15).normal(size=(50, 2))
    chain[:, 0] = 2.0

    autocorrelation = compute_autocorrelation(chain, max_lag=4)

    assert np.isnan(autocorrelation[:, 0]).all()
    assert np.isfinite(autocorrelation[:, 1]).all()
