r"""Convergence diagnostics for real-valued MCMC chains: autocovariance, ESS, autocorrelation time.

The estimators follow Vehtari et al. (2021), "Rank-normalization, folding, and localization: An
improved $\widehat R$ for assessing convergence of MCMC", and are the same as those behind
`arviz.ess(..., method="mean")`: the autocovariance of each (half-)chain is combined with the
between-chain variance into the autocorrelation estimate $\hat\rho_k$, which is summed up to
Geyer's (1992) initial monotone positive sequence of lag pairs,

$$
\hat\tau = -1 + 2\sum_{k=0}^{2J-1} \hat\rho_k + \hat\rho_{2J}^{+},
\qquad
\mathrm{ESS} = \frac{M N}{\hat\tau},
$$

for $M$ chains of length $N$ each. A single chain is split into two halves ($M = 2$), as `arviz`
does, so that the between-chain variance also detects a drifting chain.

In contrast to `arviz`, which needs the whole chain in memory (an `xarray` object), the
autocovariance is computed by streaming over blocks of samples, so that a disk-backed chain, e.g.
a Zarr array from [`ZarrStorage`][ls_bayesian.mcmc.storage.ZarrStorage], never has to fit in
memory. Only lags up to a given `max_lag` are accumulated; if the lag-pair sum has not turned
non-positive by then, the estimate is flagged as `truncated`.

Not provided: the rank-normalized bulk/tail ESS and $\widehat R$ of Vehtari et al. They need
global ranks of every component, which cannot be streamed.

Classes:
    ChainStore: Protocol for chains sliceable along their first axis.
    AutocovarianceEstimate: Mean and autocovariance of one chain segment.
    EffectiveSampleSizeResult: Effective sample size and autocorrelation time per component.

Functions:
    compute_autocovariance: Mean and autocovariance of a chain segment, streamed over blocks.
    compute_autocorrelation: Autocorrelation function of a chain segment.
    compute_effective_sample_size: Effective sample size from the autocovariances of chains.
    estimate_effective_sample_size: Effective sample size of one chain, split into two halves.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# Number of samples read at once when streaming over a chain.
DEFAULT_BLOCK_SIZE = 1000

# Smallest number of samples per (half-)chain for which the estimators are defined: they use the
# lag-pair sums and `N - 1` in the variance. Same as the minimum `arviz` accepts.
MINIMUM_NUM_SAMPLES_PER_CHAIN = 4


# ==================================================================================================
@runtime_checkable
class ChainStore(Protocol):
    """Chain read in slices along its first axis, e.g. a `numpy` or `zarr` array.

    Runtime-checkable, since the package type-checks its public functions with `beartype`.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape `(num_samples, num_components)`."""
        ...

    def __getitem__(self, key: slice, /) -> np.ndarray:
        """Return the samples in `key`."""
        ...


# ==================================================================================================
@dataclass(frozen=True)
class AutocovarianceEstimate:
    r"""Mean and autocovariance of one chain segment, per component.

    Attributes:
        mean (np.ndarray): Sample mean $\bar x$ of each component, shape `(num_components,)`.
        autocovariance (np.ndarray): Biased autocovariance estimate
            $\hat c_k = \frac{1}{N}\sum_{t=0}^{N-1-k}(x_t - \bar x)(x_{t+k} - \bar x)$ for lags
            $k = 0, \dots, K$, shape `(K + 1, num_components)`. The factor $1/N$ (not
            $1/(N - k)$) keeps the estimated autocorrelation sequence positive semidefinite.
        num_samples (int): Number of samples $N$ of the segment.
    """

    mean: np.ndarray[tuple[int], np.dtype[np.float64]]
    autocovariance: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    num_samples: int


# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EffectiveSampleSizeResult:
    r"""Effective sample size and integrated autocorrelation time per component.

    Attributes:
        effective_sample_size (np.ndarray): $\mathrm{ESS} = M N / \hat\tau$, shape
            `(num_components,)`. `nan` for a component that is constant.
        integrated_autocorrelation_time (np.ndarray): $\hat\tau$, the number of correlated
            samples that carry the information of one independent sample, shape
            `(num_components,)`. It is bounded below by $1/\log_{10}(M N)$ (the clamp of `arviz`,
            which keeps the ESS finite for strongly anticorrelated chains).
        truncated (np.ndarray): Boolean, shape `(num_components,)`. `True` if the lag-pair sum
            had not turned non-positive within the available lags, so that $\hat\tau$ misses the
            contribution of longer lags and the ESS is an upper bound; repeat with a larger
            `max_lag`.
        num_samples (int): Total number of samples $M N$ of all chains.
    """

    effective_sample_size: np.ndarray[tuple[int], np.dtype[np.float64]]
    integrated_autocorrelation_time: np.ndarray[tuple[int], np.dtype[np.float64]]
    truncated: np.ndarray[tuple[int], np.dtype[np.bool_]]
    num_samples: int


# ==================================================================================================
def compute_autocovariance(
    chain: ChainStore,
    max_lag: int,
    start: int = 0,
    stop: int | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> AutocovarianceEstimate:
    r"""Compute the mean and autocovariance of `chain[start:stop]`, streamed over blocks.

    Two passes over the segment: the first for the mean, the second for the products of the
    centered samples at lags $0, \dots, K$. Between blocks, the last $K$ centered samples are
    carried over, so that the result does not depend on `block_size`. Memory is
    $O((\text{block\_size} + K)\, d)$ and independent of the chain length; the work is
    $O(N K d)$.

    Args:
        chain (ChainStore): Chain, shape `(num_samples, num_components)`.
        max_lag (int): Largest lag $K$ to accumulate; must be smaller than the segment length.
        start (int): First sample of the segment. Defaults to `0`.
        stop (int | None): End of the segment (exclusive). Defaults to the end of the chain.
        block_size (int): Number of samples read at once. Defaults to `DEFAULT_BLOCK_SIZE`.

    Returns:
        AutocovarianceEstimate: Mean and autocovariance of the segment.

    Raises:
        ValueError: If the chain is not 2D, the segment is not inside the chain, `max_lag` is
            negative or not smaller than the segment length, or `block_size` is not positive.
    """
    if len(chain.shape) != 2:
        raise ValueError(f"chain must have shape (num_samples, num_components), got {chain.shape}.")
    num_chain_samples, num_components = chain.shape
    stop = num_chain_samples if stop is None else stop
    if not 0 <= start < stop <= num_chain_samples:
        raise ValueError(
            f"The segment [{start}, {stop}) must be non-empty and inside the chain of "
            f"{num_chain_samples} samples."
        )
    num_samples = stop - start
    if not 0 <= max_lag < num_samples:
        raise ValueError(f"max_lag must be in [0, {num_samples}), got {max_lag}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    block_starts = range(start, stop, block_size)

    def read_block(block_start: int) -> np.ndarray:
        block = chain[block_start : min(block_start + block_size, stop)]
        return np.asarray(block, dtype=np.float64)

    sample_sum = np.zeros(num_components)
    for block_start in block_starts:
        sample_sum += read_block(block_start).sum(axis=0)
    mean = sample_sum / num_samples

    lagged_product_sums = np.zeros((max_lag + 1, num_components))
    carried_samples = np.empty((0, num_components))
    for block_start in block_starts:
        centered_block = read_block(block_start) - mean
        extended = np.concatenate([carried_samples, centered_block], axis=0)
        num_carried = carried_samples.shape[0]
        # Each pair (t, t + lag) is counted in the block that holds t + lag; its partner t is in
        # the block or among the carried samples, since `max_lag` samples are carried.
        for lag in range(max_lag + 1):
            first_row = max(num_carried, lag)
            if first_row >= extended.shape[0]:
                continue  # no pair with its later element in this block
            lagged_product_sums[lag] += np.einsum(
                "ij,ij->j",
                extended[first_row:],
                extended[first_row - lag : extended.shape[0] - lag],
            )
        # Clamped: a negative start would count from the end instead of keeping all rows.
        carried_samples = (
            extended[max(extended.shape[0] - max_lag, 0) :] if max_lag > 0 else extended[:0]
        )

    return AutocovarianceEstimate(mean, lagged_product_sums / num_samples, num_samples)


# ==================================================================================================
def compute_autocorrelation(
    chain: ChainStore,
    max_lag: int,
    start: int = 0,
    stop: int | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    r"""Compute the autocorrelation function $\hat\rho_k = \hat c_k / \hat c_0$ of a chain segment.

    The ordinary single-chain estimate over the whole segment, from the biased autocovariance of
    `compute_autocovariance` (so $|\hat\rho_k| \le 1$ and $\hat\rho_0 = 1$). It is not the
    split-chain estimate inside `compute_effective_sample_size`, which also contains the variance
    of the half-chain means and is not meant to be read as a function of the lag.

    Args:
        chain (ChainStore): Chain, shape `(num_samples, num_components)`.
        max_lag (int): Largest lag $K$; must be smaller than the segment length.
        start (int): First sample of the segment. Defaults to `0`.
        stop (int | None): End of the segment (exclusive). Defaults to the end of the chain.
        block_size (int): Number of samples read at once. Defaults to `DEFAULT_BLOCK_SIZE`.

    Returns:
        np.ndarray: Autocorrelation, shape `(K + 1, num_components)`; `nan` for a constant
            component.

    Raises:
        ValueError: As `compute_autocovariance`.
    """
    autocovariance = compute_autocovariance(chain, max_lag, start, stop, block_size).autocovariance
    with np.errstate(divide="ignore", invalid="ignore"):
        autocorrelation = autocovariance / autocovariance[0]
    autocorrelation[:, autocovariance[0] == 0.0] = np.nan
    return autocorrelation


# ==================================================================================================
def compute_effective_sample_size(
    chain_estimates: Sequence[AutocovarianceEstimate],
) -> EffectiveSampleSizeResult:
    r"""Compute the effective sample size from the autocovariances of $M$ chains of equal length.

    With the within-chain variance $W = \frac{N}{N-1}\,\overline{\hat c_0}$ (average over chains),
    the between-chain variance $B$ of the chain means (the sample variance with $M - 1$ degrees of
    freedom, zero for $M = 1$) and $\widehat{\mathrm{var}}^+ = \frac{N-1}{N}W + B$, the
    autocorrelation estimate is
    $\hat\rho_k = 1 - (W - \overline{\hat c_k}) / \widehat{\mathrm{var}}^+$, with $\hat\rho_0 := 1$.
    The sum runs over the lag pairs $P_j = \hat\rho_{2j} + \hat\rho_{2j+1}$ before the first
    non-positive one, made monotone by $P_j \leftarrow \min(P_j, P_{j-1})$ (Geyer's initial monotone
    sequence); the first lag of the terminating pair is added if positive (as `arviz` does).
    Everything is vectorized over the components.

    Args:
        chain_estimates (Sequence[AutocovarianceEstimate]): One estimate per chain, all with the
            same number of samples, number of components and lags.

    Returns:
        EffectiveSampleSizeResult: Effective sample size and autocorrelation time per component.

    Raises:
        ValueError: If there is no estimate, the estimates differ in shape, or the chains have
            fewer than `MINIMUM_NUM_SAMPLES_PER_CHAIN` samples or fewer than two lags.
    """
    if len(chain_estimates) == 0:
        raise ValueError("At least one chain estimate is needed.")
    reference = chain_estimates[0]
    num_samples_per_chain = reference.num_samples
    if any(
        estimate.num_samples != num_samples_per_chain
        or estimate.autocovariance.shape != reference.autocovariance.shape
        for estimate in chain_estimates
    ):
        raise ValueError("All chain estimates must have the same number of samples and lags.")
    if num_samples_per_chain < MINIMUM_NUM_SAMPLES_PER_CHAIN:
        raise ValueError(
            f"Each chain needs at least {MINIMUM_NUM_SAMPLES_PER_CHAIN} samples, "
            f"got {num_samples_per_chain}."
        )
    num_lags = reference.autocovariance.shape[0]
    num_lag_pairs = num_lags // 2
    if num_lag_pairs < 1:
        raise ValueError(f"At least lags 0 and 1 are needed, got max_lag {num_lags - 1}.")

    num_chains = len(chain_estimates)
    mean_autocovariance = np.mean([estimate.autocovariance for estimate in chain_estimates], axis=0)
    within_chain_variance = (
        mean_autocovariance[0] * num_samples_per_chain / (num_samples_per_chain - 1)
    )
    total_variance_estimate = (
        within_chain_variance * (num_samples_per_chain - 1) / (num_samples_per_chain)
    )
    if num_chains > 1:
        chain_means = np.stack([estimate.mean for estimate in chain_estimates], axis=0)
        total_variance_estimate = total_variance_estimate + np.var(chain_means, axis=0, ddof=1)

    num_samples_total = num_chains * num_samples_per_chain
    with np.errstate(divide="ignore", invalid="ignore"):
        autocorrelation = 1.0 - (within_chain_variance - mean_autocovariance) / (
            total_variance_estimate
        )
    autocorrelation[0] = 1.0

    lag_pair_sums = (
        autocorrelation[0 : 2 * num_lag_pairs : 2] + autocorrelation[1 : 2 * num_lag_pairs : 2]
    )
    is_terminating = lag_pair_sums <= 0.0
    truncated = ~is_terminating.any(axis=0)
    num_summed_pairs = np.where(truncated, num_lag_pairs, is_terminating.argmax(axis=0))
    summed = np.arange(num_lag_pairs)[:, None] < num_summed_pairs
    monotone_pair_sums = np.minimum.accumulate(lag_pair_sums, axis=0)
    # The first lag of the terminating pair; there is none for a truncated sequence.
    terminating_lag = np.minimum(2 * num_summed_pairs, 2 * num_lag_pairs - 1)
    first_terminating_autocorrelation = np.take_along_axis(
        autocorrelation, terminating_lag[None, :], axis=0
    )[0]
    extra_term = np.where(~truncated & (first_terminating_autocorrelation > 0.0), 1.0, 0.0)
    extra_term = extra_term * first_terminating_autocorrelation

    autocorrelation_time = -1.0 + 2.0 * (monotone_pair_sums * summed).sum(axis=0) + extra_term
    autocorrelation_time = np.maximum(autocorrelation_time, 1.0 / np.log10(num_samples_total))
    is_constant = ~(total_variance_estimate > 0.0)
    autocorrelation_time = np.where(is_constant, np.nan, autocorrelation_time)
    return EffectiveSampleSizeResult(
        effective_sample_size=num_samples_total / autocorrelation_time,
        integrated_autocorrelation_time=autocorrelation_time,
        truncated=truncated & ~is_constant,
        num_samples=num_samples_total,
    )


# ==================================================================================================
def estimate_effective_sample_size(
    chain: ChainStore,
    max_lag: int,
    burn_in: int = 0,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> EffectiveSampleSizeResult:
    """Estimate the effective sample size of one chain, split into two halves.

    After discarding `burn_in` samples, the chain is split into its first and last
    `(num_samples - burn_in) // 2` samples (a middle sample is dropped for an odd length, as in
    `arviz`) and the halves are treated as two chains; see `compute_effective_sample_size`. Reads
    the halves in two passes each; see `compute_autocovariance`.

    Args:
        chain (ChainStore): Chain, shape `(num_samples, num_components)`.
        max_lag (int): Largest lag to accumulate, smaller than the half-chain length.
        burn_in (int): Number of leading samples to discard. Defaults to `0`.
        block_size (int): Number of samples read at once. Defaults to `DEFAULT_BLOCK_SIZE`.

    Returns:
        EffectiveSampleSizeResult: Effective sample size and autocorrelation time per component.

    Raises:
        ValueError: If `burn_in` is outside the chain, the half-chains are shorter than
            `MINIMUM_NUM_SAMPLES_PER_CHAIN`, or `max_lag` is not smaller than their length.
    """
    num_chain_samples = chain.shape[0]
    if not 0 <= burn_in < num_chain_samples:
        raise ValueError(f"burn_in must be in [0, {num_chain_samples}), got {burn_in}.")
    half_length = (num_chain_samples - burn_in) // 2
    if half_length < MINIMUM_NUM_SAMPLES_PER_CHAIN:
        raise ValueError(
            f"After the burn-in of {burn_in}, {num_chain_samples - burn_in} samples are left; "
            f"need at least {2 * MINIMUM_NUM_SAMPLES_PER_CHAIN} to split the chain."
        )
    first_half = compute_autocovariance(
        chain, max_lag, start=burn_in, stop=burn_in + half_length, block_size=block_size
    )
    second_half = compute_autocovariance(
        chain, max_lag, start=num_chain_samples - half_length, block_size=block_size
    )
    return compute_effective_sample_size([first_half, second_half])
