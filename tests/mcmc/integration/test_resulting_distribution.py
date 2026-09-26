from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from ls_bayesian.mcmc.algorithm import MCMCAlgorithm
from ls_bayesian.mcmc.algorithms.mala import MALAAlgorithm
from ls_bayesian.mcmc.algorithms.pcn import PCNAlgorithm
from ls_bayesian.mcmc.algorithms.pmala import PMALAAlgorithm
from tests.mcmc import helpers

pytestmark = pytest.mark.integration

_NUM_STANDARD_ERRORS = 4  # ~3e-5 one-sided false-failure rate per coordinate under the CLT normal
# approximation implied by each case's assumed integrated autocorrelation time, for its fixed seeds.
_BURN_IN = 1000
_NUM_SAMPLES = 8000

# Chains use fixed seeds throughout (RNG, initial state), so their acceptance rates and posterior
# moment recovery are exactly reproducible, not merely bounded -- like the tutorial notebooks'
# regression values (`tests/mcmc/integration/test_notebooks.py`). A loose relative tolerance absorbs
# BLAS/platform-dependent summation order in the dense linear algebra each step performs.
_REGRESSION_RELATIVE_TOLERANCE = 1e-6


# ==================================================================================================
def _build_pcn(setup: helpers.QuadraticGaussianSetup, step_width: float) -> MCMCAlgorithm:
    return PCNAlgorithm(setup.to_model(), step_width)


# --------------------------------------------------------------------------------------------------
def _build_mala(setup: helpers.QuadraticGaussianSetup, step_width: float) -> MCMCAlgorithm:
    return MALAAlgorithm(setup.to_model(), step_width)


# --------------------------------------------------------------------------------------------------
def _build_pmala(setup: helpers.QuadraticGaussianSetup, step_width: float) -> MCMCAlgorithm:
    return PMALAAlgorithm(setup.to_model(setup.reference), step_width)


# ==================================================================================================
@dataclass(frozen=True)
class _ChainCase:
    """One algorithm's fixed configuration for the moment-recovery/regression checks below.

    Attributes:
        setup_seed (int): Seed for the random quadratic-Gaussian target/reference problem.
        rng_seed (int): Seed for the chain's own random number generator.
        step_width (float): Algorithm step size.
        build_algorithm (Callable[[helpers.QuadraticGaussianSetup, float], MCMCAlgorithm]):
            Constructs the algorithm from the problem setup and `step_width`.
        assumed_integrated_autocorrelation_time (float): See
            `_assert_chain_recovers_analytic_posterior_moments`. A deliberately conservative round
            number above a lag-1-autocorrelation-based estimate at this `step_width` on this fixed
            problem.
        reference_values (dict[str, float]): Recorded from a passing run, already validated against
            the analytic posterior by `_assert_chain_recovers_analytic_posterior_moments`. Update
            these only after confirming, and explaining in the commit message, why the chain's
            numerical results are expected to change.
    """

    setup_seed: int
    rng_seed: int
    step_width: float
    build_algorithm: Callable[[helpers.QuadraticGaussianSetup, float], MCMCAlgorithm]
    assumed_integrated_autocorrelation_time: float
    reference_values: dict[str, float]


# Lag-1-autocorrelation estimates at each case's step size on its fixed problem gave integrated
# autocorrelation times of ~22 (pcn), ~6.7 (mala), ~6.4 (pmala).
_CHAIN_CASES = {
    "pcn": _ChainCase(
        setup_seed=120,
        rng_seed=121,
        step_width=0.1,
        build_algorithm=_build_pcn,
        assumed_integrated_autocorrelation_time=65,
        reference_values={
            "acceptance_rate": 0.50225,
            "mean_error_norm": 0.1271942764027872,
            "std_ratio_error_norm": 0.07632448093948214,
        },
    ),
    "mala": _ChainCase(
        setup_seed=130,
        rng_seed=131,
        step_width=0.01,
        build_algorithm=_build_mala,
        assumed_integrated_autocorrelation_time=20,
        reference_values={
            "acceptance_rate": 0.73075,
            "mean_error_norm": 0.029759357510185536,
            "std_ratio_error_norm": 0.034319921019734204,
        },
    ),
    "pmala": _ChainCase(
        setup_seed=110,
        rng_seed=111,
        step_width=0.01,
        build_algorithm=_build_pmala,
        assumed_integrated_autocorrelation_time=20,
        reference_values={
            "acceptance_rate": 0.563875,
            "mean_error_norm": 0.112988818742305,
            "std_ratio_error_norm": 0.0390618858571281,
        },
    ),
}


# ==================================================================================================
@dataclass(frozen=True)
class _ChainResult:
    """The empirical outcome of running one `_ChainCase`'s chain, plus the analytic posterior it is
    checked against -- computed once per case and shared by both tests below via the `chain_result`
    fixture, since it is identical input for both."""

    case: _ChainCase
    samples: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    acceptance_rate: float
    posterior_mean: np.ndarray[tuple[int], np.dtype[np.float64]]
    posterior_covariance: np.ndarray[tuple[int, int], np.dtype[np.float64]]


# --------------------------------------------------------------------------------------------------
def _run_chain(
    algorithm: MCMCAlgorithm,
    rng: np.random.Generator,
    initial_state: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.float64]], float]:
    """Drive `algorithm.compute_step` for `_BURN_IN + _NUM_SAMPLES` steps (the full run loop, not a
    single formula evaluation) and return the post-burn-in samples and the fraction of those steps
    that were accepted."""
    state = initial_state
    for _ in range(_BURN_IN):
        state, _ = algorithm.compute_step(state, rng)

    samples = np.empty((_NUM_SAMPLES, initial_state.shape[0]))
    num_accepted = 0
    for step in range(_NUM_SAMPLES):
        state, accepted = algorithm.compute_step(state, rng)
        num_accepted += accepted
        samples[step] = state
    return samples, num_accepted / _NUM_SAMPLES


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module", params=_CHAIN_CASES.values(), ids=_CHAIN_CASES.keys())
def chain_result(request: pytest.FixtureRequest) -> _ChainResult:
    """Run `request.param`'s chain once; shared by both the statistical and regression tests below
    (module-scoped, so it is not re-run for each)."""
    case: _ChainCase = request.param
    setup = helpers.create_quadratic_gaussian_setup(seed=case.setup_seed)
    posterior_mean, posterior_covariance = helpers.gaussian_posterior_moments(
        setup.hessian, setup.minimizer, setup.covariance
    )
    algorithm = case.build_algorithm(setup, case.step_width)
    rng = np.random.default_rng(case.rng_seed)

    samples, acceptance_rate = _run_chain(algorithm, rng, posterior_mean.copy())

    return _ChainResult(case, samples, acceptance_rate, posterior_mean, posterior_covariance)


# ==================================================================================================
def test_chain_recovers_analytic_posterior_moments(chain_result: _ChainResult) -> None:
    """Checks that the chain's empirical mean/variance recovers the analytically known Gaussian
    posterior moments -- an end-to-end check that the propose/accept-reject loop as a whole samples
    correctly against an independent oracle, complementing (not replacing) each algorithm's exact
    single-step detailed-balance tests in `unit/algorithms/`.

    `chain_result.case.assumed_integrated_autocorrelation_time` converts `_NUM_SAMPLES` into an
    effective sample size for the CLT-based standard error below; it is not measured directly (that
    would need a real ESS estimator, adding complexity to a test meant as a coarse sanity check),
    but a deliberately conservative (too-generous) round number, keeping the false-failure rate low
    at the cost of a looser check.
    """
    posterior_variance = np.diag(chain_result.posterior_covariance)
    effective_sample_size = _NUM_SAMPLES / chain_result.case.assumed_integrated_autocorrelation_time

    # Per-coordinate errors normalized by their standard error, compared against the scalar
    # _NUM_STANDARD_ERRORS threshold directly (rather than passing an array `atol` to
    # `assert_allclose`, whose mismatch-reporting path does not support that).
    sample_mean = chain_result.samples.mean(axis=0)
    mean_standard_error = np.sqrt(posterior_variance / effective_sample_size)
    normalized_mean_error = (sample_mean - chain_result.posterior_mean) / mean_standard_error
    np.testing.assert_allclose(
        normalized_mean_error, np.zeros_like(normalized_mean_error), atol=_NUM_STANDARD_ERRORS
    )

    sample_variance = chain_result.samples.var(axis=0, ddof=1)
    variance_standard_error = posterior_variance * np.sqrt(2.0 / effective_sample_size)
    normalized_variance_error = (sample_variance - posterior_variance) / variance_standard_error
    np.testing.assert_allclose(
        normalized_variance_error,
        np.zeros_like(normalized_variance_error),
        atol=_NUM_STANDARD_ERRORS,
    )


# --------------------------------------------------------------------------------------------------
def test_chain_matches_regression_reference(chain_result: _ChainResult) -> None:
    """Checks `chain_result`'s acceptance rate and normalized moment-recovery error norms against
    `chain_result.case.reference_values`, recorded from a run already validated against the analytic
    posterior by `test_chain_recovers_analytic_posterior_moments`. Exactly reproducible given the
    fixed seeds, so -- unlike that statistical check -- this catches any change to the result,
    however small, tightening it considerably at the cost of no longer testing against an
    independent oracle."""
    posterior_standard_deviation = np.sqrt(np.diag(chain_result.posterior_covariance))
    mean_error_norm = np.linalg.norm(
        (chain_result.samples.mean(axis=0) - chain_result.posterior_mean)
        / posterior_standard_deviation
    )
    std_ratio_error_norm = np.linalg.norm(
        chain_result.samples.std(axis=0, ddof=1) / posterior_standard_deviation - 1.0
    )
    result_values = {
        "acceptance_rate": chain_result.acceptance_rate,
        "mean_error_norm": mean_error_norm,
        "std_ratio_error_norm": std_ratio_error_norm,
    }
    for key, expected_value in chain_result.case.reference_values.items():
        np.testing.assert_allclose(
            result_values[key], expected_value, rtol=_REGRESSION_RELATIVE_TOLERANCE, atol=0
        )
