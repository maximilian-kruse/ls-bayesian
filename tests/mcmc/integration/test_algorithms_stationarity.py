import numpy as np
import pytest

from ls_bayesian.mcmc.algorithms.pmala import PMALAAlgorithm
from tests.mcmc import helpers

pytestmark = pytest.mark.integration

# Not measured directly (that would need a real ESS estimator, adding complexity to a test meant
# as a coarse end-to-end sanity check on top of the exact single-step differential tests in
# unit/algorithms/test_pmala.py); a lag-1-autocorrelation-based estimate at this step size on this
# fixed problem gave an integrated autocorrelation time of ~6.4, so 20 is a deliberately
# conservative (too-generous) round number, keeping the false-failure rate low at the cost of a
# looser sanity check.
_ASSUMED_INTEGRATED_AUTOCORRELATION_TIME = 20
_NUM_STANDARD_ERRORS = 4  # ~3e-5 one-sided false-failure rate per coordinate under the CLT normal
# approximation implied by _ASSUMED_INTEGRATED_AUTOCORRELATION_TIME, for the fixed seeds below.


# ==================================================================================================
def test_pmala_chain_recovers_analytic_posterior_moments() -> None:
    """Drives `PMALAAlgorithm.compute_step` for many steps (the full run loop, not a single
    formula evaluation) and checks that the resulting empirical chain recovers the analytically
    known Gaussian posterior mean and per-coordinate variance -- an end-to-end check that the
    propose/accept-reject loop as a whole samples correctly, complementing (not replacing) the
    exact single-step detailed-balance tests in `unit/algorithms/test_pmala.py`.
    """
    setup = helpers.create_quadratic_gaussian_setup(seed=110)
    posterior_mean, posterior_covariance = helpers.gaussian_posterior_moments(
        setup.hessian, setup.minimizer, setup.covariance
    )
    posterior_variance = np.diag(posterior_covariance)
    step_width = 0.01
    algorithm_under_test = PMALAAlgorithm(
        setup.target, setup.reference, setup.reference, step_width
    )
    rng = np.random.default_rng(111)
    state = posterior_mean.copy()

    burn_in = 1000
    num_samples = 8000
    for _ in range(burn_in):
        state, _ = algorithm_under_test.compute_step(state, rng)

    samples = np.empty((num_samples, helpers.STATE_DIM))
    for step in range(num_samples):
        state, _ = algorithm_under_test.compute_step(state, rng)
        samples[step] = state

    effective_sample_size = num_samples / _ASSUMED_INTEGRATED_AUTOCORRELATION_TIME

    # Per-coordinate errors normalized by their standard error, compared against the scalar
    # _NUM_STANDARD_ERRORS threshold directly (rather than passing an array `atol` to
    # `assert_allclose`, whose mismatch-reporting path does not support that).
    sample_mean = samples.mean(axis=0)
    mean_standard_error = np.sqrt(posterior_variance / effective_sample_size)
    normalized_mean_error = (sample_mean - posterior_mean) / mean_standard_error
    np.testing.assert_allclose(
        normalized_mean_error, np.zeros_like(normalized_mean_error), atol=_NUM_STANDARD_ERRORS
    )

    sample_variance = samples.var(axis=0, ddof=1)
    variance_standard_error = posterior_variance * np.sqrt(2.0 / effective_sample_size)
    normalized_variance_error = (sample_variance - posterior_variance) / variance_standard_error
    np.testing.assert_allclose(
        normalized_variance_error,
        np.zeros_like(normalized_variance_error),
        atol=_NUM_STANDARD_ERRORS,
    )
