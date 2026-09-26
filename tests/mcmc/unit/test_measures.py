import numpy as np
import pytest

from tests.mcmc import helpers

pytestmark = pytest.mark.unit


# ==================================================================================================
def test_evaluate_cost_matches_quadratic_form_formula() -> None:
    """`GaussianMeasure.evaluate_cost` is the only concrete method on the ABCs in `measures.py`;
    every algorithm reads it only through composed differences (e.g.
    `PCNAlgorithm._evaluate_reference_correction`), so a uniform error in its formula -- e.g. a
    missing factor of 1/2 -- could in principle cancel out of those differences without ever being
    caught there. Check the formula directly against an independently-inverted covariance."""
    dim = helpers.STATE_DIM
    rng = np.random.default_rng(200)
    covariance = helpers.random_spd_matrix(rng, dim)
    mean = rng.standard_normal(dim)
    measure_under_test = helpers.DenseGaussianMeasure(mean, covariance, seed=201)
    state = rng.standard_normal(dim)

    difference = state - mean
    expected_cost = 0.5 * difference @ np.linalg.inv(covariance) @ difference

    np.testing.assert_allclose(measure_under_test.evaluate_cost(state), expected_cost, rtol=1e-12)


# --------------------------------------------------------------------------------------------------
def test_evaluate_cost_is_zero_at_the_mean() -> None:
    dim = helpers.STATE_DIM
    rng = np.random.default_rng(202)
    covariance = helpers.random_spd_matrix(rng, dim)
    mean = rng.standard_normal(dim)
    measure_under_test = helpers.DenseGaussianMeasure(mean, covariance, seed=203)

    assert measure_under_test.evaluate_cost(mean) == 0.0


# --------------------------------------------------------------------------------------------------
def test_evaluate_cost_returns_python_float() -> None:
    dim = helpers.STATE_DIM
    rng = np.random.default_rng(204)
    measure_under_test = helpers.DenseGaussianMeasure(
        np.zeros(dim), helpers.random_spd_matrix(rng, dim), seed=205
    )

    cost = measure_under_test.evaluate_cost(rng.standard_normal(dim))

    assert isinstance(cost, float)
