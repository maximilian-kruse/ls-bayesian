import pytest

from tests.posterior import helpers


# ==================================================================================================
@pytest.fixture
def likelihood_setup() -> helpers.LikelihoodSetup:
    return helpers.create_likelihood_setup(seed=0)


# --------------------------------------------------------------------------------------------------
@pytest.fixture
def dense_spd_likelihood_setup() -> helpers.LikelihoodSetup:
    """Likelihood setup with a dense, non-diagonal SPD precision matrix."""
    return helpers.create_dense_spd_likelihood_setup(seed=4)


# --------------------------------------------------------------------------------------------------
@pytest.fixture
def posterior_setup(likelihood_setup: helpers.LikelihoodSetup) -> helpers.PosteriorSetup:
    return helpers.create_posterior_setup(likelihood_setup, seed=1)


# --------------------------------------------------------------------------------------------------
@pytest.fixture
def nonlinear_posterior_setup(likelihood_setup: helpers.LikelihoodSetup) -> helpers.PosteriorSetup:
    return helpers.create_nonlinear_posterior_setup(likelihood_setup, seed=1)
