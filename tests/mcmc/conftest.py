import pytest

from tests.mcmc import helpers


# ==================================================================================================
@pytest.fixture
def quadratic_gaussian_setup() -> helpers.QuadraticGaussianSetup:
    return helpers.create_quadratic_gaussian_setup(seed=0)
