import numpy as np
import pytest

from ls_bayesian.optimization.algorithms.scipy_lbfgs_b import ScipyLBFGSBSettings
from tests.optimization import helpers

QUADRATIC_DIM = 5


# ==================================================================================================
@pytest.fixture
def quadratic_matrix() -> np.ndarray:
    rng = np.random.default_rng(0)
    return helpers.random_spd_matrix(rng, QUADRATIC_DIM)


# --------------------------------------------------------------------------------------------------
@pytest.fixture
def quadratic_minimizer() -> np.ndarray:
    return np.random.default_rng(1).random(QUADRATIC_DIM)


# --------------------------------------------------------------------------------------------------
@pytest.fixture
def scipy_lbfgs_b_settings() -> ScipyLBFGSBSettings:
    return helpers.default_scipy_lbfgs_b_settings()
