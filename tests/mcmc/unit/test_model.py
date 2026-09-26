import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation

from ls_bayesian.mcmc.model import MCMCModel
from tests.mcmc import helpers

pytestmark = pytest.mark.unit


# ==================================================================================================
def test_rejects_model_without_reference() -> None:
    setup = helpers.create_quadratic_gaussian_setup(seed=0)

    with pytest.raises((TypeError, BeartypeCallHintViolation)):
        MCMCModel(target=setup.target, approximation=setup.reference)


# --------------------------------------------------------------------------------------------------
def test_accepts_model_with_only_reference() -> None:
    setup = helpers.create_quadratic_gaussian_setup(seed=0)

    model = MCMCModel(target=setup.target, reference=setup.reference)

    assert model.approximation is None


# --------------------------------------------------------------------------------------------------
def test_accepts_model_with_reference_and_approximation() -> None:
    setup = helpers.create_quadratic_gaussian_setup(seed=0)
    approximation = helpers.DenseGaussianMeasure(
        np.zeros(helpers.STATE_DIM),
        helpers.random_spd_matrix(np.random.default_rng(2), helpers.STATE_DIM),
        seed=3,
    )

    model = MCMCModel(target=setup.target, reference=setup.reference, approximation=approximation)

    assert model.approximation is approximation
