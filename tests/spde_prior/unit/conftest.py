import pytest

from tests.spde_prior import helpers


# ==================================================================================================
@pytest.fixture
def exact_prior_setup(fem_space_setup: helpers.FEMSpaceSetup) -> helpers.ExactPriorSetup:
    """Dense exact prior operators, function-scoped as the converter holds mutable buffers."""
    return helpers.create_exact_prior_setup(fem_space_setup, seed=0)
