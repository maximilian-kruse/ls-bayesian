import pytest

from tests.spde_prior import helpers


# ==================================================================================================
@pytest.fixture(params=[None, helpers.ROBIN_CONSTANT], ids=["neumann", "robin"])
def robin_const(request: pytest.FixtureRequest) -> float | None:
    """Robin constant, restrict via `pytest.mark.parametrize("robin_const", ..., indirect=True)`."""
    return request.param


# --------------------------------------------------------------------------------------------------
@pytest.fixture
def built_prior_setup(
    fem_space_setup: helpers.FEMSpaceSetup, robin_const: float | None
) -> helpers.BuiltPriorSetup:
    """Prior from the builder, function-scoped as it holds solver state and buffers."""
    return helpers.build_bilaplacian_prior(fem_space_setup, robin_const, seed=0)
