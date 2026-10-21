import pytest

from tests.lowrank import helpers


# ==================================================================================================
@pytest.fixture
def dense_setup() -> helpers.DenseLowRankSetup:
    return helpers.create_dense_low_rank_setup(seed=0)
