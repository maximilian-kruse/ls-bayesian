import pytest
from beartype.roar import BeartypeCallHintViolation

from ls_bayesian.optimization.algorithms.scipy_lbfgs_b import ScipyLBFGSBSettings

pytestmark = pytest.mark.unit


# ==================================================================================================
def test_settings_reject_non_positive_maximum_num_iterations() -> None:
    with pytest.raises(BeartypeCallHintViolation):
        ScipyLBFGSBSettings(maximum_num_iterations=0)
