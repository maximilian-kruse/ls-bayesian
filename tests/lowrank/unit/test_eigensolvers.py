import numpy as np
import pytest

from ls_bayesian.lowrank.eigensolvers import (
    GeneralizedEigenpairs,
    LanczosSolver,
    RandomizedDoublePassSolver,
    compute_generalized_eigenpairs,
)
from tests.lowrank import helpers

pytestmark = pytest.mark.unit


# ==================================================================================================
def _compute(setup: helpers.DenseLowRankSetup, solver, num_eigenvalues: int, **kwargs):
    return compute_generalized_eigenpairs(
        setup.apply_hessian,
        setup.reference.apply_precision_operator,
        setup.reference.apply_covariance_operator,
        helpers.PARAMETER_SIZE,
        num_eigenvalues,
        solver,
        **kwargs,
    )


# ==================================================================================================
def test_eigenpairs_with_inconsistent_shapes_raise() -> None:
    with pytest.raises(ValueError, match="do not match"):
        GeneralizedEigenpairs(np.ones(3), np.ones((5, 2)))


# --------------------------------------------------------------------------------------------------
def test_non_positive_eigenvalues_are_discarded(dense_setup: helpers.DenseLowRankSetup) -> None:
    eigenpairs = GeneralizedEigenpairs(
        np.array([3.0, 1.0, 0.0, -1e-14]), np.eye(helpers.PARAMETER_SIZE)[:, :4]
    )

    result = _compute(dense_setup, helpers.FixedEigensolver(eigenpairs), 4)

    np.testing.assert_array_equal(result.eigenvalues, [3.0, 1.0])
    assert result.eigenvectors.shape == (helpers.PARAMETER_SIZE, 2)


# --------------------------------------------------------------------------------------------------
def test_minimum_eigenvalue_sets_the_discard_threshold(
    dense_setup: helpers.DenseLowRankSetup,
) -> None:
    eigenpairs = GeneralizedEigenpairs(
        np.array([3.0, 1.0, 1e-9]), np.eye(helpers.PARAMETER_SIZE)[:, :3]
    )

    result = _compute(dense_setup, helpers.FixedEigensolver(eigenpairs), 3, minimum_eigenvalue=1e-6)

    np.testing.assert_array_equal(result.eigenvalues, [3.0, 1.0])


# --------------------------------------------------------------------------------------------------
def test_too_many_eigenvalues_raise(dense_setup: helpers.DenseLowRankSetup) -> None:
    with pytest.raises(ValueError, match="exceeds the parameter size"):
        _compute(dense_setup, RandomizedDoublePassSolver(), helpers.PARAMETER_SIZE + 1)


# --------------------------------------------------------------------------------------------------
def test_lanczos_requires_fewer_eigenvalues_than_parameter_size(
    dense_setup: helpers.DenseLowRankSetup,
) -> None:
    with pytest.raises(ValueError, match="fewer eigenvalues"):
        _compute(dense_setup, LanczosSolver(), helpers.PARAMETER_SIZE)


# --------------------------------------------------------------------------------------------------
def test_sketch_larger_than_parameter_size_raises(dense_setup: helpers.DenseLowRankSetup) -> None:
    with pytest.raises(ValueError, match="Sketch size"):
        _compute(dense_setup, RandomizedDoublePassSolver(oversampling=helpers.PARAMETER_SIZE), 2)
