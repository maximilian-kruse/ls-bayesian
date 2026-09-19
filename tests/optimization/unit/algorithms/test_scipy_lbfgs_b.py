import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation

from ls_bayesian.optimization.algorithms.scipy_lbfgs_b import LBFGSOptimizer, LBFGSSettings
from tests.optimization import helpers

pytestmark = pytest.mark.unit

CONVERGENCE_ABSOLUTE_TOLERANCE = 1e-4


# ==================================================================================================
def test_converges_to_quadratic_minimizer(
    quadratic_matrix: np.ndarray, quadratic_minimizer: np.ndarray, lbfgs_settings: LBFGSSettings
) -> None:
    optimizer = LBFGSOptimizer(lbfgs_settings)
    model = helpers.QuadraticModel(quadratic_matrix, quadratic_minimizer)

    for seed in range(3):
        initial_guess = np.random.default_rng(seed).normal(size=quadratic_minimizer.shape)

        result = optimizer.run(initial_guess, model)

        assert result.success
        np.testing.assert_allclose(
            result.result, quadratic_minimizer, atol=CONVERGENCE_ABSOLUTE_TOLERANCE
        )
        assert result.gradient_norm_history[-1] < lbfgs_settings.relative_gradient_tolerance * 10


# --------------------------------------------------------------------------------------------------
def test_converges_on_rosenbrock(lbfgs_settings: LBFGSSettings) -> None:
    optimizer = LBFGSOptimizer(lbfgs_settings)
    initial_guess = np.array([-1.2, 1.0, -1.0, 1.5])
    model = helpers.RosenbrockModel()

    result = optimizer.run(initial_guess, model)

    assert result.success
    assert result.num_iterations > 0
    np.testing.assert_allclose(
        result.result, np.ones_like(initial_guess), atol=CONVERGENCE_ABSOLUTE_TOLERANCE
    )
    assert result.loss_history[-1] < result.loss_history[0]


# --------------------------------------------------------------------------------------------------
def test_settings_reject_non_positive_maximum_num_iterations() -> None:
    with pytest.raises(BeartypeCallHintViolation):
        LBFGSSettings(maximum_num_iterations=0)


# --------------------------------------------------------------------------------------------------
def test_history_lengths_are_consistent_with_num_iterations(
    quadratic_matrix: np.ndarray, quadratic_minimizer: np.ndarray, lbfgs_settings: LBFGSSettings
) -> None:
    optimizer = LBFGSOptimizer(lbfgs_settings)
    model = helpers.QuadraticModel(quadratic_matrix, quadratic_minimizer)

    result = optimizer.run(np.zeros_like(quadratic_minimizer), model)

    assert result.loss_history.shape[0] >= result.num_iterations
    assert result.gradient_norm_history.shape[0] >= result.num_iterations
