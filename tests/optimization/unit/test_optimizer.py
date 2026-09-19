from pathlib import Path

import numpy as np
import pytest

from ls_bayesian.common.logging import BaseLogger, LoggerSettings
from ls_bayesian.optimization import optimizer
from ls_bayesian.optimization.optimizer import OptimizationHistory
from tests.optimization import helpers

pytestmark = pytest.mark.unit

_ZERO_MODEL = helpers.ZeroModel()


# ==================================================================================================
def test_run_rejects_non_one_dimensional_initial_guess() -> None:
    optimizer_under_test = helpers.FakeOptimizer()

    with pytest.raises(ValueError, match="one-dimensional"):
        optimizer_under_test.run(np.zeros((2, 2)), _ZERO_MODEL)


# --------------------------------------------------------------------------------------------------
def test_run_rejects_non_finite_initial_guess() -> None:
    optimizer_under_test = helpers.FakeOptimizer()

    with pytest.raises(ValueError, match="finite"):
        optimizer_under_test.run(np.array([1.0, np.nan]), _ZERO_MODEL)


# --------------------------------------------------------------------------------------------------
def test_run_raises_when_hessian_required_but_not_implemented() -> None:
    optimizer_under_test = helpers.FakeHessianOptimizer()

    with pytest.raises(NotImplementedError):
        optimizer_under_test.run(np.zeros(2), _ZERO_MODEL)


# --------------------------------------------------------------------------------------------------
def test_run_passes_hvp_function_through_when_given() -> None:
    optimizer_under_test = helpers.FakeHessianOptimizer(num_iterations=1)
    model = helpers.ConstantGradientModel(gradient_value=1.0)

    result = optimizer_under_test.run(np.zeros(2), model)

    assert result.success


# --------------------------------------------------------------------------------------------------
def test_run_records_loss_and_gradient_norm_history() -> None:
    matrix = np.eye(2)
    minimizer = np.zeros(2)
    optimizer_under_test = helpers.FakeOptimizer(num_iterations=3, step_size=0.5)
    model = helpers.QuadraticModel(matrix, minimizer)

    result = optimizer_under_test.run(np.array([1.0, 1.0]), model)

    assert result.loss_history.shape == (3,)
    assert result.gradient_norm_history.shape == (3,)
    assert result.num_iterations == 3
    np.testing.assert_allclose(result.gradient_norm_history[0], np.sqrt(2.0))


# --------------------------------------------------------------------------------------------------
def test_run_logs_header_and_one_row_per_iteration(tmp_path: Path) -> None:
    logfile_path = tmp_path / "fake_optimizer.log"
    logger_settings = LoggerSettings(print_to_console=False, logfile_path=logfile_path)

    with BaseLogger(logger_settings, prefix="optimizer") as logger:
        optimizer_under_test = helpers.FakeOptimizer(num_iterations=2, logger=logger)
        optimizer_under_test.run(np.array([1.0, 1.0]), _ZERO_MODEL)

    log_content = logfile_path.read_text()
    assert "Iteration" in log_content
    assert log_content.count("\n") >= 3  # header + separator + at least 2 rows


# ==================================================================================================
def test_empty_history_returns_empty_arrays() -> None:
    history = OptimizationHistory()

    assert history.loss_history.shape == (0,)
    assert history.gradient_norm_history.shape == (0,)


# --------------------------------------------------------------------------------------------------
def test_history_records_values_in_call_order() -> None:
    history = OptimizationHistory()

    history.record_loss(3.0)
    history.record_loss(2.0)
    history.record_gradient_norm(1.0)
    history.record_gradient_norm(0.5)
    history.record_gradient_norm(0.1)

    np.testing.assert_array_equal(history.loss_history, [3.0, 2.0])
    np.testing.assert_array_equal(history.gradient_norm_history, [1.0, 0.5, 0.1])


# --------------------------------------------------------------------------------------------------
def test_history_reads_do_not_alias_each_other() -> None:
    history = OptimizationHistory()
    history.record_loss(1.0)

    first_read = history.loss_history
    first_read[0] = 999.0
    second_read = history.loss_history

    np.testing.assert_array_equal(second_read, [1.0])


# ==================================================================================================
def test_log_header_and_iteration_write_expected_content(tmp_path: Path) -> None:
    logfile_path = tmp_path / "optimization.log"
    logger_settings = LoggerSettings(print_to_console=False, logfile_path=logfile_path)

    with BaseLogger(logger_settings, prefix="optimization") as logger:
        optimizer.log_header(logger)
        optimizer.log_iteration(logger, 1, 0.123, 4.5, 0.01)

    log_content = logfile_path.read_text()
    assert "Iteration" in log_content
    assert "Loss" in log_content
    assert "Grad. norm" in log_content
    assert "4.500000e+00" in log_content
    assert "1.000000e-02" in log_content


# --------------------------------------------------------------------------------------------------
def test_log_header_and_iteration_are_no_op_without_logger() -> None:
    optimizer.log_header(None)
    optimizer.log_iteration(None, 1, 0.1, 1.0, 1.0)
