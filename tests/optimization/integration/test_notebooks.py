r"""Integration tests that the optimization tutorial notebooks still run and reproduce their
results.

Running a notebook to completion already verifies that it executes without error against the
current `ls_bayesian` API; a raised exception (e.g. from a renamed class or method) fails a test
here before the reference-value comparison is even reached.
"""

import pytest

from tests.optimization import helpers

pytestmark = pytest.mark.integration

# Expressions evaluated in the scipy_lbfgs notebook's final namespace, after all of its own cells
# have run.
SCIPY_LBFGS_RESULT_EXPRESSIONS = {
    "success": "bool(result.success)",
    "num_iterations": "int(result.num_iterations)",
    "final_loss": "float(result.loss_history[-1])",
    "distance_to_minimizer": "float(np.linalg.norm(result.result - minimizer))",
}
# Expressions evaluated in the custom_lbfgs notebook's final namespace: one result per inner
# product (Euclidean vs. Hessian-matched).
CUSTOM_LBFGS_RESULT_EXPRESSIONS = {
    "euclidean_num_iterations": "int(result_euclidean.num_iterations)",
    "natural_num_iterations": "int(result_natural.num_iterations)",
    "euclidean_final_loss": "float(result_euclidean.loss_history[-1])",
    "natural_final_loss": "float(result_natural.loss_history[-1])",
    "euclidean_distance_to_minimizer": (
        "float(np.linalg.norm(result_euclidean.result - minimizer))"
    ),
    "natural_distance_to_minimizer": "float(np.linalg.norm(result_natural.result - minimizer))",
}

# Iteration counts are reproducible exactly: both notebooks use a seeded RNG and a fixed initial
# guess, and the backtracking line search/two-loop recursion are deterministic given those.
SCIPY_LBFGS_REFERENCE_VALUES = {
    "success": True,
    "num_iterations": 56,
}
CUSTOM_LBFGS_REFERENCE_VALUES = {
    "euclidean_num_iterations": 22,
    # The Hessian-matched inner product makes the two-loop recursion's identity seed reduce to the
    # exact Newton direction on this quadratic objective, converging in a single Armijo-accepted
    # full step; see the notebook's "Run 2" markdown cell for the derivation.
    "natural_num_iterations": 1,
}
# The final loss and distance-to-minimizer are dominated by float64 roundoff once the optimizer
# has converged (down to ~1e-10 to 1e-31 depending on the run), so they are correctness bounds, not
# reproducible regression values comparable via relative tolerance.
SCIPY_LBFGS_FINAL_LOSS_BOUND = 1e-12
SCIPY_LBFGS_DISTANCE_BOUND = 1e-6
CUSTOM_LBFGS_EUCLIDEAN_FINAL_LOSS_BOUND = 1e-12
CUSTOM_LBFGS_EUCLIDEAN_DISTANCE_BOUND = 1e-6
CUSTOM_LBFGS_NATURAL_FINAL_LOSS_BOUND = 1e-15
CUSTOM_LBFGS_NATURAL_DISTANCE_BOUND = 1e-10


# ==================================================================================================
def test_scipy_lbfgs_tutorial_notebook_runs_and_converges() -> None:
    """Execute the scipy_lbfgs tutorial notebook and check its convergence results."""
    result_values = helpers.execute_notebook_and_extract_values(
        helpers.SCIPY_LBFGS_NOTEBOOK, SCIPY_LBFGS_RESULT_EXPRESSIONS
    )

    assert result_values["success"] == SCIPY_LBFGS_REFERENCE_VALUES["success"]
    assert result_values["num_iterations"] == SCIPY_LBFGS_REFERENCE_VALUES["num_iterations"]
    assert result_values["final_loss"] < SCIPY_LBFGS_FINAL_LOSS_BOUND
    assert result_values["distance_to_minimizer"] < SCIPY_LBFGS_DISTANCE_BOUND


# --------------------------------------------------------------------------------------------------
def test_custom_lbfgs_tutorial_notebook_runs_and_converges() -> None:
    """Execute the custom_lbfgs tutorial notebook and check both runs' convergence results, and
    that the Hessian-matched inner product converges in strictly fewer iterations than the
    Euclidean one."""
    result_values = helpers.execute_notebook_and_extract_values(
        helpers.CUSTOM_LBFGS_NOTEBOOK, CUSTOM_LBFGS_RESULT_EXPRESSIONS
    )

    assert (
        result_values["euclidean_num_iterations"]
        == CUSTOM_LBFGS_REFERENCE_VALUES["euclidean_num_iterations"]
    )
    assert (
        result_values["natural_num_iterations"]
        == CUSTOM_LBFGS_REFERENCE_VALUES["natural_num_iterations"]
    )
    assert result_values["natural_num_iterations"] < result_values["euclidean_num_iterations"]
    assert result_values["euclidean_final_loss"] < CUSTOM_LBFGS_EUCLIDEAN_FINAL_LOSS_BOUND
    assert result_values["natural_final_loss"] < CUSTOM_LBFGS_NATURAL_FINAL_LOSS_BOUND
    assert result_values["euclidean_distance_to_minimizer"] < CUSTOM_LBFGS_EUCLIDEAN_DISTANCE_BOUND
    assert result_values["natural_distance_to_minimizer"] < CUSTOM_LBFGS_NATURAL_DISTANCE_BOUND
