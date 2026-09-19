r"""Integration test that the posterior tutorial notebook still runs and reproduces its results.

Running the notebook to completion already verifies that it executes without error against the
current `ls_bayesian` API; a raised exception (e.g. from a renamed class or method) fails the test
here before the reference-value comparison is even reached.
"""

import numpy as np
import pytest

from tests.posterior import helpers

pytestmark = pytest.mark.integration

# Expressions evaluated in the notebook's final namespace, after all of its own cells have run.
RESULT_EXPRESSIONS = {
    "total_cost": "float(total_cost)",
    "gradient_norm": "float(np.linalg.norm(total_gradient))",
    "gradient_relative_error": "float(relative_error)",
    "map_estimate_norm": "float(np.linalg.norm(map_estimate))",
    "forward_evaluations_during_optimization": "int(forward_evaluations_during_optimization)",
    "closed_form_map_estimate_norm": "float(np.linalg.norm(closed_form_map_estimate))",
    "hessian_is_spd": "bool(hessian_is_spd)",
    "gradient_at_closed_form_estimate_norm": "float(gradient_at_closed_form_estimate_norm)",
    "map_estimate_relative_error": "float(map_estimate_relative_error)",
}
# The L-BFGS-B convergence path is sensitive to floating-point rounding, so results that depend on
# it are reproducible only to moderate precision.
REGRESSION_RELATIVE_TOLERANCE = 1e-4
# The finite-difference gradient check and the closed-form-vs-optimizer comparison are correctness
# bounds, not reproducible regression values: their magnitude is dominated by float64 roundoff and
# the optimizer's convergence tolerance, so we only require them to stay small.
GRADIENT_RELATIVE_ERROR_BOUND = 1e-6
CLOSED_FORM_GRADIENT_NORM_BOUND = 1e-10
MAP_ESTIMATE_RELATIVE_ERROR_BOUND = 1e-4

# Reference values recorded from a passing run of the notebook. Update these only after confirming,
# and explaining in the commit message, why the tutorial's numerical results are expected to change.
POSTERIOR_REFERENCE_VALUES = {
    "total_cost": 21.332560538316635,
    "gradient_norm": 16.78239967922721,
    "map_estimate_norm": 0.5955733787736166,
    "forward_evaluations_during_optimization": 7,
    "closed_form_map_estimate_norm": 0.5955733691802195,
    "hessian_is_spd": True,
}


# ==================================================================================================
def test_posterior_tutorial_notebook_runs_and_matches_reference_values() -> None:
    """Execute the posterior tutorial notebook and check its cost/gradient/MAP results."""
    result_values = helpers.execute_notebook_and_extract_values(
        helpers.POSTERIOR_NOTEBOOK, RESULT_EXPRESSIONS
    )

    for key, expected_value in POSTERIOR_REFERENCE_VALUES.items():
        if isinstance(expected_value, bool):
            assert result_values[key] == expected_value
        else:
            np.testing.assert_allclose(
                result_values[key], expected_value, rtol=REGRESSION_RELATIVE_TOLERANCE, atol=0
            )
    assert result_values["gradient_relative_error"] < GRADIENT_RELATIVE_ERROR_BOUND
    assert result_values["gradient_at_closed_form_estimate_norm"] < CLOSED_FORM_GRADIENT_NORM_BOUND
    assert result_values["map_estimate_relative_error"] < MAP_ESTIMATE_RELATIVE_ERROR_BOUND
