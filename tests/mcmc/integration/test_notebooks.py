r"""Integration tests that the mcmc tutorial notebooks still run and reproduce their results.

Running a notebook to completion already verifies that it executes without error against the
current `ls_bayesian` API; a raised exception (e.g. from a renamed class or method) fails a test
here before the reference-value comparison is even reached.
"""

from pathlib import Path

import numpy as np
import pytest

from tests import notebook_helpers
from tests.mcmc import helpers

pytestmark = pytest.mark.integration

# All three notebooks use fixed seeds throughout (RNG, initial state), so their acceptance rates
# and moment-recovery errors are exactly reproducible, not merely bounded -- like the optimization
# tutorials' iteration counts. A loose relative tolerance absorbs BLAS/platform-dependent summation
# order in the dense linear algebra each step performs.
REGRESSION_RELATIVE_TOLERANCE = 1e-6

# Expressions evaluated in the pcn notebook's final namespace, after all of its own cells have run.
PCN_RESULT_EXPRESSIONS = {
    "classical_acceptance_rate": "float(classical_acceptance_rate)",
    "informed_acceptance_rate": "float(informed_acceptance_rate)",
    "classical_mean_error_norm": (
        "float(np.linalg.norm((classical_samples.mean(axis=0) - posterior_mean) "
        "/ posterior_standard_deviation))"
    ),
    "classical_std_ratio_error_norm": (
        "float(np.linalg.norm(classical_samples.std(axis=0, ddof=1) "
        "/ posterior_standard_deviation - 1.0))"
    ),
    "informed_mean_error_norm": (
        "float(np.linalg.norm((informed_samples.mean(axis=0) - posterior_mean) "
        "/ posterior_standard_deviation))"
    ),
    "informed_std_ratio_error_norm": (
        "float(np.linalg.norm(informed_samples.std(axis=0, ddof=1) "
        "/ posterior_standard_deviation - 1.0))"
    ),
}
# Recorded from a passing run. Update these only after confirming, and explaining in the commit
# message, why the tutorial's numerical results are expected to change.
PCN_REFERENCE_VALUES = {
    "classical_acceptance_rate": 0.6105454545454514,
    # Exactly 1: the informed approximation coincides with the target itself, so the correction
    # potential is constant in the state (see the notebook's own derivation).
    "informed_acceptance_rate": 1.0,
    "classical_mean_error_norm": 0.09317589138144385,
    "classical_std_ratio_error_norm": 0.048591932621311895,
    "informed_mean_error_norm": 0.03985121889971218,
    "informed_std_ratio_error_norm": 0.014408511557937083,
}

# Expressions evaluated in the pmala notebook's final namespace.
PMALA_RESULT_EXPRESSIONS = {
    "plain_acceptance_rate": "float(plain_acceptance_rate)",
    "informed_acceptance_rate": "float(informed_acceptance_rate)",
    "plain_mean_error_norm": (
        "float(np.linalg.norm((plain_samples.mean(axis=0) - posterior_mean) "
        "/ posterior_standard_deviation))"
    ),
    "plain_std_ratio_error_norm": (
        "float(np.linalg.norm(plain_samples.std(axis=0, ddof=1) "
        "/ posterior_standard_deviation - 1.0))"
    ),
    "informed_mean_error_norm": (
        "float(np.linalg.norm((informed_samples.mean(axis=0) - posterior_mean) "
        "/ posterior_standard_deviation))"
    ),
    "informed_std_ratio_error_norm": (
        "float(np.linalg.norm(informed_samples.std(axis=0, ddof=1) "
        "/ posterior_standard_deviation - 1.0))"
    ),
}
PMALA_REFERENCE_VALUES = {
    "plain_acceptance_rate": 0.5875000000000059,
    "informed_acceptance_rate": 1.0,
    "plain_mean_error_norm": 0.024605982408821385,
    "plain_std_ratio_error_norm": 0.02384009900491648,
    "informed_mean_error_norm": 0.03531781456348716,
    "informed_std_ratio_error_norm": 0.013292478542307685,
}

# Expressions evaluated in the mala notebook's final namespace.
MALA_RESULT_EXPRESSIONS = {
    "acceptance_rate": "float(acceptance_rate)",
    "mean_error_norm": (
        "float(np.linalg.norm((samples.mean(axis=0) - posterior_mean) "
        "/ posterior_standard_deviation))"
    ),
    "std_ratio_error_norm": (
        "float(np.linalg.norm(samples.std(axis=0, ddof=1) / posterior_standard_deviation - 1.0))"
    ),
}
# `plain` above and `mala` here use the identical test problem, seeds, and step width, and
# PMALAAlgorithm with approximation=reference is proven to reduce exactly to MALAAlgorithm
# (test_pmala.py::test_collapses_to_mala_algorithm_when_preconditioner_and_reference_equal_prior),
# so these match PMALA_REFERENCE_VALUES' "plain_*" entries exactly.
MALA_REFERENCE_VALUES = {
    "acceptance_rate": PMALA_REFERENCE_VALUES["plain_acceptance_rate"],
    "mean_error_norm": PMALA_REFERENCE_VALUES["plain_mean_error_norm"],
    "std_ratio_error_norm": PMALA_REFERENCE_VALUES["plain_std_ratio_error_norm"],
}


# ==================================================================================================
@pytest.mark.parametrize(
    ("notebook", "result_expressions", "reference_values"),
    [
        pytest.param(helpers.PCN_NOTEBOOK, PCN_RESULT_EXPRESSIONS, PCN_REFERENCE_VALUES, id="pcn"),
        pytest.param(
            helpers.MALA_NOTEBOOK, MALA_RESULT_EXPRESSIONS, MALA_REFERENCE_VALUES, id="mala"
        ),
        pytest.param(
            helpers.PMALA_NOTEBOOK, PMALA_RESULT_EXPRESSIONS, PMALA_REFERENCE_VALUES, id="pmala"
        ),
    ],
)
def test_tutorial_notebook_runs_and_matches_reference_values(
    notebook: Path,
    result_expressions: dict[str, str],
    reference_values: dict[str, float],
) -> None:
    """Execute a tutorial notebook and check its acceptance rate(s) and posterior moment recovery
    against recorded reference values."""
    result_values = notebook_helpers.execute_notebook_and_extract_values(
        notebook, result_expressions, timeout_seconds=helpers.NOTEBOOK_EXECUTION_TIMEOUT_SECONDS
    )

    for key, expected_value in reference_values.items():
        np.testing.assert_allclose(
            result_values[key], expected_value, rtol=REGRESSION_RELATIVE_TOLERANCE, atol=0
        )
