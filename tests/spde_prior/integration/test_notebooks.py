r"""Integration tests that the prior tutorial notebooks still run and reproduce their results.

Running a notebook to completion already verifies that it executes without error against the
current `ls_bayesian` API; a raised exception (e.g. from a renamed class or method) fails a test
here before the reference-value comparison is even reached.
"""

from pathlib import Path

import numpy as np
import pytest

from tests.spde_prior import helpers

pytestmark = pytest.mark.integration

# Expressions evaluated in each notebook's final namespace, after all of its own cells have run.
# Vector-valued results are reduced to their L2 norm, so that reference values below stay small
# numeric literals rather than stored array data.
RESULT_EXPRESSIONS = {
    "cost": "float(cost)",
    "grad_norm": "float(np.linalg.norm(grad))",
    "hvp_norm": "float(np.linalg.norm(hvp))",
    "sample_norm": "float(np.linalg.norm(sample))",
}
# The components notebook's Krylov solvers converge to a relative residual of 1e-6 (the builder
# notebook uses the builder's tighter 1e-12 default), so results are only reproducible to about
# that precision; this leaves headroom for solver iteration counts to vary slightly across
# platforms and BLAS implementations.
REGRESSION_RELATIVE_TOLERANCE = 1e-5

# Reference values recorded from a passing run of each notebook. Update these only after
# confirming, and explaining in the commit message, why the tutorial's numerical results are
# expected to change.
BUILDER_REFERENCE_VALUES = {
    "cost": 272.1929235670469,
    "grad_norm": 11.886363007658126,
    "hvp_norm": 17.82954451148719,
    "sample_norm": 53.52361095021013,
}
COMPONENTS_REFERENCE_VALUES = {
    "cost": 4188594.5802976065,
    "grad_norm": 705744.8210442518,
    "hvp_norm": 1045649.2839584103,
    "sample_norm": 34.834708464893104,
}


# ==================================================================================================
@pytest.mark.slow
@pytest.mark.parametrize(
    ("notebook_path", "reference_values"),
    [
        (helpers.BUILDER_NOTEBOOK, BUILDER_REFERENCE_VALUES),
        (helpers.COMPONENTS_NOTEBOOK, COMPONENTS_REFERENCE_VALUES),
    ],
    ids=["builder", "components"],
)
def test_prior_tutorial_notebook_runs_and_matches_reference_values(
    notebook_path: Path, reference_values: dict[str, float]
) -> None:
    """Execute a prior tutorial notebook and check its cost/gradient/HVP/sample results."""
    result_values = helpers.execute_notebook_and_extract_values(notebook_path, RESULT_EXPRESSIONS)

    for key, expected_value in reference_values.items():
        np.testing.assert_allclose(
            result_values[key], expected_value, rtol=REGRESSION_RELATIVE_TOLERANCE, atol=0
        )
