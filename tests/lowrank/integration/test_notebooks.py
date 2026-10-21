r"""Integration tests that the lowrank tutorial notebook still runs and reproduces its results.

Running a notebook to completion already verifies that it executes without error against the
current `ls_bayesian` API. The checks below compare the notebook's low-rank results with the exact
dense posterior it computes itself, so no recorded reference values are needed.
"""

from pathlib import Path

import pytest

from tests import notebook_helpers

pytestmark = pytest.mark.integration

NOTEBOOK_TIMEOUT_SECONDS = 300
NOTEBOOK_PATH = notebook_helpers.REPO_ROOT / "tutorials" / "lowrank" / "lowrank_gaussian.ipynb"

# Expressions evaluated in the notebook's final namespace, after all of its own cells have run.
RESULT_EXPRESSIONS = {
    "lanczos_eigenvalue_error": (
        "float(np.max(np.abs(lanczos_pairs.eigenvalues "
        "/ dense_eigenvalues[:NUM_EIGENVALUES] - 1.0)))"
    ),
    "double_pass_eigenvalue_error": (
        "float(np.max(np.abs(double_pass_pairs.eigenvalues "
        "/ dense_eigenvalues[:NUM_EIGENVALUES] - 1.0)))"
    ),
    "map_error": "float(map_error)",
    "variance_error": "float(variance_error)",
    "log_determinant_error": (
        "float(abs(log_determinant_ratio - (log_det_posterior - log_det_prior)))"
    ),
    "sample_variance_error": "float(sample_variance_error)",
    "full_rank_variance_error": "float(variance_errors[-1])",
}


# ==================================================================================================
def test_lowrank_notebook_matches_dense_posterior(notebook_path: Path = NOTEBOOK_PATH) -> None:
    values = notebook_helpers.execute_notebook_and_extract_values(
        notebook_path, RESULT_EXPRESSIONS, timeout_seconds=NOTEBOOK_TIMEOUT_SECONDS
    )

    # The rank equals the number of observations, so the approximation is exact up to the
    # accuracy of the eigensolvers.
    assert values["lanczos_eigenvalue_error"] < 1e-6
    assert values["double_pass_eigenvalue_error"] < 1e-6
    assert values["map_error"] < 1e-6
    assert values["variance_error"] < 1e-6
    assert values["full_rank_variance_error"] < 1e-6
    assert values["log_determinant_error"] < 1e-6
    # Five Monte Carlo standard errors of a sample variance from 2000 samples, over 200 nodes.
    assert values["sample_variance_error"] < 5 * (2 / 2000) ** 0.5
