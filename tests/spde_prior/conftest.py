import numpy as np
import pytest

from ls_bayesian.spde_prior import fem
from tests.spde_prior import helpers


# ==================================================================================================
@pytest.fixture(scope="session", params=helpers.FEM_CASE_IDS, ids=helpers.FEM_CASE_IDS)
def fem_case(request: pytest.FixtureRequest) -> helpers.FEMCase:
    """FEM test case, restrict via `pytest.mark.parametrize("fem_case", ids, indirect=True)`."""
    return helpers.FEM_CASES[request.param]


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def fem_space_setup(fem_case: helpers.FEMCase) -> helpers.FEMSpaceSetup:
    """Mesh and function space, session-scoped as they are not modified by the tests."""
    return helpers.create_fem_space_setup(fem_case, helpers.MESH_COMMUNICATOR)


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def assembled_matrices(fem_space_setup: helpers.FEMSpaceSetup) -> helpers.AssembledMatrices:
    """Dense mass and SPDE matrices, read-only, session-scoped as they are immutable."""
    function_space = fem_space_setup.function_space
    mass_form, neumann_spde_form = fem.generate_forms(function_space, helpers.KAPPA, helpers.TAU)
    _, robin_spde_form = fem.generate_forms(
        function_space, helpers.KAPPA, helpers.TAU, helpers.ROBIN_CONSTANT
    )
    matrices = helpers.AssembledMatrices(
        mass_matrix=helpers.assemble_dense_matrix(mass_form),
        neumann_spde_matrix=helpers.assemble_dense_matrix(neumann_spde_form),
        robin_spde_matrix=helpers.assemble_dense_matrix(robin_spde_form),
    )
    for matrix in (matrices.mass_matrix, matrices.neumann_spde_matrix, matrices.robin_spde_matrix):
        helpers.assert_condition_number_bounded(matrix)
        matrix.setflags(write=False)
    jacobi_preconditioned_mass_matrix = (
        matrices.mass_matrix / np.diag(matrices.mass_matrix)[:, None]
    )
    helpers.assert_condition_number_bounded(jacobi_preconditioned_mass_matrix)
    return matrices
