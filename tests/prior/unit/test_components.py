import numpy as np
import pytest
from petsc4py import PETSc

import tests.prior.conftest as main_config
from ls_bayesian.prior import components

pytestmark = pytest.mark.unit


# ==================================================================================================
def test_matrix_component(
    parametrized_matrix_component_setup: main_config.MatrixComponentSetup,
) -> None:
    numpy_matrix = parametrized_matrix_component_setup.mass_matrix_array
    petsc_matrix = parametrized_matrix_component_setup.mass_matrix_petsc
    input_array = parametrized_matrix_component_setup.input_array
    input_vector = parametrized_matrix_component_setup.input_vector

    matrix_component = components.Matrix(petsc_matrix)
    output_vector = matrix_component.create_output_vector()
    matrix_component.apply(input_vector, output_vector)
    expected_output = numpy_matrix @ input_array
    assert np.allclose(output_vector.getArray(), expected_output)


# --------------------------------------------------------------------------------------------------
def test_inverse_cg_solver_component(
    parametrized_matrix_component_setup: main_config.MatrixComponentSetup,
) -> None:
    mass_matrix_array = parametrized_matrix_component_setup.mass_matrix_array
    inverse_mass_matrix = np.linalg.inv(mass_matrix_array)
    mass_matrix_petsc = parametrized_matrix_component_setup.mass_matrix_petsc
    input_array = parametrized_matrix_component_setup.input_array
    input_vector = parametrized_matrix_component_setup.input_vector

    cg_solver_settings = components.InverseMatrixSolverSettings(
        solver_type=PETSc.KSP.Type.CG,
        preconditioner_type=PETSc.PC.Type.JACOBI,
        relative_tolerance=1e-8,
    )

    cg_solver_component = components.InverseMatrixSolver(cg_solver_settings, mass_matrix_petsc)
    output_vector = cg_solver_component.create_output_vector()
    cg_solver_component.apply(input_vector, output_vector)
    expected_output = inverse_mass_matrix @ input_array
    assert np.allclose(output_vector.getArray(), expected_output)


# --------------------------------------------------------------------------------------------------
def test_inverse_amg_solver_component(
    parametrized_matrix_component_setup: main_config.MatrixComponentSetup,
) -> None:
    spde_matrix_array = parametrized_matrix_component_setup.spde_matrix_array
    inverse_spde_matrix = np.linalg.inv(spde_matrix_array)
    spde_matrix_petsc = parametrized_matrix_component_setup.spde_matrix_petsc
    input_array = parametrized_matrix_component_setup.input_array
    input_vector = parametrized_matrix_component_setup.input_vector

    amg_solver_settings = components.InverseMatrixSolverSettings(
        solver_type=PETSc.KSP.Type.CG,
        preconditioner_type=PETSc.PC.Type.GAMG,
        relative_tolerance=1e-8,
    )

    amg_solver_component = components.InverseMatrixSolver(amg_solver_settings, spde_matrix_petsc)
    output_vector = amg_solver_component.create_output_vector()
    amg_solver_component.apply(input_vector, output_vector)
    expected_output = inverse_spde_matrix @ input_array
    assert np.allclose(output_vector.getArray(), expected_output)
