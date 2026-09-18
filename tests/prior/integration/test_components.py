import numpy as np
import pytest
from petsc4py import PETSc

import tests.prior.conftest as main_config
from ls_bayesian.prior import components

pytestmark = pytest.mark.integration


# ==================================================================================================
def test_bilaplacian_precision(
    parametrized_matrix_component_setup: main_config.MatrixComponentSetup,
) -> None:
    mass_matrix_array = parametrized_matrix_component_setup.mass_matrix_array
    spde_matrix_array = parametrized_matrix_component_setup.spde_matrix_array
    mass_matrix_petsc = parametrized_matrix_component_setup.mass_matrix_petsc
    spde_matrix_petsc = parametrized_matrix_component_setup.spde_matrix_petsc
    input_array = parametrized_matrix_component_setup.input_array
    input_vector = parametrized_matrix_component_setup.input_vector

    bilaplacian_precision_array = (
        spde_matrix_array @ np.linalg.inv(mass_matrix_array) @ spde_matrix_array
    )
    solver_settings = components.InverseMatrixSolverSettings(
        solver_type=PETSc.KSP.Type.CG,
        preconditioner_type=PETSc.PC.Type.JACOBI,
        relative_tolerance=1e-8,
    )
    mass_matrix_inverse_component = components.InverseMatrixSolver(
        solver_settings,
        mass_matrix_petsc,
    )
    spde_matrix_component = components.Matrix(spde_matrix_petsc)
    bilaplacian_precision_component = components.PETScComponentComposition(
        spde_matrix_component,
        mass_matrix_inverse_component,
        spde_matrix_component,
    )

    output_vector = bilaplacian_precision_component.create_output_vector()
    bilaplacian_precision_component.apply(input_vector, output_vector)
    expected_output = bilaplacian_precision_array @ input_array
    assert np.allclose(output_vector.getArray(), expected_output)


# --------------------------------------------------------------------------------------------------
def test_bilaplacian_covariance(
    parametrized_matrix_component_setup: main_config.MatrixComponentSetup,
) -> None:
    mass_matrix_array = parametrized_matrix_component_setup.mass_matrix_array
    spde_matrix_array = parametrized_matrix_component_setup.spde_matrix_array
    mass_matrix_petsc = parametrized_matrix_component_setup.mass_matrix_petsc
    spde_matrix_petsc = parametrized_matrix_component_setup.spde_matrix_petsc
    input_array = parametrized_matrix_component_setup.input_array
    input_vector = parametrized_matrix_component_setup.input_vector

    inverse_spde_matrix_array = np.linalg.inv(spde_matrix_array)
    bilaplacian_covariance_array = (
        inverse_spde_matrix_array @ mass_matrix_array @ inverse_spde_matrix_array
    )
    solver_settings = components.InverseMatrixSolverSettings(
        solver_type=PETSc.KSP.Type.CG,
        preconditioner_type=PETSc.PC.Type.GAMG,
        relative_tolerance=1e-8,
    )
    spde_matrix_inverse_component = components.InverseMatrixSolver(
        solver_settings,
        spde_matrix_petsc,
    )
    mass_matrix_component = components.Matrix(mass_matrix_petsc)
    bilaplacian_covariance_component = components.PETScComponentComposition(
        spde_matrix_inverse_component,
        mass_matrix_component,
        spde_matrix_inverse_component,
    )

    output_vector = bilaplacian_covariance_component.create_output_vector()
    bilaplacian_covariance_component.apply(input_vector, output_vector)
    expected_output = bilaplacian_covariance_array @ input_array
    assert np.allclose(output_vector.getArray(), expected_output)
