import dolfinx as dlx
import numpy as np
import pytest
import ufl
from dolfinx.fem import petsc

import tests.prior.conftest as main_config
from ls_bayesian.prior import fem

pytestmark = pytest.mark.unit


# ==================================================================================================
def test_matrix_assembly(
    parametrized_matrix_assembly_setup: main_config.MatrixAssemblySetup,
) -> None:
    mass_matrix_form, spde_matrix_form = fem.generate_forms(
        parametrized_matrix_assembly_setup.function_space,
        parametrized_matrix_assembly_setup.kappa,
        parametrized_matrix_assembly_setup.tau,
        parametrized_matrix_assembly_setup.robin_const,
    )

    mass_matrix, spde_matrix = (
        petsc.assemble_matrix(dlx.fem.form(mass_matrix_form)),
        petsc.assemble_matrix(dlx.fem.form(spde_matrix_form)),
    )
    mass_matrix.assemble()
    spde_matrix.assemble()
    mass_matrix_array = mass_matrix.getValues(
        np.arange(mass_matrix.getSize()[0], dtype=np.int32),
        np.arange(mass_matrix.getSize()[1], dtype=np.int32),
    )
    spde_matrix_array = spde_matrix.getValues(
        np.arange(spde_matrix.getSize()[0], dtype=np.int32),
        np.arange(spde_matrix.getSize()[1], dtype=np.int32),
    )
    assert np.allclose(mass_matrix_array, parametrized_matrix_assembly_setup.expected_mass_matrix)
    assert np.allclose(spde_matrix_array, parametrized_matrix_assembly_setup.expected_spde_matrix)


# --------------------------------------------------------------------------------------------------
def test_fem_converter_vertex_to_dofs(
    parametrized_fem_converter_setup: main_config.FEMConverterSetup,
) -> None:
    converter = fem.FEMConverter(parametrized_fem_converter_setup.function_space)
    dof_values = converter.convert_vertex_values_to_dofs(
        parametrized_fem_converter_setup.input_vertex_values
    )
    assert np.allclose(dof_values, parametrized_fem_converter_setup.expected_dof_values)


# --------------------------------------------------------------------------------------------------
def test_fem_converter_dofs_to_vertex(
    parametrized_fem_converter_setup: main_config.FEMConverterSetup,
) -> None:
    converter = fem.FEMConverter(parametrized_fem_converter_setup.function_space)
    vertex_values = converter.convert_dofs_to_vertex_values(
        parametrized_fem_converter_setup.expected_dof_values
    )
    assert np.allclose(
        vertex_values, parametrized_fem_converter_setup.expected_output_vertex_values
    )


# --------------------------------------------------------------------------------------------------
def test_matrix_factorization_assembler(
    parametrized_factorization_assembler_setup: main_config.FactorizationAssemblerSetup,
) -> None:
    mesh = parametrized_factorization_assembler_setup.fem_setup.mesh
    function_space = parametrized_factorization_assembler_setup.fem_setup.function_space
    trial_function = ufl.TrialFunction(function_space)
    test_function = ufl.TestFunction(function_space)
    weak_form = ufl.inner(trial_function, test_function) * ufl.dx

    factorization_assembler = fem.FEMMatrixFactorizationAssembler(mesh, function_space, weak_form)
    block_diagonal_matrix, dof_map_matrix = factorization_assembler.assemble()
    dof_map_matrix.transpose()
    matrix_factor = dof_map_matrix.matMult(block_diagonal_matrix)
    transposed_matrix_factor = matrix_factor.copy()
    transposed_matrix_factor.transpose()
    reconstructed_matrix = matrix_factor.matMult(transposed_matrix_factor)
    reconstructed_matrix.assemble()
    reconstructed_array = reconstructed_matrix.getValues(
        np.arange(reconstructed_matrix.getSize()[0], dtype=np.int32),
        np.arange(reconstructed_matrix.getSize()[1], dtype=np.int32),
    )

    assert np.allclose(
        reconstructed_array, parametrized_factorization_assembler_setup.expected_mass_matrix
    )
