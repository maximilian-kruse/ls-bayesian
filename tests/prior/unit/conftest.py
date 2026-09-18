from dataclasses import dataclass

import dolfinx as dlx
import numpy as np
import pytest
from petsc4py import PETSc

import tests.prior.conftest as main_config


# ==================================================================================================
@dataclass
class MatrixAssemblySetup:
    function_space: dlx.fem.FunctionSpace
    kappa: float
    tau: float
    robin_const: float
    expected_mass_matrix: np.ndarray
    expected_spde_matrix: np.ndarray


@dataclass
class FEMConverterSetup:
    function_space: dlx.fem.FunctionSpace
    input_vertex_values: np.ndarray
    expected_dof_values: PETSc.Vec
    expected_output_vertex_values: np.ndarray


@dataclass
class FactorizationAssemblerSetup:
    fem_setup: main_config.FEMSpaceSetup
    expected_mass_matrix: np.ndarray


# ==================================================================================================
@pytest.fixture(scope="session")
def matrix_assembly_setups(
    fem_setup_combinations: list[main_config.FEMSpaceSetup],
    precomputed_assembly_matrices: list[main_config.PrecomputedAssemblyMatrices],
) -> list[MatrixAssemblySetup]:
    kappa = 1.0
    tau = 1.0
    robin_const = None
    setups = []
    for fem_setup, precomputed_results in zip(
        fem_setup_combinations, precomputed_assembly_matrices, strict=True
    ):
        setups.append(
            MatrixAssemblySetup(
                function_space=fem_setup.function_space,
                kappa=kappa,
                tau=tau,
                robin_const=robin_const,
                expected_mass_matrix=precomputed_results.mass_matrix,
                expected_spde_matrix=precomputed_results.spde_matrix,
            )
        )
    return setups


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def fem_converter_setups(
    fem_setup_combinations: list[main_config.FEMSpaceSetup],
    precomputed_converter_vectors: list[main_config.PrecomputedConverterVectors],
) -> list[FEMConverterSetup]:
    setups = []
    for fem_setup, precomputed_results in zip(
        fem_setup_combinations, precomputed_converter_vectors, strict=True
    ):
        setups.append(
            FEMConverterSetup(
                function_space=fem_setup.function_space,
                input_vertex_values=precomputed_results.input_vertex_values,
                expected_dof_values=precomputed_results.dof_values,
                expected_output_vertex_values=precomputed_results.output_vertex_values,
            )
        )
    return setups


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def factorization_assembler_setups(
    fem_setup_combinations: list[main_config.FEMSpaceSetup],
    precomputed_assembly_matrices: list[main_config.PrecomputedAssemblyMatrices],
) -> list[FactorizationAssemblerSetup]:
    setups = []
    for fem_setup, precomputed_results in zip(
        fem_setup_combinations, precomputed_assembly_matrices, strict=True
    ):
        setups.append(
            FactorizationAssemblerSetup(
                fem_setup=fem_setup,
                expected_mass_matrix=precomputed_results.mass_matrix,
            )
        )
    return setups


# ==================================================================================================
@pytest.fixture(params=list(range(main_config.NUM_FEM_SETUPS)), ids=main_config.FEM_SETUP_IDS)
def parametrized_matrix_assembly_setup(
    request: pytest.FixtureRequest, matrix_assembly_setups: list[MatrixAssemblySetup]
) -> MatrixAssemblySetup:
    return matrix_assembly_setups[request.param]


@pytest.fixture(params=list(range(main_config.NUM_FEM_SETUPS)), ids=main_config.FEM_SETUP_IDS)
def parametrized_fem_converter_setup(
    request: pytest.FixtureRequest, fem_converter_setups: list[FEMConverterSetup]
) -> FEMConverterSetup:
    return fem_converter_setups[request.param]


@pytest.fixture(params=list(range(main_config.NUM_FEM_SETUPS)), ids=main_config.FEM_SETUP_IDS)
def parametrized_factorization_assembler_setup(
    request: pytest.FixtureRequest,
    factorization_assembler_setups: list[FactorizationAssemblerSetup],
) -> FactorizationAssemblerSetup:
    return factorization_assembler_setups[request.param]
