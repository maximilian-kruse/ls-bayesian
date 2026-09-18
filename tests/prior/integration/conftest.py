from dataclasses import dataclass

import dolfinx as dlx
import numpy as np
import pytest
from petsc4py import PETSc

import tests.prior.conftest as main_config
from ls_bayesian.prior import builder, components, fem, prior


# ==================================================================================================
@dataclass
class PriorComponentSetup:
    mean_vector: np.ndarray
    precision_array: np.ndarray
    covariance_array: np.ndarray
    sampling_factor_array: np.ndarray
    precision_interface: components.InterfaceComponent
    covariance_interface: components.InterfaceComponent
    sampling_factor_interface: components.InterfaceComponent
    fem_converter: fem.FEMConverter
    mesh: dlx.mesh.Mesh
    function_space: dlx.fem.FunctionSpace


# ==================================================================================================
def compute_covariance_cholesky_factor(
    mass_matrix: np.ndarray,
    spde_matrix: np.ndarray,
) -> PETSc.Mat:
    inverse_spde_matrix = np.linalg.inv(spde_matrix)
    covariance_matrix = inverse_spde_matrix @ mass_matrix @ inverse_spde_matrix
    cholesky_factor = np.linalg.cholesky(covariance_matrix)
    cholesky_factor_petsc = PETSc.Mat().createAIJ(
        size=cholesky_factor.shape, comm=main_config.MPI_COMMUNICATOR
    )
    cholesky_factor_petsc.setUp()
    row_inds = np.arange(cholesky_factor.shape[0], dtype=np.int32)
    col_inds = np.arange(cholesky_factor.shape[1], dtype=np.int32)
    cholesky_factor_petsc.setValues(row_inds, col_inds, cholesky_factor)
    cholesky_factor_petsc.assemble()
    return cholesky_factor, cholesky_factor_petsc


# --------------------------------------------------------------------------------------------------
def set_up_components(
    mass_matrix_petsc: PETSc.Mat, spde_matrix_petsc: PETSc.Mat, sampling_factor_petsc: PETSc.Mat
) -> tuple[
    components.InterfaceComponent,
    components.InterfaceComponent,
    components.InterfaceComponent,
]:
    cg_solver_settings = components.InverseMatrixSolverSettings(
        solver_type=PETSc.KSP.Type.CG,
        preconditioner_type=PETSc.PC.Type.JACOBI,
        relative_tolerance=1e-8,
    )
    amg_solver_settings = components.InverseMatrixSolverSettings(
        solver_type=PETSc.KSP.Type.CG,
        preconditioner_type=PETSc.PC.Type.GAMG,
        relative_tolerance=1e-8,
    )
    mass_matrix_component = components.Matrix(mass_matrix_petsc)
    spde_matrix_component = components.Matrix(spde_matrix_petsc)
    mass_matrix_inverse_component = components.InverseMatrixSolver(
        cg_solver_settings,
        mass_matrix_petsc,
    )
    spde_matrix_inverse_component = components.InverseMatrixSolver(
        amg_solver_settings,
        spde_matrix_petsc,
    )
    precision_component = components.PETScComponentComposition(
        spde_matrix_component,
        mass_matrix_inverse_component,
        spde_matrix_component,
    )
    covariance_component = components.PETScComponentComposition(
        spde_matrix_inverse_component,
        mass_matrix_component,
        spde_matrix_inverse_component,
    )
    sampling_factor_component = components.Matrix(sampling_factor_petsc)
    precision_interface = components.InterfaceComponent(precision_component)
    covariance_interface = components.InterfaceComponent(covariance_component)
    sampling_factor_interface = components.InterfaceComponent(sampling_factor_component)

    return precision_interface, covariance_interface, sampling_factor_interface


# ==================================================================================================
@pytest.fixture(scope="session")
def bilaplacian_component_setups(
    fem_setup_combinations: list[main_config.FEMSpaceSetup],
    matrix_component_setups: list[main_config.MatrixComponentSetup],
) -> list[PriorComponentSetup]:
    component_setups = []

    for fem_setup, matrix_components in zip(
        fem_setup_combinations, matrix_component_setups, strict=True
    ):
        mass_matrix_petsc = matrix_components.mass_matrix_petsc
        spde_matrix_petsc = matrix_components.spde_matrix_petsc
        mass_matrix_array = matrix_components.mass_matrix_array
        spde_matrix_array = matrix_components.spde_matrix_array
        cholesky_factor_array, cholesky_factor_petsc = compute_covariance_cholesky_factor(
            mass_matrix_array, spde_matrix_array
        )
        inverse_mass_array = np.linalg.inv(mass_matrix_array)
        inverse_spde_array = np.linalg.inv(spde_matrix_array)
        precision_array = spde_matrix_array @ inverse_mass_array @ spde_matrix_array
        covariance_array = inverse_spde_array @ mass_matrix_array @ inverse_spde_array
        sampling_factor_array = cholesky_factor_array

        fem_converter = fem.FEMConverter(fem_setup.function_space)
        precision_interface, covariance_interface, sampling_factor_interface = set_up_components(
            mass_matrix_petsc, spde_matrix_petsc, cholesky_factor_petsc
        )
        rng = np.random.default_rng(0)
        mean_array = rng.random(fem_setup.mesh.geometry.x.shape[0])

        component_setups.append(
            PriorComponentSetup(
                mean_vector=mean_array,
                precision_array=precision_array,
                covariance_array=covariance_array,
                sampling_factor_array=sampling_factor_array,
                precision_interface=precision_interface,
                covariance_interface=covariance_interface,
                sampling_factor_interface=sampling_factor_interface,
                fem_converter=fem_converter,
                mesh=fem_setup.mesh,
                function_space=fem_setup.function_space,
            )
        )

    return component_setups


# --------------------------------------------------------------------------------------------------
@pytest.fixture(params=list(range(main_config.NUM_FEM_SETUPS)), ids=main_config.FEM_SETUP_IDS)
def parametrized_bilaplacian_component_setup(
    request: pytest.FixtureRequest,
    bilaplacian_component_setups: list[PriorComponentSetup],
) -> PriorComponentSetup:
    return bilaplacian_component_setups[request.param]


# ==================================================================================================
@pytest.fixture
def prior_build_setup(
    parametrized_bilaplacian_component_setup: PriorComponentSetup,
) -> tuple[prior.Prior, prior.Prior, int]:
    mesh = parametrized_bilaplacian_component_setup.mesh
    function_space = parametrized_bilaplacian_component_setup.function_space
    mean_array = parametrized_bilaplacian_component_setup.mean_vector
    prior_object = prior.Prior(
        mean_array,
        parametrized_bilaplacian_component_setup.precision_interface,
        parametrized_bilaplacian_component_setup.covariance_interface,
        parametrized_bilaplacian_component_setup.sampling_factor_interface,
        parametrized_bilaplacian_component_setup.fem_converter,
        seed=0,
    )

    builder_settings = builder.BilaplacianPriorSettings(
        mesh,
        mean_array,
        kappa=1.0,
        tau=1.0,
        seed=0,
        fe_data=(function_space.ufl_element().family_name, function_space.ufl_element().degree),
        cg_relative_tolerance=1e-8,
        amg_relative_tolerance=1e-8,
    )
    prior_builder = builder.BilaplacianPriorBuilder(builder_settings)
    built_prior_object = prior_builder.build()

    return prior_object, built_prior_object, mesh.geometry.x.shape[0]
