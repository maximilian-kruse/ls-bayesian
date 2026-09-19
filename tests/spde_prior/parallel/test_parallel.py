"""Checks that the SPDE prior works in parallel.

The tests run on `MPI.COMM_WORLD` and are skipped for a single process. Run them with e.g.
`mpirun -n 2 python -m pytest tests/spde_prior/parallel -m spde_prior_parallel -p no:cacheprovider`.

Vertex vectors are given in the input node order of the mesh, which does not depend on the number
of processes. Every rank holds the complete vector. The expected values are the results of the same
computation in serial, so these tests check the parallel semantics (distributed DoFs, ghost updates,
reductions, the process-independent order of the sampling input), not the mathematics of the
prior, which is covered by the serial tests.
"""

from collections.abc import Callable

import dolfinx as dlx
import numpy as np
import pytest
from dolfinx.fem import petsc
from mpi4py import MPI

from ls_bayesian.spde_prior import builder, fem, spde_prior

pytestmark = [
    pytest.mark.spde_prior_parallel,
    pytest.mark.skipif(MPI.COMM_WORLD.size == 1, reason="requires several MPI processes"),
]

MESH_RESOLUTION = 8
FE_DATA = [("Lagrange", 1), ("Lagrange", 2)]
FE_DATA_IDS = ["p1", "p2"]
# Parallel and serial solves use different AMG hierarchies and iterates, so results only agree up
# to the effect of the solver tolerance (1e-12 relative residual) on the solution.
PARALLEL_RELATIVE_TOLERANCE = 1e-8
PARALLEL_ABSOLUTE_TOLERANCE = 1e-10
# Prior operations with vertex-space (or scalar) results, applied to a smooth test field.
PRIOR_OPERATION_NAMES = [
    "evaluate_cost",
    "evaluate_gradient",
    "evaluate_hessian_vector_product",
    "apply_covariance_operator",
    "apply_covariance_factorization",
    "generate_sample",
]


# ==================================================================================================
def _create_mesh(communicator: MPI.Comm) -> dlx.mesh.Mesh:
    return dlx.mesh.create_unit_square(communicator, MESH_RESOLUTION, MESH_RESOLUTION)


def _input_ordered_field(mesh: dlx.mesh.Mesh) -> np.ndarray:
    """Evaluate a smooth field at the mesh vertices, in input node order, on every rank."""
    serial_mesh = _create_mesh(MPI.COMM_SELF)
    input_order = np.argsort(serial_mesh.geometry.input_global_indices)
    coordinates = serial_mesh.geometry.x[input_order]
    assert coordinates.shape[0] == mesh.topology.index_map(0).size_global
    return np.sin(np.pi * coordinates[:, 0]) * np.cos(2 * np.pi * coordinates[:, 1])


def _build_prior(mesh: dlx.mesh.Mesh, fe_data: tuple[str, int]) -> spde_prior.SPDEPrior:
    settings = builder.BilaplacianPriorSettings(
        mesh,
        np.zeros(mesh.topology.index_map(0).size_global),
        kappa=5.0,
        tau=0.5,
        robin_const=1.0,
        seed=0,
        fe_data=fe_data,
    )
    return builder.BilaplacianPriorBuilder(settings).build()


def _apply_prior_operation(
    prior: spde_prior.SPDEPrior, operation_name: str, parameter_vector: np.ndarray
) -> np.ndarray | float:
    """Apply an operation, with a fixed random vector for the covariance factorization."""
    operation: Callable[..., np.ndarray | float] = getattr(prior, operation_name)
    if operation_name == "generate_sample":
        return operation()
    if operation_name == "apply_covariance_factorization":
        return operation(np.random.default_rng(0).standard_normal(prior.random_vector_size))
    return operation(parameter_vector)


def _assert_close(actual: np.ndarray | float, desired: np.ndarray | float) -> None:
    np.testing.assert_allclose(
        actual, desired, rtol=PARALLEL_RELATIVE_TOLERANCE, atol=PARALLEL_ABSOLUTE_TOLERANCE
    )


# ==================================================================================================
@pytest.mark.parametrize("fe_data", FE_DATA, ids=FE_DATA_IDS)
def test_factorization_reproduces_mass_matrix(fe_data: tuple[str, int]) -> None:
    r"""The assembled factors satisfy $L^T \widehat{M}_e \widehat{M}_e^T L = M$ in parallel."""
    mesh = _create_mesh(MPI.COMM_WORLD)
    function_space = dlx.fem.functionspace(mesh, fe_data)
    mass_matrix_form, _ = fem.generate_forms(function_space, kappa=1.0, tau=1.0)
    mass_matrix = petsc.assemble_matrix(dlx.fem.form(mass_matrix_form))
    mass_matrix.assemble()

    assembler = fem.FEMMatrixFactorizationAssembler(mesh, function_space, mass_matrix_form)
    block_diagonal_matrix, dof_map_matrix = assembler.assemble()

    input_vector = mass_matrix.createVecRight()
    ownership_start, ownership_end = input_vector.getOwnershipRange()
    complete_input = np.random.default_rng(0).random(input_vector.getSize())
    input_vector.setArray(complete_input[ownership_start:ownership_end])
    expected_vector = mass_matrix.createVecLeft()
    mass_matrix.mult(input_vector, expected_vector)

    block_vector = dof_map_matrix.createVecLeft()
    dof_map_matrix.mult(input_vector, block_vector)
    factor_transpose_vector = block_diagonal_matrix.createVecRight()
    block_diagonal_matrix.multTranspose(block_vector, factor_transpose_vector)
    block_diagonal_matrix.mult(factor_transpose_vector, block_vector)
    reconstructed_vector = dof_map_matrix.createVecRight()
    dof_map_matrix.multTranspose(block_vector, reconstructed_vector)

    _assert_close(reconstructed_vector.getArray(), expected_vector.getArray())


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fe_data", FE_DATA, ids=FE_DATA_IDS)
def test_fem_converter_round_trip(fe_data: tuple[str, int]) -> None:
    """Vertex values of a P1 field are recovered after conversion to DoFs and back."""
    mesh = _create_mesh(MPI.COMM_WORLD)
    converter = fem.FEMConverter(dlx.fem.functionspace(mesh, fe_data))
    vertex_values = _input_ordered_field(mesh)

    dof_values = converter.convert_vertex_values_to_dofs(vertex_values)
    recovered_vertex_values = converter.convert_dofs_to_vertex_values(dof_values)

    _assert_close(recovered_vertex_values, vertex_values)


# --------------------------------------------------------------------------------------------------
def test_get_ordered_vertices_and_cells_matches_serial() -> None:
    """The geometry returned for a distributed mesh matches a serial reference exactly."""
    parallel_coordinates, parallel_connectivity = fem.get_ordered_vertices_and_cells(
        _create_mesh(MPI.COMM_WORLD)
    )
    serial_coordinates, serial_connectivity = fem.get_ordered_vertices_and_cells(
        _create_mesh(MPI.COMM_SELF)
    )

    _assert_close(parallel_coordinates, serial_coordinates)
    np.testing.assert_array_equal(parallel_connectivity, serial_connectivity)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("operation_name", PRIOR_OPERATION_NAMES)
@pytest.mark.parametrize("fe_data", FE_DATA, ids=FE_DATA_IDS)
def test_parallel_prior_matches_serial_prior(fe_data: tuple[str, int], operation_name: str) -> None:
    """A prior operation gives the same result as for a serial prior on the same mesh."""
    parallel_prior = _build_prior(_create_mesh(MPI.COMM_WORLD), fe_data)
    serial_prior = _build_prior(_create_mesh(MPI.COMM_SELF), fe_data)
    parameter_vector = _input_ordered_field(_create_mesh(MPI.COMM_SELF))

    parallel_result = _apply_prior_operation(parallel_prior, operation_name, parameter_vector)
    serial_result = _apply_prior_operation(serial_prior, operation_name, parameter_vector)

    _assert_close(parallel_result, serial_result)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("operation_name", PRIOR_OPERATION_NAMES)
def test_parallel_results_are_identical_on_all_ranks(operation_name: str) -> None:
    """Replicated outputs must not diverge between processes."""
    prior = _build_prior(_create_mesh(MPI.COMM_WORLD), FE_DATA[0])
    parameter_vector = _input_ordered_field(_create_mesh(MPI.COMM_SELF))

    result = np.asarray(_apply_prior_operation(prior, operation_name, parameter_vector))

    root_result = MPI.COMM_WORLD.bcast(result, root=0)
    np.testing.assert_array_equal(result, root_result)
