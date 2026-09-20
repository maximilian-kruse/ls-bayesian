import basix.ufl
import dolfinx as dlx
import numpy as np
import pytest
import ufl
from beartype.roar import BeartypeCallHintViolation
from mpi4py import MPI

from ls_bayesian.spde_prior import fem
from tests.spde_prior import helpers

pytestmark = pytest.mark.unit


# ==================================================================================================
def field_values(field_name: str, coordinates: np.ndarray) -> np.ndarray:
    """Polynomial test fields with known integrals, evaluated at the given coordinates."""
    if field_name == "constant":
        return np.ones(coordinates.shape[0])
    return coordinates[:, 0]


def spde_energy(field_name: str, fem_case: helpers.FEMCase, robin_const: float | None) -> float:
    r"""Closed form of $f^T A f$ for the test fields $f \in \{1, x_0\}$.

    $f^T A f = \kappa^2 \tau \int f^2 + \tau \int |\nabla f|^2 + \beta \int_{\partial\Omega} f^2$.
    """
    beta = 0.0 if robin_const is None else robin_const
    kappa_squared_tau = helpers.KAPPA**2 * helpers.TAU
    if field_name == "constant":
        return kappa_squared_tau * fem_case.domain_measure + beta * fem_case.boundary_measure
    return (
        kappa_squared_tau * helpers.X0_SQUARED_DOMAIN_INTEGRAL
        + helpers.TAU * helpers.X0_GRADIENT_SQUARED_DOMAIN_INTEGRAL
        + beta * fem_case.x0_boundary_integral
    )


def create_non_affine_interval_mesh() -> dlx.mesh.Mesh:
    """Unit interval with a single cell and quadratic coordinate element."""
    coordinates = np.array([[0.0], [0.5], [1.0]])
    cells = np.array([[0, 2, 1]], dtype=np.int64)
    coordinate_element = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 2, shape=(1,)))
    return dlx.mesh.create_mesh(helpers.MESH_COMMUNICATOR, cells, coordinate_element, coordinates)


# ==================================================================================================
@pytest.mark.parametrize("field_name", ["constant", "x0"])
def test_generate_forms_mass_matrix_integrates_polynomials_exactly(
    fem_space_setup: helpers.FEMSpaceSetup, field_name: str
) -> None:
    mass_form, _ = fem.generate_forms(fem_space_setup.function_space, helpers.KAPPA, helpers.TAU)
    field = field_values(field_name, fem_space_setup.function_space.tabulate_dof_coordinates())

    mass_matrix = helpers.assemble_dense_matrix(mass_form)

    expected_integral = (
        fem_space_setup.fem_case.domain_measure
        if field_name == "constant"
        else helpers.X0_SQUARED_DOMAIN_INTEGRAL
    )
    helpers.assert_allclose_normwise(
        field @ mass_matrix @ field, expected_integral, helpers.ROUNDOFF_TOLERANCE
    )


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("field_name", ["constant", "x0"])
@pytest.mark.parametrize("robin_const", [None, helpers.ROBIN_CONSTANT], ids=["neumann", "robin"])
def test_generate_forms_spde_energy_matches_closed_form(
    fem_space_setup: helpers.FEMSpaceSetup, field_name: str, robin_const: float | None
) -> None:
    _, spde_form = fem.generate_forms(
        fem_space_setup.function_space, helpers.KAPPA, helpers.TAU, robin_const
    )
    field = field_values(field_name, fem_space_setup.function_space.tabulate_dof_coordinates())

    spde_matrix = helpers.assemble_dense_matrix(spde_form)

    expected_energy = spde_energy(field_name, fem_space_setup.fem_case, robin_const)
    helpers.assert_allclose_normwise(
        field @ spde_matrix @ field, expected_energy, helpers.ROUNDOFF_TOLERANCE
    )


# --------------------------------------------------------------------------------------------------
def test_generate_forms_neumann_spde_matrix_maps_constants_to_scaled_mass(
    assembled_matrices: helpers.AssembledMatrices,
) -> None:
    """The stiffness term annihilates constants, so that $A 1 = \\kappa^2 \\tau M 1$."""
    constant_vector = np.ones(assembled_matrices.mass_matrix.shape[0])

    spde_applied = assembled_matrices.neumann_spde_matrix @ constant_vector

    expected = helpers.KAPPA**2 * helpers.TAU * assembled_matrices.mass_matrix @ constant_vector
    helpers.assert_allclose_normwise(spde_applied, expected, helpers.ROUNDOFF_TOLERANCE)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("matrix_name", ["mass_matrix", "neumann_spde_matrix", "robin_spde_matrix"])
def test_generate_forms_matrices_are_symmetric_positive_definite(
    assembled_matrices: helpers.AssembledMatrices, matrix_name: str
) -> None:
    matrix = getattr(assembled_matrices, matrix_name)

    helpers.assert_allclose_normwise(matrix, matrix.T, helpers.ROUNDOFF_TOLERANCE)
    assert np.linalg.eigvalsh(matrix).min() > 0


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kappa", "tau", "robin_const"),
    [(0.0, 1.0, None), (1.0, -1.0, None), (1.0, 1.0, -0.1)],
    ids=["kappa_zero", "tau_negative", "robin_negative"],
)
def test_generate_forms_rejects_invalid_parameters(
    kappa: float, tau: float, robin_const: float | None
) -> None:
    mesh = helpers.create_unit_interval_mesh(helpers.MESH_COMMUNICATOR)
    function_space = dlx.fem.functionspace(mesh, ("Lagrange", 1))

    with pytest.raises(BeartypeCallHintViolation):
        fem.generate_forms(function_space, kappa, tau, robin_const)


# ==================================================================================================
def cell_measure(coordinates: np.ndarray, connectivity: np.ndarray) -> float:
    """Sum of cell measures (segment length in 1D, triangle area in 2D) for an affine mesh.

    This is invariant under any consistent relabeling of vertices and cells, so it can compare a
    (coordinates, connectivity) pair against a differently-ordered but geometrically identical one.
    """
    corners = coordinates[connectivity]
    if connectivity.shape[1] == 2:
        return float(np.linalg.norm(corners[:, 1] - corners[:, 0], axis=-1).sum())
    edge_1, edge_2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    return float(0.5 * np.linalg.norm(np.cross(edge_1, edge_2), axis=-1).sum())


def test_get_ordered_vertices_and_cells_vertex_coordinates(
    fem_space_setup: helpers.FEMSpaceSetup,
) -> None:
    coordinates, _ = fem.get_ordered_vertices_and_cells(fem_space_setup.mesh)

    expected = helpers.input_ordered_vertex_coordinates(fem_space_setup.mesh)
    helpers.assert_allclose_normwise(coordinates, expected, helpers.ROUNDOFF_TOLERANCE)


# --------------------------------------------------------------------------------------------------
def test_get_ordered_vertices_and_cells_connectivity_shape(
    fem_space_setup: helpers.FEMSpaceSetup,
) -> None:
    mesh = fem_space_setup.mesh
    coordinates, connectivity = fem.get_ordered_vertices_and_cells(mesh)

    num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
    num_cell_vertices = mesh.topology.dim + 1
    assert connectivity.shape == (num_cells, num_cell_vertices)
    assert connectivity.min() >= 0
    assert connectivity.max() < coordinates.shape[0]


# --------------------------------------------------------------------------------------------------
def test_get_ordered_vertices_and_cells_preserves_cell_measure(
    fem_space_setup: helpers.FEMSpaceSetup,
) -> None:
    """A relabeling of vertices and cells must not change the mesh's total length/area."""
    mesh = fem_space_setup.mesh
    coordinates, connectivity = fem.get_ordered_vertices_and_cells(mesh)

    reordered_measure = cell_measure(coordinates, connectivity)
    raw_connectivity = mesh.geometry.dofmaps[0][
        : mesh.topology.index_map(mesh.topology.dim).size_local
    ]
    raw_measure = cell_measure(mesh.geometry.x, raw_connectivity)
    helpers.assert_allclose_normwise(reordered_measure, raw_measure, helpers.ROUNDOFF_TOLERANCE)


# --------------------------------------------------------------------------------------------------
def test_get_ordered_vertices_and_cells_rejects_non_affine_geometry() -> None:
    mesh = create_non_affine_interval_mesh()

    with pytest.raises(ValueError, match="affine mesh geometry"):
        fem.get_ordered_vertices_and_cells(mesh)


# ==================================================================================================
def test_fem_converter_vertex_to_dofs_reproduces_linear_field(
    fem_space_setup: helpers.FEMSpaceSetup,
) -> None:
    converter = fem.FEMConverter(fem_space_setup.function_space)
    vertex_coordinates = helpers.input_ordered_vertex_coordinates(fem_space_setup.mesh)

    dof_values = converter.convert_vertex_values_to_dofs(helpers.linear_field(vertex_coordinates))

    expected = helpers.linear_field(fem_space_setup.function_space.tabulate_dof_coordinates())
    helpers.assert_allclose_normwise(dof_values, expected, helpers.ROUNDOFF_TOLERANCE)


# --------------------------------------------------------------------------------------------------
def test_fem_converter_dofs_to_vertex_reproduces_linear_field(
    fem_space_setup: helpers.FEMSpaceSetup,
) -> None:
    converter = fem.FEMConverter(fem_space_setup.function_space)
    dof_coordinates = fem_space_setup.function_space.tabulate_dof_coordinates()

    vertex_values = converter.convert_dofs_to_vertex_values(helpers.linear_field(dof_coordinates))

    expected = helpers.linear_field(helpers.input_ordered_vertex_coordinates(fem_space_setup.mesh))
    helpers.assert_allclose_normwise(vertex_values, expected, helpers.ROUNDOFF_TOLERANCE)


# --------------------------------------------------------------------------------------------------
def test_fem_converter_vertex_round_trip_is_identity(
    fem_space_setup: helpers.FEMSpaceSetup,
) -> None:
    converter = fem.FEMConverter(fem_space_setup.function_space)
    vertex_values = np.random.default_rng(0).random(converter.global_vertex_space_dim)

    round_trip_values = converter.convert_dofs_to_vertex_values(
        converter.convert_vertex_values_to_dofs(vertex_values)
    )

    helpers.assert_allclose_normwise(round_trip_values, vertex_values, helpers.ROUNDOFF_TOLERANCE)


# --------------------------------------------------------------------------------------------------
def test_fem_converter_convert_vertex_values_to_dof_adjoint_satisfies_adjoint_identity(
    fem_space_setup: helpers.FEMSpaceSetup,
) -> None:
    r"""$\langle I^T d, v \rangle = \langle d, I v \rangle$ defines $I^T$ as the adjoint of $I$."""
    converter = fem.FEMConverter(fem_space_setup.function_space)
    rng = np.random.default_rng(0)
    vertex_vector = rng.random(converter.global_vertex_space_dim)
    dof_covector = rng.random(converter.local_dof_space_dim)

    pulled_back_inner_product = (
        converter.convert_vertex_values_to_dof_adjoint(dof_covector) @ vertex_vector
    )

    forward_inner_product = dof_covector @ converter.convert_vertex_values_to_dofs(vertex_vector)
    helpers.assert_allclose_normwise(
        pulled_back_inner_product, forward_inner_product, helpers.ROUNDOFF_TOLERANCE
    )


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fem_case", helpers.P1_CASE_IDS, indirect=True)
def test_fem_converter_convert_vertex_values_to_dof_adjoint_matches_dofs_to_vertex_for_p1(
    fem_space_setup: helpers.FEMSpaceSetup,
) -> None:
    """For a P1 space, the vertex-to-DoF interpolation is a permutation, so its adjoint (the
    transpose of a permutation matrix) coincides with the plain DoF-to-vertex conversion.
    """
    converter = fem.FEMConverter(fem_space_setup.function_space)
    dof_covector = np.random.default_rng(0).random(converter.local_dof_space_dim)

    pulled_back = converter.convert_vertex_values_to_dof_adjoint(dof_covector)

    helpers.assert_allclose_normwise(
        pulled_back,
        converter.convert_dofs_to_vertex_values(dof_covector),
        helpers.ROUNDOFF_TOLERANCE,
    )


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fem_case", helpers.P2_CASE_IDS, indirect=True)
def test_fem_converter_convert_vertex_values_to_dof_adjoint_differs_from_dofs_to_vertex_for_p2(
    fem_space_setup: helpers.FEMSpaceSetup,
) -> None:
    """For a P2 space, the vertex-to-DoF interpolation also sets edge-midpoint DoFs from vertex
    values, so it is not square, and its adjoint genuinely differs from the DoF-to-vertex
    conversion (which merely reads off a DoF-space field's own value at each vertex).
    """
    converter = fem.FEMConverter(fem_space_setup.function_space)
    dof_covector = np.random.default_rng(0).random(converter.local_dof_space_dim)

    pulled_back = converter.convert_vertex_values_to_dof_adjoint(dof_covector)
    converted = converter.convert_dofs_to_vertex_values(dof_covector)

    assert not np.allclose(pulled_back, converted)


# --------------------------------------------------------------------------------------------------
def test_fem_converter_dimensions(fem_space_setup: helpers.FEMSpaceSetup) -> None:
    converter = fem.FEMConverter(fem_space_setup.function_space)

    fem_case = fem_space_setup.fem_case
    assert converter.global_vertex_space_dim == fem_case.num_vertices
    assert converter.local_dof_space_dim == fem_case.num_dofs
    assert converter.global_dof_space_dim == fem_case.num_dofs


# --------------------------------------------------------------------------------------------------
def test_fem_converter_cell_block_indices_form_permutation(
    fem_space_setup: helpers.FEMSpaceSetup,
) -> None:
    function_space = fem_space_setup.function_space
    converter = fem.FEMConverter(function_space)

    mesh = fem_space_setup.mesh
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
    num_block_entries = num_cells * function_space.dofmap.dof_layout.num_dofs
    np.testing.assert_array_equal(
        np.sort(converter.cell_block_indices), np.arange(num_block_entries)
    )


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "direction", ["vertex_to_dofs", "dofs_to_vertex", "convert_vertex_values_to_dof_adjoint"]
)
@pytest.mark.parametrize("shape_change", ["one_too_long", "column_vector"])
def test_fem_converter_rejects_wrong_shape(direction: str, shape_change: str) -> None:
    mesh = helpers.create_unit_interval_mesh(helpers.MESH_COMMUNICATOR)
    converter = fem.FEMConverter(dlx.fem.functionspace(mesh, ("Lagrange", 1)))
    if direction == "vertex_to_dofs":
        convert, size = converter.convert_vertex_values_to_dofs, converter.global_vertex_space_dim
    elif direction == "dofs_to_vertex":
        convert, size = converter.convert_dofs_to_vertex_values, converter.local_dof_space_dim
    else:
        convert = converter.convert_vertex_values_to_dof_adjoint
        size = converter.local_dof_space_dim
    shape = (size + 1,) if shape_change == "one_too_long" else (size, 1)

    with pytest.raises(ValueError, match=r"Expected \w+ to have shape"):
        convert(np.zeros(shape))


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "invalid_input",
    [np.arange(4), np.zeros(4, dtype=np.float32), [0.0] * 4],
    ids=["int64", "float32", "list"],
)
def test_fem_converter_rejects_non_float64_input(invalid_input: object) -> None:
    mesh = helpers.create_unit_interval_mesh(helpers.MESH_COMMUNICATOR)
    converter = fem.FEMConverter(dlx.fem.functionspace(mesh, ("Lagrange", 1)))

    with pytest.raises(BeartypeCallHintViolation):
        converter.convert_vertex_values_to_dofs(invalid_input)


# --------------------------------------------------------------------------------------------------
def test_fem_converter_rejects_vector_space() -> None:
    mesh = helpers.create_unit_square_mesh(helpers.MESH_COMMUNICATOR)
    vector_space = dlx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))

    with pytest.raises(ValueError, match="scalar function space"):
        fem.FEMConverter(vector_space)


# --------------------------------------------------------------------------------------------------
def test_fem_converter_rejects_non_affine_geometry() -> None:
    mesh = create_non_affine_interval_mesh()

    with pytest.raises(ValueError, match="affine mesh geometry"):
        fem.FEMConverter(dlx.fem.functionspace(mesh, ("Lagrange", 1)))


# --------------------------------------------------------------------------------------------------
def test_fem_converter_leaves_input_unchanged(fem_space_setup: helpers.FEMSpaceSetup) -> None:
    converter = fem.FEMConverter(fem_space_setup.function_space)
    rng = np.random.default_rng(0)
    vertex_values = rng.random(converter.global_vertex_space_dim)
    dof_values = rng.random(converter.local_dof_space_dim)
    vertex_values_copy, dof_values_copy = vertex_values.copy(), dof_values.copy()

    converter.convert_vertex_values_to_dofs(vertex_values)
    converter.convert_dofs_to_vertex_values(dof_values)
    converter.convert_vertex_values_to_dof_adjoint(dof_values)

    np.testing.assert_array_equal(vertex_values, vertex_values_copy)
    np.testing.assert_array_equal(dof_values, dof_values_copy)


# ==================================================================================================
@pytest.mark.parametrize("form_name", ["mass", "neumann_spde"])
def test_factorization_reproduces_assembled_matrix(
    fem_space_setup: helpers.FEMSpaceSetup, form_name: str
) -> None:
    mass_form, spde_form = fem.generate_forms(
        fem_space_setup.function_space, helpers.KAPPA, helpers.TAU
    )
    form = mass_form if form_name == "mass" else spde_form
    assembler = fem.FEMMatrixFactorizationAssembler(
        fem_space_setup.mesh, fem_space_setup.function_space, form
    )

    block_diagonal_matrix, dof_map_matrix = assembler.assemble()

    block_diagonal_array = helpers.dense_from_petsc_matrix(block_diagonal_matrix)
    dof_map_array = helpers.dense_from_petsc_matrix(dof_map_matrix)
    reconstructed_matrix = (
        dof_map_array.T @ block_diagonal_array @ block_diagonal_array.T @ dof_map_array
    )
    helpers.assert_allclose_normwise(
        reconstructed_matrix, helpers.assemble_dense_matrix(form), helpers.ROUNDOFF_TOLERANCE
    )


# --------------------------------------------------------------------------------------------------
def test_factorization_dof_map_has_single_unit_entry_per_row(
    fem_space_setup: helpers.FEMSpaceSetup,
) -> None:
    mass_form, _ = fem.generate_forms(fem_space_setup.function_space, helpers.KAPPA, helpers.TAU)
    assembler = fem.FEMMatrixFactorizationAssembler(
        fem_space_setup.mesh, fem_space_setup.function_space, mass_form
    )

    _, dof_map_matrix = assembler.assemble()

    dof_map_array = helpers.dense_from_petsc_matrix(dof_map_matrix)
    np.testing.assert_array_equal(np.count_nonzero(dof_map_array, axis=1), 1)
    np.testing.assert_array_equal(dof_map_array.sum(axis=1), 1.0)
    assert np.all(np.count_nonzero(dof_map_array, axis=0) >= 1)


# --------------------------------------------------------------------------------------------------
def test_factorization_shapes(fem_space_setup: helpers.FEMSpaceSetup) -> None:
    mesh, function_space = fem_space_setup.mesh, fem_space_setup.function_space
    mass_form, _ = fem.generate_forms(function_space, helpers.KAPPA, helpers.TAU)
    assembler = fem.FEMMatrixFactorizationAssembler(mesh, function_space, mass_form)

    block_diagonal_matrix, dof_map_matrix = assembler.assemble()

    num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
    num_block_entries = num_cells * function_space.dofmap.dof_layout.num_dofs
    assert block_diagonal_matrix.getSize() == (num_block_entries, num_block_entries)
    assert dof_map_matrix.getSize() == (num_block_entries, fem_space_setup.fem_case.num_dofs)


# --------------------------------------------------------------------------------------------------
def test_factorization_rejects_vector_space() -> None:
    mesh = helpers.create_unit_square_mesh(helpers.MESH_COMMUNICATOR)
    vector_space = dlx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    form = ufl.inner(ufl.TrialFunction(vector_space), ufl.TestFunction(vector_space)) * ufl.dx

    with pytest.raises(ValueError, match="scalar function space"):
        fem.FEMMatrixFactorizationAssembler(mesh, vector_space, form)


# --------------------------------------------------------------------------------------------------
def test_factorization_rejects_float32_mesh() -> None:
    mesh = dlx.mesh.create_unit_interval(MPI.COMM_SELF, 3, dtype=np.float32)
    function_space = dlx.fem.functionspace(mesh, ("Lagrange", 1))
    form = ufl.inner(ufl.TrialFunction(function_space), ufl.TestFunction(function_space)) * ufl.dx

    with pytest.raises(TypeError, match="float64 mesh coordinates"):
        fem.FEMMatrixFactorizationAssembler(mesh, function_space, form)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("integral", ["boundary", "cell_subdomain"])
def test_factorization_rejects_unsupported_integrals(integral: str) -> None:
    """Integrals other than cell integrals over the whole domain used to be dropped silently."""
    mesh = helpers.create_unit_interval_mesh(helpers.MESH_COMMUNICATOR)
    function_space = dlx.fem.functionspace(mesh, ("Lagrange", 1))
    _, robin_spde_form = fem.generate_forms(
        function_space, helpers.KAPPA, helpers.TAU, helpers.ROBIN_CONSTANT
    )
    trial_function = ufl.TrialFunction(function_space)
    test_function = ufl.TestFunction(function_space)
    form = (
        robin_spde_form
        if integral == "boundary"
        else ufl.inner(trial_function, test_function) * ufl.dx(1)
    )

    with pytest.raises(ValueError, match="only supports cell integrals over the whole domain"):
        fem.FEMMatrixFactorizationAssembler(mesh, function_space, form)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("coefficient_type", ["constant", "function"])
def test_factorization_rejects_forms_with_coefficients(coefficient_type: str) -> None:
    """Coefficients and constants used to be read from null pointers, crashing the process."""
    mesh = helpers.create_unit_interval_mesh(helpers.MESH_COMMUNICATOR)
    function_space = dlx.fem.functionspace(mesh, ("Lagrange", 1))
    coefficient = (
        dlx.fem.Constant(mesh, 2.0)
        if coefficient_type == "constant"
        else dlx.fem.Function(function_space)
    )
    form = (
        coefficient
        * ufl.inner(ufl.TrialFunction(function_space), ufl.TestFunction(function_space))
        * ufl.dx
    )

    with pytest.raises(ValueError, match="does not support forms with coefficients or constants"):
        fem.FEMMatrixFactorizationAssembler(mesh, function_space, form)


# --------------------------------------------------------------------------------------------------
def test_factorization_rejects_cell_matrices_that_are_not_positive_definite() -> None:
    """The stiffness form annihilates constants on each cell, its cell matrices are singular."""
    mesh = helpers.create_unit_interval_mesh(helpers.MESH_COMMUNICATOR)
    function_space = dlx.fem.functionspace(mesh, ("Lagrange", 1))
    trial_function = ufl.TrialFunction(function_space)
    test_function = ufl.TestFunction(function_space)
    stiffness_form = ufl.inner(ufl.grad(trial_function), ufl.grad(test_function)) * ufl.dx
    assembler = fem.FEMMatrixFactorizationAssembler(mesh, function_space, stiffness_form)

    with pytest.raises(ValueError, match="symmetric positive definite on each cell"):
        assembler.assemble()
