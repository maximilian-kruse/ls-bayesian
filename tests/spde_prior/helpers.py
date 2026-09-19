r"""Constants, setup objects and pure helper functions for the `spde_prior` tests.

This is a regular module, imported by test modules and `conftest.py` files alike. Fixtures live in
the `conftest.py` files, everything that is imported by name lives here.

The serial test setups are small FEM problems on the unit interval and the unit square with
Lagrange elements of degree one and two. Their expected values are derived from closed-form
integrals, mathematical properties of the prior operators, or dense numpy computations that are
independent of the code under test.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dolfinx as dlx
import nbformat
import numpy as np
import ufl
from dolfinx.fem import petsc
from mpi4py import MPI
from nbclient import NotebookClient
from petsc4py import PETSc

from ls_bayesian.spde_prior import builder, components, fem, spde_prior, strategies

# ==================================================================================================
# Dolfinx meshes require an MPI communicator, all serial test objects live on a single process.
MESH_COMMUNICATOR = MPI.COMM_SELF
# SPDE parameters. They are chosen such that all plausible mis-scalings of the SPDE operator,
# kappa^2 tau = 4.5, kappa tau = 1.5, kappa = 3, kappa^2 = 9, tau^2 = 0.25 and
# kappa^4 tau^2 = 20.25, differ from each other and from one.
KAPPA = 3.0
TAU = 0.5
ROBIN_CONSTANT = 0.3
# Upper bound for the condition numbers of the assembled mass and SPDE matrices, of their
# preconditioners, and of the P1 precision matrices in the test setups. Measured values are at most
# 30, 6 and 75, respectively. The bound is asserted in the fixtures, so that the tolerance
# derivations below cannot become stale silently.
MAX_CONDITION_NUMBER = 1e2
# Largest matrix dimension in the serial setups (DoFs of the 2D P2 case).
MAX_PROBLEM_DIMENSION = 25
# Normwise relative error of direct dense and sparse computations, bounded by
# cond * n * eps <= 1e2 * 25 * 2.2e-16 ~ 5.5e-13.
ROUNDOFF_TOLERANCE = 1e-12
# Relative residual reduction of the Krylov solves in the tests, identical to the builder default.
SOLVER_RELATIVE_TOLERANCE = builder.DEFAULT_SOLVER_RELATIVE_TOLERANCE
# Maximum number of Krylov solves involved in a tested relation (C P = I uses three).
MAX_NUM_KRYLOV_SOLVES = 3
# Normwise relative error of relations involving Krylov solves. PETSc's CG stops on the
# preconditioned residual, so the relative error of a single solve is bounded by
# cond(A) cond(B) * rtol. It is amplified by at most the condition number of the operator applied
# afterwards, giving 3 * (1e2)^3 * 1e-12 = 3e-6 as a worst-case first-order bound. Measured errors
# are of the order 1e-14.
KRYLOV_TOLERANCE = MAX_NUM_KRYLOV_SOLVES * MAX_CONDITION_NUMBER**3 * SOLVER_RELATIVE_TOLERANCE
# Statistical tests compare z-scores against this threshold. With fixed seeds,
# the tests are deterministic.
STATISTICAL_Z_THRESHOLD = 5.0
NUM_SAMPLES = 4000
# Seed for mean vector generation
MEAN_VECTOR_SEED = 0


# ==================================================================================================
@dataclass(frozen=True)
class FEMCase:
    r"""Serial FEM test case, a mesh and Lagrange degree with known integrals.

    Attributes:
        case_id: Name of the case, used as pytest id.
        create_mesh: Mesh factory, taking the MPI communicator.
        degree: Degree of the Lagrange function space.
        domain_measure: Volume $|\Omega|$ of the domain.
        boundary_measure: Measure $|\partial\Omega|$ of the boundary (counting measure in 1D).
        x0_boundary_integral: Boundary integral $\int_{\partial\Omega} x_0^2 ds$.
        num_vertices: Number of mesh vertices.
        num_dofs: Number of DoFs of the function space.
    """

    case_id: str
    create_mesh: Callable[[MPI.Comm], dlx.mesh.Mesh]
    degree: int
    domain_measure: float
    boundary_measure: float
    x0_boundary_integral: float
    num_vertices: int
    num_dofs: int


def create_unit_interval_mesh(communicator: MPI.Comm) -> dlx.mesh.Mesh:
    return dlx.mesh.create_unit_interval(communicator, nx=3)


def create_unit_square_mesh(communicator: MPI.Comm) -> dlx.mesh.Mesh:
    return dlx.mesh.create_unit_square(
        communicator, nx=2, ny=2, cell_type=dlx.mesh.CellType.triangle
    )

# Note: x_0 is the first spatial coordinate domain of the mesh
# On both domains, int_Omega x_0^2 dx = 1/3 and int_Omega |grad x_0|^2 dx = 1. On the unit square,
# int_{dOmega} x_0^2 ds = 0 (x_0 = 0) + 1 (x_0 = 1) + 2 * 1/3 (y = 0 and y = 1) = 5/3.
X0_SQUARED_DOMAIN_INTEGRAL = 1 / 3
X0_GRADIENT_SQUARED_DOMAIN_INTEGRAL = 1.0
FEM_CASES = {
    fem_case.case_id: fem_case
    for fem_case in (
        FEMCase("1d_p1", create_unit_interval_mesh, 1, 1.0, 2.0, 1.0, 4, 4),
        FEMCase("1d_p2", create_unit_interval_mesh, 2, 1.0, 2.0, 1.0, 4, 7),
        FEMCase("2d_p1", create_unit_square_mesh, 1, 1.0, 4.0, 5 / 3, 9, 9),
        FEMCase("2d_p2", create_unit_square_mesh, 2, 1.0, 4.0, 5 / 3, 9, 25),
    )
}
FEM_CASE_IDS = list(FEM_CASES)
# For P1 spaces, the conversion between vertex and DoF vectors is a permutation, so that operators
# in vertex space are similar to their DoF-space counterparts.
P1_CASE_IDS = [case_id for case_id, fem_case in FEM_CASES.items() if fem_case.degree == 1]
# For higher-degree spaces, the vertex-to-DoF interpolation is a genuine, non-square embedding, so
# its adjoint (used for gradient pullback) differs from the plain DoF-to-vertex conversion.
P2_CASE_IDS = [case_id for case_id, fem_case in FEM_CASES.items() if fem_case.degree == 2]


@dataclass
class FEMSpaceSetup:
    fem_case: FEMCase
    mesh: dlx.mesh.Mesh
    function_space: dlx.fem.FunctionSpace


@dataclass
class AssembledMatrices:
    """Dense, read-only mass and SPDE matrices, assembled with KAPPA and TAU."""

    mass_matrix: np.ndarray
    neumann_spde_matrix: np.ndarray
    robin_spde_matrix: np.ndarray


def create_fem_space_setup(fem_case: FEMCase, communicator: MPI.Comm) -> FEMSpaceSetup:
    mesh = fem_case.create_mesh(communicator)
    function_space = dlx.fem.functionspace(mesh, ("Lagrange", fem_case.degree))
    return FEMSpaceSetup(fem_case, mesh, function_space)


# ==================================================================================================
@dataclass
class ExactPriorSetup:
    r"""SPDE prior on exact, dense operators in DoF space.

    The precision $P$ is a random SPD matrix, the covariance is $C = P^{-1}$, and the covariance
    factorization $F = L Q^T$ combines the Cholesky factor $C = L L^T$ with a matrix $Q$ with
    orthonormal columns, so that $F$ is rectangular with $F F^T = C$.
    """

    fem_space_setup: FEMSpaceSetup
    converter: fem.FEMConverter
    mean_vector: np.ndarray
    precision_matrix: np.ndarray
    covariance_matrix: np.ndarray
    covariance_factorization_matrix: np.ndarray


# Additional columns of the rectangular covariance factorization in the exact prior setup.
NUM_EXTRA_FACTORIZATION_COLUMNS = 2


def create_exact_prior_setup(fem_space_setup: FEMSpaceSetup, seed: int) -> ExactPriorSetup:
    rng = np.random.default_rng(seed)
    num_dofs = fem_space_setup.fem_case.num_dofs
    precision_matrix = random_spd_matrix(rng, num_dofs)
    covariance_matrix = np.linalg.inv(precision_matrix)
    orthonormal_columns, _ = np.linalg.qr(
        rng.standard_normal((num_dofs + NUM_EXTRA_FACTORIZATION_COLUMNS, num_dofs))
    )
    return ExactPriorSetup(
        fem_space_setup=fem_space_setup,
        converter=fem.FEMConverter(fem_space_setup.function_space),
        mean_vector=rng.random(fem_space_setup.fem_case.num_vertices),
        precision_matrix=precision_matrix,
        covariance_matrix=covariance_matrix,
        covariance_factorization_matrix=np.linalg.cholesky(covariance_matrix)
        @ orthonormal_columns.T,
    )


def create_exact_spde_prior(
    setup: ExactPriorSetup,
    seed: int,
    precision_matrix: np.ndarray | None = None,
    covariance_matrix: np.ndarray | None = None,
    covariance_factorization_matrix: np.ndarray | None = None,
    mean_vector: np.ndarray | None = None,
) -> spde_prior.SPDEPrior:
    """Create an `SPDEPrior` from the dense operators of the setup, optionally replacing some."""
    precision_matrix = setup.precision_matrix if precision_matrix is None else precision_matrix
    covariance_matrix = setup.covariance_matrix if covariance_matrix is None else covariance_matrix
    if covariance_factorization_matrix is None:
        covariance_factorization_matrix = setup.covariance_factorization_matrix
    mean_vector = setup.mean_vector if mean_vector is None else mean_vector
    return spde_prior.SPDEPrior(
        mean_vector,
        dense_interface_component(precision_matrix),
        dense_interface_component(covariance_matrix),
        dense_interface_component(covariance_factorization_matrix),
        setup.converter,
        seed=seed,
    )


# ==================================================================================================
@dataclass
class BuiltPriorSetup:
    """Bilaplacian prior assembled by the builder, together with its settings."""

    fem_space_setup: FEMSpaceSetup
    robin_const: float | None
    mean_vector: np.ndarray
    prior: spde_prior.SPDEPrior


def build_bilaplacian_prior(
    fem_space_setup: FEMSpaceSetup,
    robin_const: float | None,
    seed: int = 0,
    **settings_overrides: object,
) -> BuiltPriorSetup:
    """Build a prior with KAPPA, TAU and explicit solver tolerances, settings can be overridden."""
    fem_case = fem_space_setup.fem_case
    mean_vector = np.random.default_rng(MEAN_VECTOR_SEED).random(fem_case.num_vertices)
    settings_arguments = {
        "mesh": fem_space_setup.mesh,
        "mean_vector": mean_vector,
        "kappa": KAPPA,
        "tau": TAU,
        "robin_const": robin_const,
        "seed": seed,
        "fe_data": ("Lagrange", fem_case.degree),
        "cg_relative_tolerance": SOLVER_RELATIVE_TOLERANCE,
        "amg_relative_tolerance": SOLVER_RELATIVE_TOLERANCE,
    }
    settings_arguments.update(settings_overrides)
    settings = builder.SPDEPriorSettings(**settings_arguments)
    prior = builder.SPDEPriorBuilder(
        settings, strategies.BilaplacianComponentStrategy()
    ).build()
    return BuiltPriorSetup(fem_space_setup, robin_const, mean_vector, prior)


# ==================================================================================================
def assert_allclose_normwise(
    actual: np.ndarray | float,
    desired: np.ndarray | float,
    tolerance: float,
    scale: float | None = None,
) -> None:
    r"""Assert that $\|\text{actual} - \text{desired}\| \leq \text{tolerance} \cdot \text{scale}$.

    Normwise error bounds, as for roundoff and Krylov errors, do not bound the relative error of
    individual entries. The entrywise absolute tolerance `tolerance * scale` is implied by the
    normwise bound, with `scale` defaulting to the 2-norm (Frobenius norm for matrices) of
    `desired`.
    """
    if scale is None:
        scale = float(np.linalg.norm(desired))
    np.testing.assert_allclose(actual, desired, rtol=0, atol=tolerance * scale)


def assert_condition_number_bounded(matrix: np.ndarray) -> None:
    """Assert the premise of the tolerance derivations, see `MAX_CONDITION_NUMBER`."""
    condition_number = np.linalg.cond(matrix)
    assert condition_number <= MAX_CONDITION_NUMBER, (
        f"Condition number {condition_number:.1e} exceeds {MAX_CONDITION_NUMBER:.1e}, "
        "the tolerance derivations are no longer valid."
    )


# ==================================================================================================
def random_spd_matrix(rng: np.random.Generator, dim: int) -> np.ndarray:
    factor = rng.random((dim, dim))
    return factor @ factor.T + dim * np.eye(dim)


def petsc_matrix_from_dense(array: np.ndarray) -> PETSc.Mat:
    num_rows, num_cols = array.shape
    matrix = PETSc.Mat().createAIJ((num_rows, num_cols), comm=PETSc.COMM_SELF)
    matrix.setUp()
    matrix.setValues(
        np.arange(num_rows, dtype=PETSc.IntType), np.arange(num_cols, dtype=PETSc.IntType), array
    )
    matrix.assemble()
    return matrix


def petsc_vector_from_array(array: np.ndarray) -> PETSc.Vec:
    return PETSc.Vec().createWithArray(array.copy(), comm=PETSc.COMM_SELF)


def dense_from_petsc_matrix(matrix: PETSc.Mat) -> np.ndarray:
    num_rows, num_cols = matrix.getSize()
    return matrix.getValues(
        np.arange(num_rows, dtype=PETSc.IntType), np.arange(num_cols, dtype=PETSc.IntType)
    )


def dense_interface_component(array: np.ndarray) -> components.InterfaceComponent:
    return components.InterfaceComponent(components.Matrix(petsc_matrix_from_dense(array)))


def assemble_dense_matrix(form: ufl.Form) -> np.ndarray:
    """Assemble a bilinear form with dolfinx and return it as dense array."""
    matrix = petsc.assemble_matrix(dlx.fem.form(form))
    matrix.assemble()
    return dense_from_petsc_matrix(matrix)


def materialize_operator(
    apply_operator: Callable[[np.ndarray], np.ndarray], input_dim: int
) -> np.ndarray:
    """Build the dense matrix of a linear operator column by column from unit vectors."""
    return np.column_stack([apply_operator(unit_vector) for unit_vector in np.eye(input_dim)])


# ==================================================================================================
def input_ordered_vertex_coordinates(mesh: dlx.mesh.Mesh) -> np.ndarray:
    """Vertex coordinates of a serial, affine mesh in input node order, shape `(n_vertices, 3)`."""
    return mesh.geometry.x[np.argsort(mesh.geometry.input_global_indices)]


def linear_field(coordinates: np.ndarray) -> np.ndarray:
    """Linear field $f(x) = 1 + 2 x_0 - x_1 / 2$, reproduced exactly by Lagrange interpolation."""
    return 1.0 + 2.0 * coordinates[:, 0] - 0.5 * coordinates[:, 1]


def vertex_to_dof_selection_matrix(function_space: dlx.fem.FunctionSpace) -> np.ndarray:
    r"""Permutation matrix $S$ with $S v$ the DoF vector of vertex values $v$ on a P1 space.

    The permutation is determined by matching DoF and vertex coordinates, independently of the
    [`FEMConverter`][ls_bayesian.spde_prior.fem.FEMConverter]. A vertex-space operator is then
    $S^T A S$ for a DoF-space operator $A$.
    """
    dof_coordinates = function_space.tabulate_dof_coordinates()
    vertex_coordinates = input_ordered_vertex_coordinates(function_space.mesh)
    distances = np.linalg.norm(dof_coordinates[:, None, :] - vertex_coordinates[None, :, :], axis=2)
    vertex_indices = np.argmin(distances, axis=1)
    # Coordinates lie in the unit cube, so a match holds up to absolute roundoff.
    np.testing.assert_allclose(
        distances[np.arange(vertex_indices.size), vertex_indices],
        0.0,
        atol=ROUNDOFF_TOLERANCE,
    )
    return np.eye(vertex_coordinates.shape[0])[vertex_indices]


# ==================================================================================================
def whitened_sample_z_scores(
    samples: np.ndarray, mean_vector: np.ndarray, covariance_matrix: np.ndarray
) -> tuple[float, float]:
    r"""Z-scores of mean and variance of whitened Gaussian samples.

    Samples $s_i \sim \mathcal{N}(\bar{m}, C)$ are whitened as $z_i = L^{-1}(s_i - \bar{m})$ with
    $C = L L^T$, so that all $N n$ entries are i.i.d. standard normal. Their mean has standard
    deviation $1/\sqrt{Nn}$, and their (population) variance has approximately standard deviation
    $\sqrt{2/(Nn)}$.

    Args:
        samples: Samples, shape `(num_samples, dim)`.
        mean_vector: Mean $\bar{m}$, shape `(dim,)`.
        covariance_matrix: Covariance $C$, shape `(dim, dim)`.

    Returns:
        tuple[float, float]: Z-scores of the sample mean and sample variance.
    """
    cholesky_factor = np.linalg.cholesky(covariance_matrix)
    whitened_samples = np.linalg.solve(cholesky_factor, (samples - mean_vector).T)
    num_values = whitened_samples.size
    mean_z_score = whitened_samples.mean() * np.sqrt(num_values)
    variance_z_score = (np.mean(whitened_samples**2) - 1.0) * np.sqrt(num_values / 2)
    return float(mean_z_score), float(variance_z_score)


# ==================================================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
PRIOR_TUTORIALS_DIR = REPO_ROOT / "tutorials" / "prior"
BUILDER_NOTEBOOK = PRIOR_TUTORIALS_DIR / "builder.ipynb"
COMPONENTS_NOTEBOOK = PRIOR_TUTORIALS_DIR / "components.ipynb"
# The notebooks assemble FEM matrices, run Krylov solves, and render PyVista/matplotlib figures,
# which can take tens of seconds on CI hardware.
NOTEBOOK_EXECUTION_TIMEOUT_SECONDS = 600


def execute_notebook_and_extract_values(
    notebook_path: Path, expressions: Mapping[str, str]
) -> dict[str, Any]:
    """Execute a tutorial notebook and evaluate expressions against its final namespace.

    The notebook is executed unmodified in its own kernel, except for one appended code cell that
    evaluates the given expressions and serializes the results to stdout as JSON. Expressions
    reduce large results (e.g. vertex vectors) to compact scalars, such as a norm, so that
    reference values stay small numeric literals in test code rather than stored array data.

    Args:
        notebook_path (Path): Path to the `.ipynb` file to execute.
        expressions (Mapping[str, str]): Mapping from a result key to a Python expression,
            evaluated in the notebook's namespace after all of its own cells have run. Expression
            results must be JSON-serializable.

    Returns:
        dict[str, Any]: Mapping from result key to its JSON-deserialized value.
    """
    notebook = nbformat.read(notebook_path, as_version=4)
    probe_source = (
        "import json as _json\n"
        f"_probe_values = {{key: eval(expr) for key, expr in {dict(expressions)!r}.items()}}\n"
        "print(_json.dumps(_probe_values))"
    )
    notebook.cells.append(nbformat.v4.new_code_cell(source=probe_source))

    client = NotebookClient(
        notebook,
        timeout=NOTEBOOK_EXECUTION_TIMEOUT_SECONDS,
        kernel_name=notebook.metadata["kernelspec"]["name"],
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()

    probe_outputs = notebook.cells[-1]["outputs"]
    stdout_text = "".join(
        output["text"] for output in probe_outputs if output.get("name") == "stdout"
    )
    return json.loads(stdout_text)
