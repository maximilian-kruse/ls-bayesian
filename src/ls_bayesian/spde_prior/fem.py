"""This module contains all FEM specific functionality in this package, based on dolfinx.

All functionality supports meshes distributed over several processes. Vertex-based vectors are
replicated on all processes and ordered by the input node indices of the mesh, so that they do not
depend on the number of processes. DoF-based vectors are distributed, i.e. every process holds the
values of the DoFs it owns, in the layout of the corresponding dolfinx and PETSc vectors.

Classes:
    FEMConverter: Converter between vertex based data and DoF representation on a dolfinx
        function space.
    FEMMatrixFactorizationAssembler: Assembler for the rectangular factorization of an FEM matrix.

Functions:
    generate_forms: Generate variational forms for the mass matrix and SPDE system matrix.
    order_vertices_and_cells: Get vertex coordinates and cell connectivity of a mesh, ordered to
        match the vertex vectors handled elsewhere in this module. This corresponds to the ordering
        of, e.g. meshes loaded via pyvista.
"""

from numbers import Real
from typing import Annotated

import cffi
import dolfinx as dlx
import numpy as np
import scifem
import ufl
from beartype.vale import Is
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc

ffi = cffi.FFI()


# ==================================================================================================
def generate_forms(
    function_space: dlx.fem.FunctionSpace,
    kappa: Annotated[Real, Is[lambda x: x > 0]],
    tau: Annotated[Real, Is[lambda x: x > 0]],
    robin_const: Annotated[Real, Is[lambda x: x >= 0]] | None = None,
) -> tuple[ufl.Form, ufl.Form]:
    r"""Construct dolfinx forms for the mass matrix and SPDE system matrix.

    This method constructs dolfinx variational forms that resemble the mass matrix contribution
    and SPDE system matrix, i.e. the left-hand-side of the SPDE generating the random field.
    More specifically, let $\Omega$ be the domain of interest, and $\phi$ both the trial and test
    function (defined over the same function space).
    The mass matrix contribution is then given as $(\phi, \phi)_{L^2(\Omega)}$, and the total SPDE
    matrix contribution is

    $$
    \begin{equation*}
        \kappa^2 \tau (\phi, \phi)_{L^2(\Omega)} + \tau (\nabla \phi, \nabla \phi)_{L^2(\Omega)}
        + \beta (\phi, \phi)_{L^2(\partial\Omega)}
    \end{equation*}
    $$

    $\kappa$ and $\tau$ correspond to the parameters `kappa` and `tau` for the parameterization of
    the field.
    $\beta$ is the optional `robin_const` parameter that enforces Robin boundary conditions
    instead of homogeneous Neumann boundary conditions. As the boundary term is not scaled with
    $\tau$, the boundary condition of the operator $\tau(\kappa^2 - \Delta)$ reads
    $\tau \nabla \phi \cdot n + \beta \phi = 0$. For $\kappa, \tau > 0$ and $\beta \geq 0$, the
    SPDE matrix is symmetric positive definite, a negative $\beta$ can render it indefinite.

    Args:
        function_space (dlx.fem.FunctionSpace): dolfinx function space to construct forms over.
        kappa (Real): Parameter $\kappa > 0$ of the prior field.
        tau (Real): Parameter $\tau > 0$ of the prior field.
        robin_const (Real | None, optional): Parameter $\beta \geq 0$ of the prior field
            enforcing Robin boundary conditions. Defaults to None.

    Returns:
        tuple[ufl.Form, ufl.Form]: dolfinx forms for the mass matrix and SPDE system matrix
            contributions.
    """
    trial_function = ufl.TrialFunction(function_space)
    test_function = ufl.TestFunction(function_space)
    mass_matrix_form = ufl.inner(trial_function, test_function) * ufl.dx
    stiffness_matrix_form = ufl.inner(ufl.grad(trial_function), ufl.grad(test_function)) * ufl.dx
    spde_matrix_form = kappa**2 * tau * mass_matrix_form + tau * stiffness_matrix_form
    if robin_const is not None:
        robin_boundary_form = robin_const * ufl.inner(trial_function, test_function) * ufl.ds
        spde_matrix_form += robin_boundary_form
    return mass_matrix_form, spde_matrix_form


# --------------------------------------------------------------------------------------------------
def get_ordered_vertices_and_cells(
    mesh: dlx.mesh.Mesh,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    np.ndarray[tuple[int, int], np.dtype[np.int64]],
]:
    """Get vertex coordinates and cell connectivity of a mesh, in the input node order.

    Vertex vectors handled elsewhere in this module (see the module docstring, and
    [`FEMConverter`][ls_bayesian.spde_prior.fem.FEMConverter]) are ordered by input node index, not
    by dolfinx's own internal geometry node order: dolfinx is free to renumber geometry nodes
    internally, e.g. for cache locality, even on a single process, so `mesh.geometry.x` and
    `mesh.geometry.dofmaps[0]` correspond, in general, not to the input node order.
    Plotting or exporting a vertex vector directly against `mesh.geometry.x` and
    `mesh.geometry.dofmaps[0]` therefore assigns it to the wrong points. This function returns the
    same geometry, consistently reindexed by input node order, so that a vertex vector can be
    combined with it directly.

    Args:
        mesh (dlx.mesh.Mesh): dolfinx mesh with affine (degree-1) geometry.

    Raises:
        ValueError: If the mesh geometry is not affine, as vertices are then not the only
            geometry nodes.

    Returns:
        tuple[np.ndarray[tuple[int, int], np.dtype[np.float64]],
              np.ndarray[tuple[int, int], np.dtype[np.int64]]]: Vertex coordinates, shape
            `(vertex_space_dim, geometric_dim)`, and cell connectivity indexing into the vertex
            coordinates, shape `(num_cells, num_cell_vertices)`. Both are replicated identically
            on every process, and cells are ordered by their input index.
    """
    if mesh.geometry.cmaps[0].degree != 1:
        raise ValueError(
            "get_ordered_vertices_and_cells requires an affine mesh geometry, but the geometry has "
            f"degree {mesh.geometry.cmaps[0].degree}."
        )
    communicator = mesh.comm

    # Get Vertices
    vertex_index_map = mesh.topology.index_map(0)
    num_vertices_global = vertex_index_map.size_global
    num_owned_vertices = vertex_index_map.size_local
    mesh.topology.create_connectivity(0, mesh.topology.dim)
    owned_vertex_geometry_nodes = dlx.mesh.entities_to_geometry(
        mesh, 0, np.arange(num_owned_vertices, dtype=np.int32)
    ).reshape(-1)
    owned_vertex_input_indices = mesh.geometry.input_global_indices[owned_vertex_geometry_nodes]
    owned_vertex_coordinates = mesh.geometry.x[owned_vertex_geometry_nodes]

    geometric_dim = owned_vertex_coordinates.shape[1]
    vertex_coordinates = np.column_stack(
        [
            _gather_by_index(
                communicator,
                owned_vertex_input_indices,
                owned_vertex_coordinates[:, component],
                num_vertices_global,
            )
            for component in range(geometric_dim)
        ]
    )

    # Get connectivity
    cell_index_map = mesh.topology.index_map(mesh.topology.dim)
    num_owned_cells = cell_index_map.size_local
    num_cells_global = cell_index_map.size_global
    owned_cell_input_indices = mesh.topology.original_cell_index[:num_owned_cells]
    # `mesh.geometry.dofmaps[0]` indexes local geometry rows, which `input_global_indices` maps to
    # input node indices; these coincide with positions in `vertex_coordinates` by construction.
    owned_cell_connectivity = mesh.geometry.input_global_indices[
        mesh.geometry.dofmaps[0][:num_owned_cells]
    ].astype(np.float64)
    num_cell_vertices = owned_cell_connectivity.shape[1]
    cell_connectivity = np.column_stack(
        [
            _gather_by_index(
                communicator,
                owned_cell_input_indices,
                owned_cell_connectivity[:, local_vertex],
                num_cells_global,
            )
            for local_vertex in range(num_cell_vertices)
        ]
    ).astype(np.int64)

    return vertex_coordinates, cell_connectivity


# --------------------------------------------------------------------------------------------------
def _check_factorizable_form(form: ufl.Form) -> None:
    """Check that a form only consists of what the cell-wise factorization assembly supports.

    The assembly calls the compiled kernel of the first cell integral of the form directly, and
    passes neither coefficient nor constant values to it. Other integrals would be silently
    dropped, and coefficients or constants would be read from null pointers.
    """
    unsupported_integrals = [
        (integral.integral_type(), integral.subdomain_id())
        for integral in form.integrals()
        if (integral.integral_type(), integral.subdomain_id()) != ("cell", "everywhere")
    ]
    if unsupported_integrals:
        raise ValueError(
            "Matrix factorization assembly only supports cell integrals over the whole domain, "
            f"but the form contains the integrals (type, subdomain) {unsupported_integrals}."
        )
    if form.coefficients() or form.constants():
        raise ValueError(
            "Matrix factorization assembly does not support forms with coefficients or constants, "
            f"but the form contains {len(form.coefficients())} coefficients and "
            f"{len(form.constants())} constants."
        )


# --------------------------------------------------------------------------------------------------
def _gather_by_index(
    communicator: MPI.Comm,
    indices: np.ndarray[tuple[int], np.dtype[np.integer]],
    values: np.ndarray[tuple[int], np.dtype[np.float64]],
    size: int,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Assemble a complete vector on all processes from disjoint, indexed parts of each process."""
    counts = communicator.allgather(values.size)
    assert sum(counts) == size, f"Gathered {sum(counts)} entries, but expected {size}."
    all_indices = np.empty(size, dtype=np.int64)
    all_values = np.empty(size, dtype=np.float64)
    communicator.Allgatherv(indices.astype(np.int64), (all_indices, counts))
    communicator.Allgatherv(np.ascontiguousarray(values, dtype=np.float64), (all_values, counts))
    complete_vector = np.empty(size, dtype=np.float64)
    complete_vector[all_indices] = all_values
    return complete_vector


# ==================================================================================================
class FEMConverter:
    r"""Converter between vertex based data and DoF representation on a dolfinx function space.

    This class connects the representation of arrays in dolfinx on a specified function space
    with a vertex-based viewpoint. Think of it as the adapter required for the prior to
    communicate with external components. The underlying idea is that such outside components
    only see the computational mesh, and define discrete data over the vertices of that mesh.
    An `FEMConverter` object takes such data structures and interpolates them to the provided
    function space. On the other hand, it can interpolate any data defined on the DoFs of the
    underlying function space onto the vertices of the mesh.
    Internally, the `FEMConverter` assigns vertex based input data to the DoFs of a P1
    function space (whose degrees of freedom are exactly the vertices).
    It subsequently utilizes dolfinx's efficient interpolation between function spaces.
    On the other hand, data from some function space is interpolated to a P1 space, and
    subsequently extracted to vertex values.

    Vertex vectors are complete on every process and ordered by the input node indices of the mesh,
    i.e. the node order of the mesh as it was created or read. DoF vectors are distributed, every
    process holds the values of its owned DoFs. The converter further provides the ordering of
    vectors indexed by cell-local DoFs, the input space of cell-wise block matrices: entry
    `c * num_cell_dofs + k` belongs to local DoF `k` of the cell with input index `c`.

    Denote the vertex-to-DoF interpolation `convert_vertex_values_to_dofs` performs by the linear
    map $I$. For a P1 function space, $I$ is a permutation of the vertex values, and its inverse
    coincides with its transpose $I^T$, which in turn coincides with `convert_dofs_to_vertex_values`
    (denoted $R$): $R = I^{-1} = I^T$. For a function space of higher degree, $I$ is an injective,
    non-square map into the larger DoF space (e.g. it also determines edge-midpoint DoFs from
    vertex values), so $R \neq I^T$ in general; $R$ still recovers a DoF-space *field's own value*
    at the vertices (used, e.g., to express a sample or the covariance/precision operator as a
    field-to-field map on vertex space), but it is not the adjoint of $I$. Differentiating a
    functional of $I(m)$ with respect to $m$ requires that adjoint, $\nabla_m = I^T \nabla_u$,
    which `pull_back_gradient` provides. Use `pull_back_gradient`, not
    `convert_dofs_to_vertex_values`, wherever a DoF-space gradient or Hessian-vector product needs
    to be expressed in vertex space.

    Methods:
        convert_vertex_values_to_dofs: Convert vertex based data to DoF representation
        convert_dofs_to_vertex_values: Convert DoF based data to vertex representation
        pull_back_gradient: Pull back a gradient (covector) from DoF space to vertex space.

    Attributes:
        vertex_space_dim (int): Number of mesh vertices, length of vertex vectors.
        dof_space_dim (int): Number of DoFs owned by this process, length of DoF vectors.
        global_dof_space_dim (int): Number of DoFs over all processes.
        cell_block_input_indices (np.ndarray): Indices of the entries of a complete cell-local
            DoF vector that belong to the cells owned by this process.
        comm (MPI.Comm): Communicator of the mesh.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, function_space: dlx.fem.FunctionSpace) -> None:
        """Initialize the converter for a given dolfinx function space.

        The function space implicitly carries the mesh, and thus the vertices we want to convert
        from/to.

        Args:
            function_space (dlx.fem.FunctionSpace): Scalar function space in which degrees of
                freedom lie, on a mesh with affine geometry.

        Raises:
            ValueError: If the mesh geometry is not affine, as vertices are then not the only
                geometry nodes.
            ValueError: If the function space is not scalar-valued.
        """
        mesh = function_space.mesh
        if mesh.geometry.cmaps[0].degree != 1:
            raise ValueError(
                f"FEMConverter requires an affine mesh geometry, but the geometry has degree "
                f"{mesh.geometry.cmaps[0].degree}."
            )
        if function_space.dofmap.index_map_bs != 1:
            raise ValueError(
                "FEMConverter requires a scalar function space, but the DoF block size is "
                f"{function_space.dofmap.index_map_bs}."
            )
        vertex_space = dlx.fem.functionspace(mesh, ("Lagrange", 1))
        vertex_index_map = mesh.topology.index_map(0)
        dof_index_map = function_space.dofmap.index_map
        cell_index_map = mesh.topology.index_map(mesh.topology.dim)
        num_cell_dofs = function_space.dofmap.dof_layout.num_dofs

        self.comm = mesh.comm
        self.global_vertex_space_dim = vertex_index_map.size_global
        self.local_dof_space_dim = dof_index_map.size_local
        self.global_dof_space_dim = dof_index_map.size_global
        self._dof_function = dlx.fem.Function(function_space)
        self._vertex_function = dlx.fem.Function(vertex_space)

        self._num_owned_vertices = vertex_index_map.size_local
        self._p1_vertex_to_dof_map = scifem.vertex_to_dofmap(vertex_space)
        local_vertices = np.arange(
            vertex_index_map.size_local + vertex_index_map.num_ghosts, dtype=np.int32
        )
        mesh.topology.create_connectivity(0, mesh.topology.dim)
        vertex_geometry_nodes = dlx.mesh.entities_to_geometry(mesh, 0, local_vertices).reshape(-1)
        self._global_vertex_indices = mesh.geometry.input_global_indices[vertex_geometry_nodes]

        owned_cell_indices = mesh.topology.original_cell_index[: cell_index_map.size_local]
        self.cell_block_indices = (
            owned_cell_indices[:, None] * num_cell_dofs + np.arange(num_cell_dofs)
        ).reshape(-1)

        # Adjoint $I^T$ of the vertex-to-DoF interpolation $I$, for `pull_back_gradient`. Built as
        # an explicit transpose of dolfinx's own interpolation matrix (verified to reproduce
        # `convert_vertex_values_to_dofs` exactly), so that applying it is an ordinary assembled
        # matrix-vector product, with no manual handling of ghost contributions required.
        vertex_to_dof_matrix = petsc.interpolation_matrix(vertex_space, function_space)
        vertex_to_dof_matrix.assemble()
        self._dof_to_vertex_adjoint_matrix = vertex_to_dof_matrix.transpose()
        self._adjoint_input_vector = self._dof_to_vertex_adjoint_matrix.createVecRight()
        self._adjoint_output_vector = self._dof_to_vertex_adjoint_matrix.createVecLeft()

    # ----------------------------------------------------------------------------------------------
    def convert_vertex_values_to_dofs(
        self, global_vertex_values: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Convert vertex based data to DoF representation.

        Args:
            vertex_values (np.ndarray[tuple[int], np.dtype[np.float64]]): Array of data defined on
                the vertices of the underlying computational mesh, in input node order, shape
                `(vertex_space_dim,)`.

        Raises:
            ValueError: Checks that `vertex_values` has one entry per mesh vertex.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Input data interpolated to the owned
                function space DoFs, shape `(dof_space_dim,)`, copied.
        """
        if not global_vertex_values.shape == (self.global_vertex_space_dim,):
            raise ValueError(
                f"Expected vertex_values to have shape {(self.global_vertex_space_dim,)}, "
                f"but got {global_vertex_values.shape}"
            )
        # All local vertices, including ghosts, are set from the complete vector, so no ghost
        # update is required before interpolation.
        self._vertex_function.x.array[self._p1_vertex_to_dof_map] = global_vertex_values[
            self._global_vertex_indices
        ]
        self._dof_function.interpolate(self._vertex_function)
        return self._dof_function.x.array[: self.local_dof_space_dim].copy()

    # ----------------------------------------------------------------------------------------------
    def convert_dofs_to_vertex_values(
        self, local_dof_values: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Convert DoF based data to vertex representation.

        Args:
            dof_values (np.ndarray[tuple[int], np.dtype[np.float64]]): Array of data defined on
                the owned DoFs of the underlying function space, shape `(local_dof_space_dim,)`.

        Raises:
            ValueError: Checks that the shape of the input vector matches the number of owned DoFs
                on the underlying function space.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Array of data interpolated to the vertices
                of the underlying mesh, in input node order, shape `(global_vertex_space_dim,)`.
                The array is identical on all processes.
        """
        if not local_dof_values.shape == (self.local_dof_space_dim,):
            raise ValueError(
                f"Expected dof_values to have shape {(self.local_dof_space_dim,)}, "
                f"but got {local_dof_values.shape}"
            )
        self._dof_function.x.array[: self.local_dof_space_dim] = local_dof_values
        self._dof_function.x.scatter_forward()
        self._vertex_function.interpolate(self._dof_function)
        owned_vertex_values = self._vertex_function.x.array[
            self._p1_vertex_to_dof_map[: self._num_owned_vertices]
        ]
        return _gather_by_index(
            self.comm,
            self._global_vertex_indices[: self._num_owned_vertices],
            owned_vertex_values,
            self.global_vertex_space_dim,
        )

    # ----------------------------------------------------------------------------------------------
    def pull_back_gradient(
        self, local_dof_space_gradient: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Pull back a gradient (covector) from DoF space to vertex space.

        For a scalar functional $J$ of a DoF-space quantity $u = I(m)$, with $I$ the vertex-to-DoF
        interpolation of `convert_vertex_values_to_dofs`, the chain rule gives
        $\nabla_m J = I^T \nabla_u J$. This method applies the adjoint $I^T$, and is the correct
        way to express a DoF-space gradient or Hessian-vector product in vertex space.

        `convert_dofs_to_vertex_values` is not equivalent: it recovers a DoF-space field's own
        value at the vertices, coinciding with $I^T$ only when $I$ is square (a P1 function
        space), see the class docstring.

        Args:
            dof_space_gradient (np.ndarray[tuple[int], np.dtype[np.float64]]): Gradient defined on
                the owned DoFs of the underlying function space, shape `(dof_space_dim,)`.

        Raises:
            ValueError: Checks that the shape of the input vector matches the number of owned DoFs
                on the underlying function space.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Pulled-back gradient on the vertices of
                the underlying mesh, in input node order, shape `(vertex_space_dim,)`. The array is
                identical on all processes.
        """
        if not local_dof_space_gradient.shape == (self.local_dof_space_dim,):
            raise ValueError(
                f"Expected dof_space_gradient to have shape {(self.local_dof_space_dim,)}, "
                f"but got {local_dof_space_gradient.shape}"
            )
        self._adjoint_input_vector.setArray(local_dof_space_gradient)
        self._dof_to_vertex_adjoint_matrix.mult(
            self._adjoint_input_vector, self._adjoint_output_vector
        )
        owned_vertex_gradient = self._adjoint_output_vector.getArray()[
            self._p1_vertex_to_dof_map[: self._num_owned_vertices]
        ]
        return _gather_by_index(
            self.comm,
            self._global_vertex_indices[: self._num_owned_vertices],
            owned_vertex_gradient,
            self.global_vertex_space_dim,
        )


# ==================================================================================================
class FEMMatrixFactorizationAssembler:
    r"""Assembler for the rectangular factorization of an FEM matrix.

    This class provides the functionality for efficient factorization of a finite element matrix
    $M$, i.e. it implements the (sparse representation of an) assembly of a rectangular matrix
    $\widehat{M}$ s.th. $M = \widehat{M}\widehat{M}^T$. The factorization exploits the
    characteristics of standard finite element assembly procedures. This assembly is typically done
    locally for the contribution of each mesh cell. The contributions are then gathered into a
    global matrix, with overlap at indices of vertices that are shared between cells. Now let $N$
    be the number of degrees of freedom of the finite element space, $N_c$ the number of cells in
    the underlying mesh, and $N_e$ the number of degrees of freedom per cell. The matrix $M$ has
    size $(N, N)$. Importantly, it can be decomposed as $M=L^T M_e L$. Here, the matrix
    $M_e\in \mathbb{R}^{N_c N_e \times N_c N_e}$ is block-diagonal, containing the local FEM matrix
    contributions over all cells in each block. $L\in \mathbb{R}^{N_c N_e \times N}$ is a sparse
    matrix that maps the local cell DoFs in each block $M_e$ to their respective global DoFs. We
    can efficiently compute the Cholesky factor $\widehat{M}_e$ of $M_e$ by computing the
    Cholesky factorization of each individual block. In particular, $\widehat{M}_e$ is still sparse,
    as opposed to standard Cholesky factors of sparse matrices. $\widehat{M}_e$ and $L$ now
    represent a sparse factorization $\widehat{M} = L^T \widehat{M}_e$ as above.
    The two matrices shouldn't be multiplied directly to avoid fill-in effects. However,
    matrix-vector products can be computed efficiently, by first multiplying with $\widehat{M}_e$,
    and the result with $L^T$. For further information on the factorization procedure, we refer to
    [this publication](https://epubs.siam.org/doi/abs/10.1137/18M1175239).

    !!! warning
        The assembly procedure uses internals of dolfinx, which might be subject to change in
        future versions. The compiled assembly kernel is called with `double` buffers, so only
        real-valued, double precision PETSc builds are supported.

    Methods:
        assemble: Assemble the block factorization.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, mesh: dlx.mesh.Mesh, function_space: dlx.fem.FunctionSpace, form: ufl.Form
    ) -> None:
        """Initialize the block factorization assembler.

        Args:
            mesh (dlx.mesh.Mesh): Dolfinx Mesh object representing the computational domain.
            function_space (dlx.fem.FunctionSpace): Scalar function space of the FEM problem.
            form (ufl.Form): Weak form of the FEM matrix to factorize. The resulting matrix has to
                be symmetric positive definite on each cell, for the cell-wise Cholesky
                factorization to exist. Only cell integrals over the whole domain without
                coefficients or constants are supported, as the assembly calls the compiled cell
                kernel directly.

        Raises:
            TypeError: If PETSc is not built with real, double precision scalars.
            TypeError: If the mesh geometry is not given in double precision.
            ValueError: If the function space is not scalar-valued.
            ValueError: If the form contains other than cell integrals over the whole domain.
            ValueError: If the form contains coefficients or constants.
        """
        if np.dtype(PETSc.ScalarType) != np.float64:
            raise TypeError(
                "Matrix factorization assembly requires a PETSc build with float64 scalars, "
                f"but the scalar type is {np.dtype(PETSc.ScalarType)}."
            )
        if mesh.geometry.x.dtype != np.float64:
            raise TypeError(
                "Matrix factorization assembly requires float64 mesh coordinates, "
                f"but they have dtype {mesh.geometry.x.dtype}."
            )
        if function_space.dofmap.index_map_bs != 1:
            raise ValueError(
                "Matrix factorization assembly requires a scalar function space, but the DoF "
                f"block size is {function_space.dofmap.index_map_bs}."
            )
        _check_factorizable_form(form)
        cell_index_map = mesh.topology.index_map(mesh.topology.dim)
        self._communicator = mesh.comm
        self._vertex_coordinates = mesh.geometry.x
        self._cell_vertex_indices = mesh.geometry.dofmaps[0]
        self._dofmap = function_space.dofmap
        self._dof_index_map = function_space.dofmap.index_map
        self._num_owned_cells = cell_index_map.size_local
        self._num_global_cells = cell_index_map.size_global
        self._first_owned_cell = cell_index_map.local_range[0]
        self._num_cell_dofs = function_space.dofmap.dof_layout.num_dofs

        self._assembly_kernel = self._init_assembly_kernel(form)

    # ----------------------------------------------------------------------------------------------
    def assemble(self) -> tuple[PETSc.Mat, PETSc.Mat]:
        r"""Assemble the rectangular matrix factorization.

        Returns:
            tuple[PETSc.Mat, PETSc.Mat]: PETSc Matrices $\widehat{M}_e$ and $L$ representing
                the rectangular factorization of an FEM matrix.

        Raises:
            ValueError: If the local matrix of a cell is not symmetric positive definite.
        """
        block_diagonal_matrix, dof_map_matrix = self._set_up_petsc_mats()
        self._assemble_matrices_over_cells(block_diagonal_matrix, dof_map_matrix)
        block_diagonal_matrix.assemble()
        dof_map_matrix.assemble()
        return block_diagonal_matrix, dof_map_matrix

    # ----------------------------------------------------------------------------------------------
    def _init_assembly_kernel(self, form: ufl.Form) -> cffi.FFI.CData:
        """Initialize the dolfinx assembly kernel.

        Kernel objects in dolfinx generate local matrix contributions over a single mesh cell,
        corresponding to a provided weak form. They are typically called internally during
        the assembly process of FEM system matrices.
        Under the hood, this method compiles the form with dolfinx, and returns the
        `tabulate_tensor` method of its (single) cell integral.

        !!! warning
            This method uses internals of dolfinx, which might be subject to change in future
            versions.

        Args:
            form (ufl.Form): Weak form to compile per cell.

        Returns:
            cffi.FFI.CData: Compiled callable, returning the local FEM matrix contribution for a
                single mesh cell.
        """
        compiled_form = dlx.fem.form(form, dtype=np.float64)
        return compiled_form.ufcx_form.form_integrals[0].tabulate_tensor_float64

    # ----------------------------------------------------------------------------------------------
    def _set_up_petsc_mats(self) -> tuple[PETSc.Mat, PETSc.Mat]:
        r"""Initialize the PETSc matrices $\widehat{M}_e$ and $L$ for the assembly process.

        The parallel layouts are set explicitly: every process owns the block rows of its owned
        cells, which dolfinx numbers contiguously. The columns of $L$ follow the DoF layout of
        dolfinx, so that $L$ and $L^T$ can be applied to vectors of dolfinx-assembled matrices.
        All entries of $\widehat{M}_e$ lie in the diagonal block of the owning process. The
        single entry per row of $L$ may belong to a DoF owned by another process.

        Returns:
            tuple[PETSc.Mat, PETSc.Mat]: Empty block diagonal matrix $\widehat{M}_e$ of shape
                $(N_c N_e, N_c N_e)$ and DoF map matrix $L$ of shape $(N_c N_e, N)$.
        """
        block_row_sizes = (
            self._num_owned_cells * self._num_cell_dofs,
            self._num_global_cells * self._num_cell_dofs,
        )
        dof_sizes = (self._dof_index_map.size_local, self._dof_index_map.size_global)

        block_diagonal_matrix = PETSc.Mat().createAIJ(
            (block_row_sizes, block_row_sizes), comm=self._communicator
        )
        block_diagonal_matrix.setPreallocationNNZ((self._num_cell_dofs, 0))
        block_diagonal_matrix.setUp()

        dof_map_matrix = PETSc.Mat().createAIJ(
            (block_row_sizes, dof_sizes), comm=self._communicator
        )
        dof_map_matrix.setPreallocationNNZ((1, 1))
        dof_map_matrix.setUp()

        return block_diagonal_matrix, dof_map_matrix

    # ----------------------------------------------------------------------------------------------
    def _get_block_indices(
        self, global_cell_ind: int
    ) -> np.ndarray[tuple[int], np.dtype[np.integer]]:
        """Return the global row indices of the block belonging to a mesh cell."""
        return np.arange(
            global_cell_ind * self._num_cell_dofs,
            (global_cell_ind + 1) * self._num_cell_dofs,
            dtype=PETSc.IntType,
        )

    # ----------------------------------------------------------------------------------------------
    def _assemble_matrices_over_cells(
        self, block_diagonal_matrix: PETSc.Mat, dof_map_matrix: PETSc.Mat
    ) -> None:
        r"""Assemble the matrices $\widehat{M}_e$ and $L$.

        The method loops over the owned cells and invokes the kernel for the local FEM matrix
        contributions. It further computes the Cholesky factor of each local matrix, and
        inserts the result and the cell DoF mapping into the matrices $\widehat{M}_e$ and $L$,
        respectively. Ghost cells are skipped, so that every cell contributes exactly once.
        Rows and columns are inserted with global indices.

        Args:
            block_diagonal_matrix (PETSc.Mat): Block diagonal matrix $\widehat{M}_e$ for cell
                Cholesky factors.
            dof_map_matrix (PETSc.Mat): DoF map matrix $L$.
        """
        for cell_ind in range(self._num_owned_cells):
            block_indices = self._get_block_indices(self._first_owned_cell + cell_ind)
            global_cell_dofs = self._dof_index_map.local_to_global(
                self._dofmap.cell_dofs(cell_ind)
            ).astype(PETSc.IntType)
            cell_vertex_coordinates = self._vertex_coordinates[self._cell_vertex_indices[cell_ind]]
            cell_matrix = np.zeros((self._num_cell_dofs, self._num_cell_dofs), dtype=np.float64)
            self._assembly_kernel(
                ffi.cast("double *", ffi.from_buffer(cell_matrix)),  # A - output matrix
                ffi.NULL,  # w - coefficient values (none for this form)
                ffi.NULL,  # c - constants (none for this form)
                ffi.cast("double *", ffi.from_buffer(cell_vertex_coordinates)),  # coordinate_dofs
                ffi.NULL,  # entity_local_index (cell integral)
                ffi.NULL,  # quadrature_permutation
                ffi.NULL,  # custom_data (only used for runtime integrals)
            )
            try:
                cell_matrix_factor = np.linalg.cholesky(cell_matrix)
            except np.linalg.LinAlgError as error:
                raise ValueError(
                    "Matrix factorization assembly requires a form that is symmetric positive "
                    f"definite on each cell, but the matrix of cell {cell_ind} is not."
                ) from error
            block_diagonal_matrix.setValues(
                block_indices,
                block_indices,
                cell_matrix_factor,
                addv=PETSc.InsertMode.INSERT_VALUES,
            )
            for row_ind, col_ind in zip(block_indices, global_cell_dofs, strict=True):
                dof_map_matrix.setValues(row_ind, col_ind, 1.0, addv=PETSc.InsertMode.INSERT_VALUES)
