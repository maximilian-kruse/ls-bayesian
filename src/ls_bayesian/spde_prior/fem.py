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
"""

from numbers import Real
from typing import Annotated

import cffi
import dolfinx as dlx
import numpy as np
import scifem
import ufl
from beartype.vale import Is
from mpi4py import MPI
from petsc4py import PETSc

ffi = cffi.FFI()


# ==================================================================================================
def generate_forms(
    function_space: dlx.fem.FunctionSpace,
    kappa: Annotated[Real, Is[lambda x: x > 0]],
    tau: Annotated[Real, Is[lambda x: x > 0]],
    robin_const: Real | None = None,
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
    $\tau \nabla \phi \cdot n + \beta \phi = 0$.

    Args:
        function_space (dlx.fem.FunctionSpace): dolfinx function space to construct forms over.
        kappa (Real): Parameter $\kappa$ of the prior field.
        tau (Real): Parameter $\tau$ of the prior field.
        robin_const (Real | None, optional): Parameter $\beta$ of the prior field enforcing
            Robin boundary conditions. Defaults to None.

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


# ==================================================================================================
class FEMConverter:
    """Converter between vertex based data and DoF representation on a dolfinx function space.

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

    Methods:
        convert_vertex_values_to_dofs: Convert vertex based data to DoF representation
        convert_dofs_to_vertex_values: Convert DoF based data to vertex representation

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

        self._dof_function = dlx.fem.Function(function_space)
        self._vertex_function = dlx.fem.Function(vertex_space)
        self.comm = mesh.comm
        self.vertex_space_dim = vertex_index_map.size_global
        self.dof_space_dim = dof_index_map.size_local
        self.global_dof_space_dim = dof_index_map.size_global

        self._num_owned_vertices = vertex_index_map.size_local
        self._vertex_to_dof_map = scifem.vertex_to_dofmap(vertex_space)
        local_vertices = np.arange(
            vertex_index_map.size_local + vertex_index_map.num_ghosts, dtype=np.int32
        )
        mesh.topology.create_connectivity(0, mesh.topology.dim)
        vertex_geometry_nodes = dlx.mesh.entities_to_geometry(mesh, 0, local_vertices).reshape(-1)
        self._vertex_input_indices = mesh.geometry.input_global_indices[vertex_geometry_nodes]

        owned_cell_input_indices = mesh.topology.original_cell_index[: cell_index_map.size_local]
        self.cell_block_input_indices = (
            owned_cell_input_indices[:, None] * num_cell_dofs + np.arange(num_cell_dofs)
        ).reshape(-1)

    # ----------------------------------------------------------------------------------------------
    def convert_vertex_values_to_dofs(
        self, vertex_values: np.ndarray[tuple[int], np.dtype[np.float64]]
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
        if not vertex_values.shape == (self.vertex_space_dim,):
            raise ValueError(
                f"Expected vertex_values to have shape {(self.vertex_space_dim,)}, "
                f"but got {vertex_values.shape}"
            )
        # All local vertices, including ghosts, are set from the complete vector, so no ghost
        # update is required before interpolation.
        self._vertex_function.x.array[self._vertex_to_dof_map] = vertex_values[
            self._vertex_input_indices
        ]
        self._dof_function.interpolate(self._vertex_function)
        return self._dof_function.x.array[: self.dof_space_dim].copy()

    # ----------------------------------------------------------------------------------------------
    def convert_dofs_to_vertex_values(
        self, dof_values: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Convert DoF based data to vertex representation.

        Args:
            dof_values (np.ndarray[tuple[int], np.dtype[np.float64]]): Array of data defined on
                the owned DoFs of the underlying function space, shape `(dof_space_dim,)`.

        Raises:
            ValueError: Checks that the shape of the input vector matches the number of owned DoFs
                on the underlying function space.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Array of data interpolated to the vertices
                of the underlying mesh, in input node order, shape `(vertex_space_dim,)`. The array
                is identical on all processes.
        """
        if not dof_values.shape == (self.dof_space_dim,):
            raise ValueError(
                f"Expected dof_values to have shape {(self.dof_space_dim,)}, "
                f"but got {dof_values.shape}"
            )
        self._dof_function.x.array[: self.dof_space_dim] = dof_values
        self._dof_function.x.scatter_forward()
        self._vertex_function.interpolate(self._dof_function)
        owned_vertex_values = self._vertex_function.x.array[
            self._vertex_to_dof_map[: self._num_owned_vertices]
        ]
        return _gather_by_index(
            self.comm,
            self._vertex_input_indices[: self._num_owned_vertices],
            owned_vertex_values,
            self.vertex_space_dim,
        )


# ==================================================================================================
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
                factorization to exist.

        Raises:
            TypeError: If PETSc is not built with real, double precision scalars.
            TypeError: If the mesh geometry is not given in double precision.
            ValueError: If the function space is not scalar-valued.
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
            cell_matrix_factor = np.linalg.cholesky(cell_matrix)
            block_diagonal_matrix.setValues(
                block_indices,
                block_indices,
                cell_matrix_factor,
                addv=PETSc.InsertMode.INSERT_VALUES,
            )
            for row_ind, col_ind in zip(block_indices, global_cell_dofs, strict=True):
                dof_map_matrix.setValues(row_ind, col_ind, 1.0, addv=PETSc.InsertMode.INSERT_VALUES)
