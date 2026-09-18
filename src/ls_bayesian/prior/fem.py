"""This module contains all FEM specific functionality in this package, based on dolfinx.

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
import scifem as dlx_helper
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
    $\beta$ is the optional `robin_const` parameter that enforces Robin boundary
    conditions of the form $\nabla \phi \cdot n + \beta \phi = 0$, instead of homogeneous Neumann
    boundary conditions.

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
    An `FEMConverter` object takes such data structures and inerpolates them to the provided
    function space. On the other hand, it can interpolate any data defined on the DoFs of the
    underlying function space onto the vertices of the mesh.
    Internally, the `FEMConverter` assigns vertex based input data to the DoFs of a P1
    function space (whose degrees of freedom are exactly the vertices).
    It subsequently utilizes dolfinx's efficient interpolation between function spaces.
    On the other hand, data from some function space is interpolated to a P1 space, and
    subsequently extracted to vertex values.

    Methods:
        convert_vertex_values_to_dofs: Convert vertex based data to DoF representation
        convert_dofs_to_vertex_values: Convert DoF based data to vertex representation
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, function_space: dlx.fem.FunctionSpace) -> None:
        """Initialize the converter for a given dolfinx function space.

        The function space implicitly carries the mesh, and thus the vertices we want to convert
        from/to.

        Args:
            function_space (dlx.fem.FunctionSpace): Function space in which degrees of freedom lie.
        """
        vertex_space = dlx.fem.functionspace(function_space.mesh, ("Lagrange", 1))
        self._dof_function = dlx.fem.Function(function_space)
        self._vertex_function = dlx.fem.Function(vertex_space)
        self.dof_space_dim = function_space.dofmap.index_map.size_local
        self.vertex_space_dim = vertex_space.dofmap.index_map.size_local
        self._vertex_to_dof_map = dlx_helper.vertex_to_dofmap(vertex_space)
        self._dof_to_vertex_map = dlx_helper.dof_to_vertexmap(vertex_space)

    # ----------------------------------------------------------------------------------------------
    def convert_vertex_values_to_dofs(
        self, vertex_values: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Convert vertex based data to DoF representation.

        Args:
            vertex_values (np.ndarray[tuple[int], np.dtype[np.float64]]): Array of data defined on
                the vertices of the underlying computational mesh.

        Raises:
            ValueError: Checks that `vertex_values` has the same dimension as a P1 function space
                on the underlying mesh.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Input data interpolated to the required
                function space DoFs, copied.
        """
        if not vertex_values.shape == (self.vertex_space_dim,):
            raise ValueError(
                f"Expected vertex_values to have shape {(self.vertex_space_dim,)}, "
                f"but got {vertex_values.shape}"
            )
        self._vertex_function.x.array[:] = vertex_values[self._vertex_to_dof_map]
        self._vertex_function.x.scatter_forward()
        self._dof_function.interpolate(self._vertex_function)
        assert self._dof_function.x.array.shape == (self.dof_space_dim,), (
            f"Created PETSc vector has size {self._dof_function.x.array.shape}, "
            f"but expected {(self.dof_space_dim,)}"
        )
        return self._dof_function.x.array.copy()

    # ----------------------------------------------------------------------------------------------
    def convert_dofs_to_vertex_values(
        self, dof_values: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Convert DoF based data to vertex representation.

        Args:
            dof_values (np.ndarray[tuple[int], np.dtype[np.float64]]): Array of data defined on
                the DoFs of the underlying function space.

        Raises:
            ValueError: Checks that the dimension of the input vector matches the number of DoFs
                on the underlying function space, copied.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Array of data interpolated to the vertices
                of the underlying mesh.
        """
        if not dof_values.shape[0] == self.dof_space_dim:
            raise ValueError(
                f"Expected dof_values to have size {self.dof_space_dim}, "
                f"but got {dof_values.shape[0]}"
            )
        self._dof_function.x.array[:] = dof_values
        self._dof_function.x.scatter_forward()
        self._vertex_function.interpolate(self._dof_function)
        assert self._vertex_function.x.array.shape == (self.vertex_space_dim,), (
            f"Created array has size {self._vertex_function.x.array.shape}, "
            f"but expected {(self.vertex_space_dim,)}"
        )
        vertex_values = self._vertex_function.x.array[self._dof_to_vertex_map]
        return vertex_values.copy()


# ==================================================================================================
class FEMMatrixFactorizationAssembler:
    r"""Assembler for the rectangular factorization of an FEM matrix.

    This class provides the functionality for efficient factorization of a finite element matrix
    $M$, i.e. it implements the (sparse representation of an) assembly of a rectangular matrix
    $\widehat{M}$ s.th. $M = \widehat{M}\widehat{M}^T$. The factorization exploits the
    characteristics of standard finite element assembly procedures. This assembly is typically done
    locally for the contribution of each mesh cell. The contributions are then gathered into a
    global  matrix, with overlap at indices of vertices that are shared between cells. Now let $N$
    be the number of degrees of freedom of the finite element space, $M$ the number of cells in the
    underlying mesh, and $N_e$ the number of degrees of freedom per cell. The matrix $M$ clearly has
    size $(N, N)$. Importantly, it can be decomposed as $M=L^T M_e L$. Here, the Matrix
    $M_e\in \mathbb{R}^{MN_e}$ is block-diagonal, containing the local FEM matrix
    contributions over all cells in each block. $L\in \mathbb{R}^{MN_e \times N}$ is a sparse
    matrix that maps the local cell DoFs in each block $M_e$ to their respective global DoFs. We
    can efficiently compute the Cholesky factor $\widehat{M}_e$ of $M_e$ by computing the
    cholesky factorization of each individual block. In particular, $\widehat{M}_e$ is still sparse,
    as opposed to standard Cholesky factors of sparse matrices. $\widehat{M}_e$ and $L$ now
    represent a sparse factorization $\widehat{M} = L^T \widehat{M}_e$ as above.
    The two matrices shouldn't be multiplied directly to avoid fill-in effects. However,
    matrix-vector products can be computed efficiently, by first multiplying with $\widehat{M}_e$,
    and the result with $L$. For further information on the factorization procedure, we refer to
    [this publication](https://epubs.siam.org/doi/abs/10.1137/18M1175239).

    !!! warning
        The assembly procedure uses internals of dolfinx, which might be subject to change in
        future versions.

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
            function_space (dlx.fem.FunctionSpace): Function space of the FEM problem.
            form (ufl.Form): Weak form of the FEM matrix to factorize.
        """
        self._mpi_communicator = mesh.comm
        self._local_vertex_coordinates = mesh.geometry.x
        self._local_cell_vertex_indices = mesh.geometry.dofmaps[0]
        self._pproc_cell_distribution_map = mesh.topology.index_map(mesh.topology.dim)
        self._dofmap = function_space.dofmap
        self._pproc_dof_distribution_map = function_space.dofmap.index_map

        self._num_local_cells = self._pproc_cell_distribution_map.size_local
        self._num_global_cells = self._pproc_cell_distribution_map.size_global
        self._num_local_dofs = self._pproc_dof_distribution_map.size_local
        self._num_global_dofs = self._pproc_dof_distribution_map.size_global
        self._num_cell_dofs = function_space.dofmap.dof_layout.num_dofs

        self._assembly_kernel = self._init_assembly_kernel(mesh.comm, form)

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
    def _init_assembly_kernel(self, mpi_communicator: MPI.Comm, form: ufl.Form) -> cffi.FFI.CData:
        """Initialize the dolfinx assembly kernel.

        Kernel objects in dolfinx generate local matrix contributions over a single mesh cell,
        corresponding to a provided weak form. They are typically called internally during
        the assembly process of FEM system matrices.
        Under the hood, this method uses the `ffcx` jit compiler, and returns the resulting kernel's
        `tabulate_tensor` method.

        !!! warning
            This method uses internals of dolfinx, which might be subject to change in future
            versions.

        Args:
            mpi_communicator (MPI.Comm): MPI communicator for the kernel to use.
            form (ufl.Form): Weak form to compile per cell.

        Returns:
            cffi.FFI.CData: Compiled callable, returning the local FEM matrix contribution for a
                a single mesh cell.
        """
        form_compiled, *_ = dlx.jit.ffcx_jit(
            mpi_communicator,
            form,
            form_compiler_options={"scalar_type": PETSc.ScalarType},
        )
        compiled_kernel = getattr(
            form_compiled.form_integrals[0],
            f"tabulate_tensor_{np.dtype(PETSc.ScalarType).name}",
        )
        return compiled_kernel

    # ----------------------------------------------------------------------------------------------
    def _set_up_petsc_mats(self) -> tuple[PETSc.Mat, PETSc.Mat]:
        r"""Initialize the PETSc matrices $M_e$ and $L$ for the assembly process.

        Returns:
            tuple[PETSc.Mat, PETSc.Mat]: Empty block diagonal matrix $M_e$ and local-to-global
                DoF matrix $L$.
        """
        block_diagonal_matrix = PETSc.Mat().createAIJ(
            [
                self._num_global_cells * self._num_cell_dofs,
                self._num_global_cells * self._num_cell_dofs,
            ],
            comm=self._mpi_communicator,
        )
        block_diagonal_matrix.setPreallocationNNZ(self._num_cell_dofs)
        block_diagonal_matrix.setUp()

        dof_map_matrix = PETSc.Mat().createAIJ(
            [self._num_global_cells * self._num_cell_dofs, self._num_global_dofs],
            comm=self._mpi_communicator,
        )
        dof_map_matrix.setPreallocationNNZ(1)
        dof_map_matrix.setUp()

        return block_diagonal_matrix, dof_map_matrix

    # ----------------------------------------------------------------------------------------------
    def _insert_in_block_diagonal_matrix(
        self,
        global_ind: np.integer,
        cell_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        block_diagonal_matrix: PETSc.Mat,
    ) -> None:
        r"""Insert FEM matrix Cholesky factor of single mesh cell into block-diagonal matrix $M_e$.

        Insertion is done in-place.

        Args:
            global_ind (np.integer): Global index of the current mesh cell.
            cell_matrix (np.ndarray[tuple[int, int], np.dtype[np.float64]]): Local cell matrix
                contribution values.
            block_diagonal_matrix (PETSc.Mat): Global matrix $M_e$ to insert into.
        """
        row_col_inds = np.arange(
            global_ind * self._num_cell_dofs,
            (global_ind + 1) * self._num_cell_dofs,
            dtype=PETSc.IntType,
        )
        block_diagonal_matrix.setValues(
            row_col_inds,
            row_col_inds,
            cell_matrix,
            addv=PETSc.InsertMode.INSERT_VALUES,
        )

    # ----------------------------------------------------------------------------------------------
    def _insert_in_dof_map_matrix(
        self,
        global_ind: np.integer,
        global_cell_dofs: np.ndarray[tuple[int], np.dtype[np.integer]],
        dof_map_matrix: PETSc.Mat,
    ) -> None:
        """Insert global DoFs of a single mesh cell into local-to-global DoF matrix $L$.

        Insertion is done in-place.

        Args:
            global_ind (np.integer): Global index of the current mesh cell.
            global_cell_dofs (np.ndarray[tuple[int], np.dtype[np.integer]]):
                Global indices of the DoFs in the current cell.
            dof_map_matrix (PETSc.Mat): Local-to-global DoF matrix $L$.
        """
        row_inds = np.arange(
            global_ind * self._num_cell_dofs,
            (global_ind + 1) * self._num_cell_dofs,
            dtype=PETSc.IntType,
        )
        for row_ind, col_ind in zip(row_inds, global_cell_dofs, strict=True):
            dof_map_matrix.setValues(
                row_ind,
                col_ind,
                1.0,
                addv=PETSc.InsertMode.INSERT_VALUES,
            )

    # ----------------------------------------------------------------------------------------------
    def _assemble_matrices_over_cells(
        self, block_diagonal_matrix: PETSc.Mat, dof_map_matrix: PETSc.Mat
    ) -> None:
        """Assemble the matrices $M_e$ and $L$.

        The method loops over all cells and invokes the kernel for the local FEM matrix
        contributions. It further computes the Cholesky factor of each local matrix, and
        inserts the result and global index mapping into the matrices $M_e$ and $L$, respectively.

        Args:
            block_diagonal_matrix (PETSc.Mat): Block diagonal matrix $M_e$ for cell
                Cholesky factors.
            dof_map_matrix (PETSc.Mat): Local-to-global DoF matrix $L$.
        """
        local_cell_inds = np.arange(self._num_local_cells, dtype=np.int64)
        global_cell_inds = self._pproc_cell_distribution_map.local_to_global(local_cell_inds)

        for local_ind, global_ind in zip(local_cell_inds, global_cell_inds, strict=True):
            local_cell_dofs = self._dofmap.cell_dofs(local_ind)
            global_cell_dofs = self._pproc_dof_distribution_map.local_to_global(
                local_cell_dofs
            ).astype(PETSc.IntType)
            cell_vertex_inds = self._local_cell_vertex_indices[local_ind]
            cell_vertex_coordinates = self._local_vertex_coordinates[cell_vertex_inds]
            cell_matrix = np.zeros(
                (self._num_cell_dofs, self._num_cell_dofs), dtype=PETSc.ScalarType
            )
            self._assembly_kernel(
                ffi.cast("double *", ffi.from_buffer(cell_matrix)),  # A - output matrix
                ffi.NULL,  # w - coefficient values (none for this form)
                ffi.NULL,  # c - constants (none for this form)
                ffi.cast("double *", ffi.from_buffer(cell_vertex_coordinates)),  # coordinate_dofs
                ffi.NULL,  # entity_local_index (cell integral)
                ffi.NULL,  # quadrature_permutation
                ffi.NULL,  # custom_data (only used for runtime integrals)
            )
            cell_matrix = np.linalg.cholesky(cell_matrix)
            self._insert_in_block_diagonal_matrix(global_ind, cell_matrix, block_diagonal_matrix)
            self._insert_in_dof_map_matrix(global_ind, global_cell_dofs, dof_map_matrix)
