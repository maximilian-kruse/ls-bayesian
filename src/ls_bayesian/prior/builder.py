"""Builders for prior objects from lower-level components.

Classes:
    BilaplacianPriorSettings: Settings for the bilaplacian prior builder.
    BilaplacianPriorBuilder: Builder for a Bilaplacian prior.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Annotated

import dolfinx as dlx
import numpy as np
from beartype.vale import Is
from dolfinx.fem import petsc
from petsc4py import PETSc

from . import components, fem, prior


# ==================================================================================================
@dataclass
class BilaplacianPriorSettings:
    r"""Settings for the bilaplacian prior builder.

    This dataclass collects all configuration options required to set up a biplaplacian prior
    using the [`BilaplacianPriorBuilder`][ls_prior.builder.BilaplacianPriorBuilder] class.
    The builder distributes these settings to the respective components that are assembled within
    the builder.

    Attributes:
        mesh (dlx.mesh.Mesh): Dolfinx mesh on which the prior is defined.
        mean_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Mean vector of the prior,
            vertex-based representation.
        kappa (Real): Parameter $\kappa$ in the SPDE formulation of the prior
        tau (Real): Parameter $\tau$ in the SPDE formulation of the prior
        robin_const (Real | None): Robin boundary condition constant. If `None`, homogeneous
            Neumann boundary conditions are applied. Defaults to `None`.
        seed (int): Random seed for the internal random number generator. Defaults to `0`.
        fe_data (tuple[str, int]): Finite element type and degree used for the function space
            setup. Defaults to `("CG", 1)`.
        cg_relative_tolerance (Real | None): Relative tolerance for the CG solver used in the
            application of the precision operator. If `None`, the default PETSc tolerance is used.
            Defaults to `None`.
        cg_absolute_tolerance (Real | None): Absolute tolerance for the CG solver used in the
            application of the precision operator. If `None`, the default PETSc tolerance is used.
            Defaults to `None`.
        cg_max_iterations (int | None): Maximum number of iterations for the CG solver used in
            the application of the precision operator. If `None`, the default PETSc value is used
            Defaults to `None`.
        amg_relative_tolerance (Real | None): Relative tolerance for the AMG solver used in the
            application of the covariance operator and its factorization. If `None`, the default
            PETSc tolerance is used. Defaults to `None`.
        amg_absolute_tolerance (Real | None): Absolute tolerance for the AMG solver used in the
            application of the covariance operator and its factorization. If `None`, the default
            PETSc tolerance is used. Defaults to `None`.
        amg_max_iterations (int | None): Maximum number of iterations for the AMG solver used in
            the application of the covariance operator and its factorization. If `None`, the
            default PETSc value is used. Defaults to `None`.
    """

    mesh: dlx.mesh.Mesh
    mean_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    kappa: Annotated[Real, Is[lambda x: x > 0]]
    tau: Annotated[Real, Is[lambda x: x > 0]]
    robin_const: Real | None = None
    seed: Real = 0
    fe_data: tuple[str, Annotated[int, Is[lambda x: x > 0]]] = ("CG", 1)
    cg_relative_tolerance: Annotated[Real, Is[lambda x: x > 0]] | None = None
    cg_absolute_tolerance: Annotated[Real, Is[lambda x: x > 0]] | None = None
    cg_max_iterations: Annotated[int, Is[lambda x: x > 0]] | None = None
    amg_relative_tolerance: Annotated[Real, Is[lambda x: x > 0]] | None = None
    amg_absolute_tolerance: Annotated[Real, Is[lambda x: x > 0]] | None = None
    amg_max_iterations: Annotated[int, Is[lambda x: x > 0]] | None = None


# ==================================================================================================
class BilaplacianPriorBuilder:
    r"""Builder for a Bilaplacian prior.

    Specific builder class for a bilaplacian prior, i.e. the distribution that arises from solving
    the SPDE $\tau(\kappa^2 - \Delta)^2 m = W$.

    Methods:
        build: Build the Bilaplacian prior.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: BilaplacianPriorSettings) -> None:
        """Initialize the builder with the given settings.

        Args:
            settings (BilaplacianPriorSettings): Settings for the Bilaplacian prior.
        """
        self._mesh = settings.mesh
        self._mean_vector = settings.mean_vector
        self._kappa = settings.kappa
        self._tau = settings.tau
        self._robin_const = settings.robin_const
        self._seed = settings.seed
        self._fe_data = settings.fe_data

        self._cg_solver_settings = components.InverseMatrixSolverSettings(
            solver_type=PETSc.KSP.Type.CG,
            preconditioner_type=PETSc.PC.Type.JACOBI,
            relative_tolerance=settings.cg_relative_tolerance,
            absolute_tolerance=settings.cg_absolute_tolerance,
            max_num_iterations=settings.cg_max_iterations,
        )
        self._amg_solver_settings = components.InverseMatrixSolverSettings(
            solver_type=PETSc.KSP.Type.CG,
            preconditioner_type=PETSc.PC.Type.GAMG,
            relative_tolerance=settings.amg_relative_tolerance,
            absolute_tolerance=settings.amg_absolute_tolerance,
            max_num_iterations=settings.amg_max_iterations,
        )

    # ----------------------------------------------------------------------------------------------
    def build(self) -> prior.Prior:
        """Build the Bilaplacian prior.

        Internally, this method assembles all dolfinx structures, composes a hierarchy of
        [`PETScComponents`][ls_prior.components.PETScComponent], wraps them in
        [`InterfaceComponents`][ls_prior.components.InterfaceComponent] and hands them to the
        [`Prior`][ls_prior.prior.Prior] class.

        Returns:
            prior.Prior: The constructed Bilaplacian prior.
        """
        mass_matrix, spde_matrix, block_diagonal_matrix, dof_map_matrix, converter = (
            self._build_fem_structures()
        )
        precision_operator, covariance_operator, sampling_factor = self._build_components(
            mass_matrix, spde_matrix, block_diagonal_matrix, dof_map_matrix
        )
        precision_operator_interface, covariance_operator_interface, sampling_factor_interface = (
            self._build_interfaces(precision_operator, covariance_operator, sampling_factor)
        )
        bilaplace_prior = prior.Prior(
            self._mean_vector,
            precision_operator_interface,
            covariance_operator_interface,
            sampling_factor_interface,
            converter,
            seed=self._seed,
        )
        return bilaplace_prior

    # ----------------------------------------------------------------------------------------------
    def _build_fem_structures(
        self,
    ) -> tuple[PETSc.Mat, PETSc.Mat, PETSc.Mat, PETSc.Mat, fem.FEMConverter]:
        r"""Assemble FEM data structures, i.e. FEM Matrices and `FEMConverter` object.

        Returns:
            tuple[PETSc.Mat, PETSc.Mat, PETSc.Mat, fem.FEMConverter]:
                mass matrix $M$, SPDE matrix $A$, block-diagonal matrix $\widehat{M}_w$,
                DoF map matrix $L$, [`FEMConverter`][ls_prior.fem.FEMConverter] object.
        """
        function_space = dlx.fem.functionspace(self._mesh, self._fe_data)
        mass_matrix_form, spde_matrix_form = fem.generate_forms(
            function_space, self._kappa, self._tau, self._robin_const
        )
        mass_matrix = petsc.assemble_matrix(dlx.fem.form(mass_matrix_form))
        spde_matrix = petsc.assemble_matrix(dlx.fem.form(spde_matrix_form))
        mass_matrix.assemble()
        spde_matrix.assemble()
        mass_matrix_factorization = fem.FEMMatrixFactorizationAssembler(
            self._mesh, function_space, mass_matrix_form
        )
        block_diagonal_matrix, dof_map_matrix = mass_matrix_factorization.assemble()
        dof_map_matrix.transpose()
        converter = fem.FEMConverter(function_space)

        return mass_matrix, spde_matrix, block_diagonal_matrix, dof_map_matrix, converter

    # ----------------------------------------------------------------------------------------------
    def _build_components(
        self,
        mass_matrix: PETSc.Mat,
        spde_matrix: PETSc.Mat,
        block_diagonal_matrix: PETSc.Mat,
        dof_map_matrix: PETSc.Mat,
    ) -> tuple[components.PETScComponent, components.PETScComponent, components.PETScComponent]:
        r"""Build PETSc components for bilaplacian prior from FEM matrices.

        The main components for the prior are:

        1. Precision operator: $\mathcal{C}^{-1} = A M^{-1} A$
        2. Covariance operator: $\mathcal{C} = A^{-1} M A^{-1}$
        3. Sampling factor: $\widehat{\mathcal{C}} = A^{-1} L^T \widehat{M}_e$

        Args:
            mass_matrix (PETSc.Mat): Mass matrix $M$.
            spde_matrix (PETSc.Mat): SPDE matrix $A$.
            block_diagonal_matrix (PETSc.Mat): Block-diagonal matrix $\widehat{M}_e$.
            dof_map_matrix (PETSc.Mat): DoF map matrix $L$.

        Returns:
            tuple[components.PETScComponent,
                  components.PETScComponent,
                  components.PETScComponent]:
                Precision operator $\mathcal{C}^{-1}$,
                covariance operator $\mathcal{C}$,
                sampling factor $\widehat{\mathcal{C}}$.
        """
        # Set up base components
        mass_matrix_component = components.Matrix(mass_matrix)
        spde_matrix_component = components.Matrix(spde_matrix)
        block_diagonal_matrix_component = components.Matrix(block_diagonal_matrix)
        dof_map_matrix_component = components.Matrix(dof_map_matrix)
        mass_matrix_inverse_component = components.InverseMatrixSolver(
            self._cg_solver_settings, mass_matrix
        )
        spde_matrix_inverse_component = components.InverseMatrixSolver(
            self._amg_solver_settings, spde_matrix
        )
        # Bilaplacian precision:C^{-1} = A M^{-1} A
        precision_operator = components.PETScComponentComposition(
            spde_matrix_component, mass_matrix_inverse_component, spde_matrix_component
        )
        # Bilaplacian covariance: C = A^{-1} M A^{-1}
        covariance_operator = components.PETScComponentComposition(
            spde_matrix_inverse_component, mass_matrix_component, spde_matrix_inverse_component
        )
        # Sampling factor: \widehat{C} = A^{-1} \widehat{M} = A^{-1} L^T \widehat{M_e}
        sampling_factor = components.PETScComponentComposition(
            block_diagonal_matrix_component, dof_map_matrix_component, spde_matrix_inverse_component
        )
        return precision_operator, covariance_operator, sampling_factor

    # ----------------------------------------------------------------------------------------------
    def _build_interfaces(
        self,
        precision_operator: components.PETScComponent,
        covariance_operator: components.PETScComponent,
        sampling_factor: components.PETScComponent,
    ) -> tuple[
        components.InterfaceComponent, components.InterfaceComponent, components.InterfaceComponent
    ]:
        r"""Wrap PETSc components in interface components.

        Args:
            precision_operator (components.PETScComponent): Precision operator $\mathcal{C}^{-1}$.
            covariance_operator (components.PETScComponent): SPDE matrix $A$.
            sampling_factor (components.PETScComponent): Sampling factor $\widehat{\mathcal{C}}$.

        Returns:
            tuple[components.InterfaceComponent,
                  components.InterfaceComponent,
                  components.InterfaceComponent]: Wrapped interface components.
        """
        precision_operator_interface = components.InterfaceComponent(precision_operator)
        covariance_operator_interface = components.InterfaceComponent(covariance_operator)
        sampling_factor_interface = components.InterfaceComponent(sampling_factor)
        return (
            precision_operator_interface,
            covariance_operator_interface,
            sampling_factor_interface,
        )
