"""Builders for prior objects from lower-level components.

Classes:
    SPDEPriorSettings: Settings for the SPDE prior builder.
    SPDEComponentStrategy: ABC interface for the composition of components into precision,
        covariance and sampling-factor operators.
    SPDEPriorBuilder: Builder for an SPDE-based prior, parameterized by a
        [`SPDEComponentStrategy`][ls_bayesian.spde_prior.builder.SPDEComponentStrategy].
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import Annotated

import dolfinx as dlx
import numpy as np
from beartype.vale import Is
from dolfinx.fem import petsc
from petsc4py import PETSc

from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.spde_prior import components, fem, spde_prior


# ==================================================================================================
@dataclass
class SPDEPriorSettings:
    r"""Settings for the SPDE prior builder.

    This dataclass collects all configuration options required to set up an SPDE-based prior using
    the [`SPDEPriorBuilder`][ls_bayesian.spde_prior.builder.SPDEPriorBuilder] class. The builder
    distributes these settings to the respective components that are assembled within the builder.
    The field constraints are validated on initialization.

    Attributes:
        mesh (dlx.mesh.Mesh): Dolfinx mesh on which the prior is defined.
        mean_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Mean vector of the prior,
            vertex-based representation.
        kappa (Real): Parameter $\kappa > 0$ in the SPDE formulation of the prior.
        tau (Real): Parameter $\tau > 0$ in the SPDE formulation of the prior.
        robin_const (Real | None): Robin boundary condition constant $\beta \geq 0$. If `None`,
            homogeneous Neumann boundary conditions are applied. Defaults to `None`.
        seed (int): Random seed for the internal random number generator. Defaults to `0`.
        fe_data (tuple[str, int]): Finite element family and degree used for the function space
            setup. The family has to be continuous Lagrange, i.e. one of
            `LAGRANGE_FAMILY_NAMES`. Defaults to `("Lagrange", 1)`.
        cg_relative_tolerance (Real): Relative tolerance for the CG solver used in the
            application of the precision operator. Defaults to `1e-12`: the precision, covariance
            and sampling operators are applied through these solves, so their accuracy bounds the
            consistency of cost, gradient and Hessian-vector products of the prior. The value
            keeps this error far below typical optimizer and finite difference tolerances, while
            being attainable in double precision for the well-conditioned, preconditioned mass and
            SPDE systems.
        cg_absolute_tolerance (Real | None): Absolute tolerance for the CG solver used in the
            application of the precision operator. If `None`, the PETSc default of $10^{-50}$
            is used, i.e. effectively only the relative criterion applies. Defaults to `None`.
        cg_max_iterations (int): Maximum number of iterations for the CG solver used in the
            application of the precision operator. Defaults to `1000`, a safeguard against
            stagnating solves: the preconditioned systems typically converge in far fewer
            iterations, and exceeding the limit raises an error instead of returning an inaccurate
            result.
        amg_relative_tolerance (Real): Relative tolerance for the AMG-preconditioned solver used
            in the application of the covariance operator and its factorization. Defaults to
            `1e-12`, for the same reasons as `cg_relative_tolerance`.
        amg_absolute_tolerance (Real | None): Absolute tolerance for the AMG-preconditioned
            solver used in the application of the covariance operator and its factorization. If
            `None`, the PETSc default of $10^{-50}$ is used, i.e. effectively only the relative
            criterion applies. Defaults to `None`.
        amg_max_iterations (int): Maximum number of iterations for the AMG-preconditioned solver
            used in the application of the covariance operator and its factorization. Defaults
            to `1000`, for the same reasons as `cg_max_iterations`.
    """

    mesh: dlx.mesh.Mesh
    mean_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    kappa: Annotated[Real, Is[lambda x: x > 0]]
    tau: Annotated[Real, Is[lambda x: x > 0]]
    robin_const: Annotated[Real, Is[lambda x: x >= 0]] | None = None
    seed: int = 0
    fe_data: tuple[
        Annotated[str, Is[lambda family: family in ("Lagrange", "P")]],
        Annotated[int, Is[lambda x: x > 0]],
    ] = ("Lagrange", 1)
    cg_relative_tolerance: Annotated[Real, Is[lambda x: x > 0]] = 1e-12
    cg_absolute_tolerance: Annotated[Real, Is[lambda x: x > 0]] | None = None
    cg_max_iterations: Annotated[int, Is[lambda x: x > 0]] = 1000
    amg_relative_tolerance: Annotated[Real, Is[lambda x: x > 0]] = 1e-12
    amg_absolute_tolerance: Annotated[Real, Is[lambda x: x > 0]] | None = None
    amg_max_iterations: Annotated[int, Is[lambda x: x > 0]] = 1000


# ==================================================================================================
class SPDEComponentStrategy(ABC):
    r"""ABC interface for the composition of components into an SPDE-based prior's operators.

    An SPDE-based prior is fully characterized, at the component level, by the composition of a
    mass matrix $M$, SPDE matrix $A$, block-diagonal matrix $\widehat{M}_e$ and transposed DoF map
    matrix $L^T$ (and their inverses) into three operators: the precision operator, the covariance
    operator, and the sampling factor. Which composition realizes these operators depends on the
    order of the underlying SPDE (e.g. bilaplacian vs. a first-order formulation); this interface
    isolates that choice so that
    [`SPDEPriorBuilder`][ls_bayesian.spde_prior.builder.SPDEPriorBuilder] itself stays agnostic to
    it. All other steps of the build pipeline (FEM assembly, interface wrapping) are independent of
    this choice and remain fixed in the builder.

    Methods:
        build_components: Build precision operator, covariance operator and sampling factor
            components from FEM matrices and solver settings.
    """

    @abstractmethod
    def build_components(
        self,
        mass_matrix: PETSc.Mat,
        spde_matrix: PETSc.Mat,
        block_diagonal_matrix: PETSc.Mat,
        dof_map_matrix: PETSc.Mat,
        cg_solver_settings: components.InverseMatrixSolverSettings,
        amg_solver_settings: components.InverseMatrixSolverSettings,
        logger: BaseLogger | None = None,
    ) -> tuple[components.PETScComponent, components.PETScComponent, components.PETScComponent]:
        r"""Build PETSc components for the prior's operators from FEM matrices.

        Args:
            mass_matrix (PETSc.Mat): Mass matrix $M$.
            spde_matrix (PETSc.Mat): SPDE matrix $A$.
            block_diagonal_matrix (PETSc.Mat): Block-diagonal matrix $\widehat{M}_e$.
            dof_map_matrix (PETSc.Mat): Transposed DoF map matrix $L^T$.
            cg_solver_settings (components.InverseMatrixSolverSettings): Solver settings for the
                CG-preconditioned inverse of the mass matrix.
            amg_solver_settings (components.InverseMatrixSolverSettings): Solver settings for the
                AMG-preconditioned inverse of the SPDE matrix.
            logger (BaseLogger | None, optional): Logger passed on to the
                [`InverseMatrixSolver`][ls_bayesian.spde_prior.components.InverseMatrixSolver]
                components for convergence diagnostics. Nothing is logged if `None`. Defaults to
                `None`.

        Returns:
            tuple[components.PETScComponent, components.PETScComponent, components.PETScComponent]:
                Precision operator, covariance operator, sampling factor.
        """


# ==================================================================================================
class SPDEPriorBuilder:
    r"""Builder for an SPDE-based prior.

    Builder class for a prior given as the solution of an elliptic SPDE, assembled from FEM
    matrices via a [`SPDEComponentStrategy`][ls_bayesian.spde_prior.builder.SPDEComponentStrategy].
    The strategy determines the composition of the mass matrix $M$ and SPDE matrix $A$ (and their
    inverses) into the prior's precision operator, covariance operator and sampling factor; the
    builder itself only handles the strategy-independent steps, i.e. FEM assembly and interface
    wrapping.

    Methods:
        build: Build the SPDE-based prior.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        settings: SPDEPriorSettings,
        component_strategy: SPDEComponentStrategy,
        logger: BaseLogger | None = None,
    ) -> None:
        """Initialize the builder with the given settings and component strategy.

        Args:
            settings (SPDEPriorSettings): Settings for the SPDE-based prior.
            component_strategy (SPDEComponentStrategy): Strategy for composing FEM matrices into
                the prior's precision operator, covariance operator and sampling factor.
            logger (BaseLogger | None, optional): Logger passed on to the built
                [`SPDEPrior`][ls_bayesian.spde_prior.spde_prior.SPDEPrior] and to the
                [`InverseMatrixSolver`][ls_bayesian.spde_prior.components.InverseMatrixSolver]
                components assembled by the strategy. Nothing is logged if `None`. The caller owns
                the logger's lifetime; the builder never constructs its own. Defaults to `None`.
        """
        self._component_strategy = component_strategy
        self._logger = logger
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
    def build(self) -> spde_prior.SPDEPrior:
        """Build the SPDE-based prior.

        Internally, this method assembles all dolfinx structures, delegates the composition of a
        hierarchy of [`PETScComponents`][ls_bayesian.spde_prior.components.PETScComponent] to the
        injected [`SPDEComponentStrategy`][ls_bayesian.spde_prior.builder.SPDEComponentStrategy],
        wraps them in [`InterfaceComponents`][ls_bayesian.spde_prior.components.InterfaceComponent]
        and hands them to the [`SPDEPrior`][ls_bayesian.spde_prior.spde_prior.SPDEPrior] class.

        Returns:
            spde_prior.SPDEPrior: The constructed SPDE-based prior.
        """
        mass_matrix, spde_matrix, block_diagonal_matrix, dof_map_matrix, converter = (
            self._build_fem_structures()
        )
        precision_operator, covariance_operator, sampling_factor = (
            self._component_strategy.build_components(
                mass_matrix,
                spde_matrix,
                block_diagonal_matrix,
                dof_map_matrix,
                self._cg_solver_settings,
                self._amg_solver_settings,
                logger=self._logger,
            )
        )
        precision_operator_interface, covariance_operator_interface, sampling_factor_interface = (
            self._build_interfaces(
                precision_operator, covariance_operator, sampling_factor, converter
            )
        )
        spde_based_prior = spde_prior.SPDEPrior(
            self._mean_vector,
            precision_operator_interface,
            covariance_operator_interface,
            sampling_factor_interface,
            converter,
            seed=self._seed,
            logger=self._logger,
        )
        return spde_based_prior

    # ----------------------------------------------------------------------------------------------
    def _build_fem_structures(
        self,
    ) -> tuple[PETSc.Mat, PETSc.Mat, PETSc.Mat, PETSc.Mat, fem.FEMConverter]:
        r"""Assemble FEM data structures, i.e. FEM Matrices and `FEMConverter` object.

        Returns:
            tuple[PETSc.Mat, PETSc.Mat, PETSc.Mat, PETSc.Mat, fem.FEMConverter]:
                mass matrix $M$, SPDE matrix $A$, block-diagonal matrix $\widehat{M}_e$,
                transposed DoF map matrix $L^T$,
                [`FEMConverter`][ls_bayesian.spde_prior.fem.FEMConverter] object.
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
    def _build_interfaces(
        self,
        precision_operator: components.PETScComponent,
        covariance_operator: components.PETScComponent,
        sampling_factor: components.PETScComponent,
        converter: fem.FEMConverter,
    ) -> tuple[
        components.InterfaceComponent, components.InterfaceComponent, components.InterfaceComponent
    ]:
        r"""Wrap PETSc components in interface components.

        Args:
            precision_operator (components.PETScComponent): Precision operator $\mathcal{C}^{-1}$.
            covariance_operator (components.PETScComponent): Covariance operator $\mathcal{C}$.
            sampling_factor (components.PETScComponent): Sampling factor $\widehat{\mathcal{C}}$.
            converter (fem.FEMConverter): Converter providing the process-independent order of the
                sampling factor input, i.e. the cell-local DoFs.

        Returns:
            tuple[components.InterfaceComponent,
                  components.InterfaceComponent,
                  components.InterfaceComponent]: Wrapped interface components.
        """
        precision_operator_interface = components.InterfaceComponent(precision_operator)
        covariance_operator_interface = components.InterfaceComponent(covariance_operator)
        sampling_factor_interface = components.InterfaceComponent(
            sampling_factor, input_indices=converter.cell_block_indices
        )
        return (
            precision_operator_interface,
            covariance_operator_interface,
            sampling_factor_interface,
        )
