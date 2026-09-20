r"""Concrete component composition strategies for SPDE-based priors.

Classes:
    BilaplacianComponentStrategy: Component strategy for a bilaplacian prior.
"""

from typing import override

from petsc4py import PETSc

from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.spde_prior import components
from ls_bayesian.spde_prior.builder import SPDEComponentStrategy


# ==================================================================================================
class BilaplacianComponentStrategy(SPDEComponentStrategy):
    r"""Component strategy for a bilaplacian prior.

    Composes the components for the distribution of the solution of the SPDE
    $\tau(\kappa^2 - \Delta) m = \mathcal{W}$ with white noise $\mathcal{W}$. Its covariance
    operator $\mathcal{C} = (\tau(\kappa^2 - \Delta))^{-2}$ is the inverse of a squared elliptic
    operator, discretized as $\mathcal{C} = A^{-1} M A^{-1}$ with mass matrix $M$ and SPDE matrix
    $A$.

    Methods:
        build_components: Build precision operator, covariance operator and sampling factor
            components from FEM matrices and solver settings.
    """

    # ----------------------------------------------------------------------------------------------
    @override
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
        r"""Build PETSc components for bilaplacian prior from FEM matrices.

        The main components for the prior are:

        1. Precision operator: $\mathcal{C}^{-1} = A M^{-1} A$
        2. Covariance operator: $\mathcal{C} = A^{-1} M A^{-1}$
        3. Sampling factor: $\widehat{\mathcal{C}} = A^{-1} L^T \widehat{M}_e$

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
            cg_solver_settings, mass_matrix, logger=logger
        )
        spde_matrix_inverse_component = components.InverseMatrixSolver(
            amg_solver_settings, spde_matrix, logger=logger
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
