"""Building blocks for prior components.

Classes:
    PETScComponent: ABC interface for PETSc-based components.
    PETScComponentComposition: Composition of variable number of PETSc components.
    InterfaceComponent: Wrapper for PETSc component for numpy-based interface.
    Matrix: Simple PETSc matrix wrapper.
    InverseMatrixSolverSettings: Settings for the inverse matrix representation via Krylov
        subspace solver.
    InverseMatrixSolver: Inverse matrix representation via Krylov subspace solver.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import Annotated

import numpy as np
from beartype.vale import Is
from petsc4py import PETSc


# ==================================================================================================
class PETScComponent(ABC):
    """ABC Interface for PETSc components.

    This class describes the common interface for all PETSc-based components in this package. The
    idea is that any such component mimics the interface of a PETSc Matrix (with a more pythonic
    interface). Under the hood, this could be the composition of several matrices, or the
    representation of a matrix inverse through a matrix solver. This allows to easily combine
    different components in a flexible manner.
    The main functionality of a PETSc component is its `apply` method, which resembles a
    matrix-vector product. Furthermore, the interface provides methods to create input and output
    PETSc vectors that match the dimensionality of the component (when thinking of it as a matrix).

    Methods:
        apply: Apply component to input vector, store result in output vector.
        create_input_vector: Create PETSc vector that could be applied to component from the right.
        create_output_vector: Create PETSc vector that could be applied to component from the left.

    Attributes:
        shape (tuple[int, int]): Shape of the component (thought of as a matrix).
    """

    @abstractmethod
    def apply(self, input_vector: PETSc.Vec, output_vector: PETSc.Vec) -> None:
        """Apply component to input vector, store result in output vector.

        Args:
            input_vector (PETSc.Vec): Vector to apply to component from the right.
            output_vector (PETSc.Vec): Vector to store result of component application in.
        """

    @abstractmethod
    def create_input_vector(self) -> PETSc.Vec:
        """Create PETSc vector that could be applied to component from the right.

        Returns:
            PETSc.Vec: PETSc vector of correct size.
        """

    @abstractmethod
    def create_output_vector(self) -> PETSc.Vec:
        """Create PETSc vector that could be applied to component from the left.

        Returns:
            PETSc.Vec: PETSc vector of correct size.
        """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """Return the shape of the component (as a matrix).

        Returns:
            tuple[int, int]: Shape of the component, as a matrix this would be (num_rows, num_cols).
        """

    def _check_dimensions(self, input_vector: PETSc.Vec, output_vector: PETSc.Vec) -> None:
        """Checks correct dimensionalities in `apply` method, should be utilized in subclasses."""
        if not input_vector.getSize() == self.shape[1]:
            raise ValueError(
                f"Input vector size {input_vector.getSize()} does not match "
                f"expected size {self.shape[1]}."
            )
        if not output_vector.getSize() == self.shape[0]:
            raise ValueError(
                f"Output vector size {output_vector.getSize()} does not match "
                f"expected size {self.shape[0]}."
            )


# ==================================================================================================
class PETScComponentComposition(PETScComponent):
    r"""Composition of variable number of PETSc components.

    The operators we require for the prior, namely covariance, precision and sampling factor,
    are compositions of several simpler, matrix-like components. This class provides the
    functionality to combine an arbitrary number of PETSc components into a new, composite
    component. The idea is very simple: Given a list of components $[M_1, M_2, ..., M_n]$, which
    can be applied to a PETSc vector, the composition will apply them in the given sequence.

    Methods:
        apply: Apply composition of components to input, store result in output vector.
        create_input_vector: Create PETSc vector that could be multiplied with composition from the
            right.
        create_output_vector: Create PETSc vector that could be multiplied with composition from the
            left.

    Attributes:
        shape: Shape of the component (thought of as a matrix).
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, *components: PETScComponent) -> None:
        """Initialize component as the composition of several PETSc components.

        Args:
            *components (PETScComponent): Components to compose, applied in the given order.

        Raises:
            ValueError: If less than two components are given, or if the dimensions of the
                components do not match for composition.
        """
        self._components = components
        self._num_components = len(components)

        if self._num_components < 2:
            raise ValueError("At least two components are required for a composition.")
        for i in range(self._num_components - 1):
            if not components[i].shape[0] == components[i + 1].shape[1]:
                raise ValueError(
                    f"Component {i} output dimension {components[i].shape[0]} does not match "
                    f"component {i + 1} input dimension {components[i + 1].shape[1]}."
                )

        self._temp_buffers = []
        for i in range(self._num_components - 1):
            self._temp_buffers.append(components[i].create_output_vector())

    # ----------------------------------------------------------------------------------------------
    def apply(self, input_vector: PETSc.Vec, output_vector: PETSc.Vec) -> None:
        """Apply the composition of components to input, store result in output vector.

        Args:
            input_vector (PETSc.Vec): Vector to apply composition to.
            output_vector (PETSc.Vec): Vector to store result in.
        """
        self._check_dimensions(input_vector, output_vector)

        current_input = input_vector
        for component, buffer in zip(self._components[:-1], self._temp_buffers, strict=True):
            component.apply(current_input, buffer)
            current_input = buffer
        self._components[-1].apply(current_input, output_vector)

    # ----------------------------------------------------------------------------------------------
    def create_input_vector(self) -> PETSc.Vec:
        """Create PETSc vector that could be multiplied with composition from the right.

        Returns:
            PETSc.Vec: PETSc vector of correct size.
        """
        return self._components[0].create_input_vector()

    # ----------------------------------------------------------------------------------------------
    def create_output_vector(self) -> PETSc.Vec:
        """Create PETSc vector that could be multiplied with composition from the left.

        Returns:
            PETSc.Vec: PETSc vector of correct size.
        """
        return self._components[-1].create_output_vector()

    # ----------------------------------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the component, i.e. number of rows and columns of the composition.

        Returns:
            tuple[int, int]: (num_rows, num_cols)
        """
        return self._components[-1].shape[0], self._components[0].shape[1]


# ==================================================================================================
class InterfaceComponent:
    """Wrapper for PETSc component for numpy-based interface.

    The prior [`PETScComponents`][ls_prior.components.PETScComponent] handle matrix representations
    and vectors in standard PETSc format. To allow for a more Pythonic interface, this class wraps
    a given PETSc component and provides methods to apply numpy arrays to the component, returning
    numpy arrays. These arrays are also copied to avoid modification of the original underlying data
    structures, which are handled as references only.

    Methods:
        apply: Apply input array to component, return result as numpy array.

    Attributes:
        shape: Shape of the component (thought of as a matrix).
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        component: PETScComponent,
    ) -> None:
        """Initialize interface component.

        Args:
            component (PETScComponent): PETSc component to wrap.
        """
        self._component = component
        self._input_buffer = component.create_input_vector()
        self._output_buffer = component.create_output_vector()

    # ----------------------------------------------------------------------------------------------
    def apply(
        self, input_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Apply input array to component, also return result as numpy array.

        Args:
            input_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Input vector
                to apply to component.

        Raises:
            ValueError: If input vector does not have correct shape.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Result of the application to the
                underlying component. Will be copied from internal buffer to avoid modification of
                internal state.
        """
        if not input_vector.shape == (self.shape[1],):
            raise ValueError(
                f"Input vector shape {input_vector.shape} does not match expected shape "
                f"({self.shape[1]},)."
            )
        self._input_buffer.setArray(input_vector)
        self._input_buffer.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self._component.apply(self._input_buffer, self._output_buffer)
        output_vector = self._output_buffer.getArray()

        return output_vector.copy()

    # ----------------------------------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the component.

        Returns:
            tuple[int, int]: Input and output dimension, as a matrix this would be
                (num_rows, num_cols).
        """
        return self._component.shape


# ==================================================================================================
class Matrix(PETScComponent):
    """Simple PETSc matrix wrapper.

    This class does nothing more than wrapping a PETSc matrix, to match the more Pythonic interface
    defined in the PETScComponent ABC. It is purely for convenience.

    Methods:
        apply: Perform matrix-vector multiplication with input, store result in output vector.
        create_input_vector: Create PETSc vector that could be multiplied with matrix from the
            right.
        create_output_vector: Create PETSc vector that could be multiplied with matrix from the
            left.

    Attributes:
        shape: Shape of the PETSc matrix (n_rows, n_cols).
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, petsc_matrix: PETSc.Mat) -> None:
        """Initialize component, simply store PETSc matrix internally.

        Args:
            petsc_matrix (PETSc.Mat): PETSc matrix to wrap, should be sparse.
        """
        self._petsc_matrix = petsc_matrix

    # ----------------------------------------------------------------------------------------------
    def apply(self, input_vector: PETSc.Vec, output_vector: PETSc.Vec) -> None:
        """Perform matrix-vector multiplication with input, store result in output vector.

        Args:
            input_vector (PETSc.Vec): Vector to perform Matrix-vector multiplication with.
            output_vector (PETSc.Vec): Vector to store the result of the multiplication.
        """
        self._petsc_matrix.mult(input_vector, output_vector)

    # ----------------------------------------------------------------------------------------------
    def create_input_vector(self) -> PETSc.Vec:
        """Create PETSc vector that could be multiplied with matrix from the right.

        Returns:
            PETSc.Vec: PETSc vector of correct size.
        """
        return self._petsc_matrix.createVecRight()

    # ----------------------------------------------------------------------------------------------
    def create_output_vector(self) -> PETSc.Vec:
        """Create PETSc vector that could be multiplied with matrix from the left.

        Returns:
            PETSc.Vec: PETSc vector of correct size.
        """
        return self._petsc_matrix.createVecLeft()

    # ----------------------------------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the component, i.e. number of rows and columns of the matrix.

        Returns:
            tuple[int, int]: (num_rows, num_cols)
        """
        return self._petsc_matrix.getSize()


# ==================================================================================================
@dataclass
class InverseMatrixSolverSettings:
    r"""Settings for the inverse matrix representation via Krylov subspace solver.

    All combinations of solvers and preconditioners provided in PETSc are supported. It is the
    user's responsibility to choose a suitable combination.

    Attributes:
        solver_type: Type of the PETSc [KSP](https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.KSP.Type.html#petsc4py.PETSc.KSP.Type)
            solver.
        preconditioner_type: Type of the PETSc [preconditioner](https://petsc.org/main/petsc4py/reference/petsc4py.PETSc.PC.Type.html).
        relative_tolerance: Relative tolerance for the solver.
        absolute_tolerance: Absolute tolerance for the solver.
        max_num_iterations: Maximum number of iterations to perform.
    """

    solver_type: str
    preconditioner_type: str
    relative_tolerance: Annotated[Real, Is[lambda x: x > 0]] | None = None
    absolute_tolerance: Annotated[Real, Is[lambda x: x > 0]] | None = None
    max_num_iterations: Annotated[int, Is[lambda x: x > 0]] | None = None


class InverseMatrixSolver(PETScComponent):
    """Inverse matrix representation via Krylov subspace solver.

    This class resembles the inverse of a given PETSc matrix, through an iterative solver. This
    allows for matrix-free representation and efficient application of the inverse to PETSc vectors.

    Methods:
        apply: Apply inverse matrix to input, store result in output vector.
        create_input_vector: Create PETSc vector that could be multiplied with inverse matrix from
            the right.
        create_output_vector: Create PETSc vector that could be multiplied with inverse matrix from
            the left.

    Attributes:
        shape: Shape of the inverse matrix (n_rows, n_cols).
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, solver_settings: InverseMatrixSolverSettings, petsc_matrix: PETSc.Mat
    ) -> None:
        """Initialize solver with given system matrix and solver config.

        Args:
            solver_settings (InverseMatrixSolverSettings): Settings for the solver.
            petsc_matrix (PETSc.Mat): System matrix to solve with, should be sparse.
        """
        self._petsc_matrix = petsc_matrix
        self._solver = PETSc.KSP().create(petsc_matrix.comm)
        self._solver.setOperators(petsc_matrix)
        self._solver.setType(solver_settings.solver_type)
        preconditioner = self._solver.getPC()
        preconditioner.setType(solver_settings.preconditioner_type)
        self._solver.setInitialGuessNonzero(False)
        self._solver.setTolerances(
            rtol=solver_settings.relative_tolerance,
            atol=solver_settings.absolute_tolerance,
            max_it=solver_settings.max_num_iterations,
        )

    # ----------------------------------------------------------------------------------------------
    def apply(self, input_vector: PETSc.Vec, output_vector: PETSc.Vec) -> None:
        """Apply inverse matrix to input, store result in output vector.

        Args:
            input_vector (PETSc.Vec): Vector to apply inverse to.
            output_vector (PETSc.Vec): Vector to store result in.
        """
        self._solver.solve(input_vector, output_vector)

    # ----------------------------------------------------------------------------------------------
    def create_input_vector(self) -> PETSc.Vec:
        """Create PETSc vector that could be multiplied with inverse matrix from the right.

        Returns:
            PETSc.Vec: PETSc vector of correct size.
        """
        return self._petsc_matrix.createVecLeft()

    # ----------------------------------------------------------------------------------------------
    def create_output_vector(self) -> PETSc.Vec:
        """Create PETSc vector that could be multiplied with matrix from the left.

        Returns:
            PETSc.Vec: PETSc vector of correct size.
        """
        return self._petsc_matrix.createVecRight()

    # ----------------------------------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the component, i.e. number of rows and columns of the inverse matrix.

        Returns:
            tuple[int, int]: (num_rows, num_cols)
        """
        return self._petsc_matrix.getSize()
