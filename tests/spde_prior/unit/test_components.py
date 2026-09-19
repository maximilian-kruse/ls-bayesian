import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation
from petsc4py import PETSc

from ls_bayesian.spde_prior import components
from tests.spde_prior import helpers

pytestmark = pytest.mark.unit

# Dimension of the random SPD test matrices, and size of the Laplacian that cannot be solved by CG
# in two iterations.
SPD_MATRIX_DIM = 6
LAPLACIAN_DIM = 50


# ==================================================================================================
def solver_settings(
    preconditioner_type: str, **overrides: float | int
) -> components.InverseMatrixSolverSettings:
    return components.InverseMatrixSolverSettings(
        solver_type=PETSc.KSP.Type.CG,
        preconditioner_type=preconditioner_type,
        relative_tolerance=overrides.pop("relative_tolerance", helpers.SOLVER_RELATIVE_TOLERANCE),
        **overrides,
    )


def apply_component(component: components.PETScComponent, input_array: np.ndarray) -> np.ndarray:
    output_vector = component.create_output_vector()
    component.apply(helpers.petsc_vector_from_array(input_array), output_vector)
    return output_vector.getArray().copy()


def laplacian_1d(dim: int) -> np.ndarray:
    return 2 * np.eye(dim) - np.eye(dim, k=1) - np.eye(dim, k=-1)


# ==================================================================================================
def test_matrix_apply_equals_dense_product() -> None:
    rng = np.random.default_rng(0)
    dense_matrix = rng.random((3, 2))
    input_array = rng.random(2)
    matrix_component = components.Matrix(helpers.petsc_matrix_from_dense(dense_matrix))

    output_array = apply_component(matrix_component, input_array)

    helpers.assert_allclose_normwise(
        output_array, dense_matrix @ input_array, helpers.ROUNDOFF_TOLERANCE
    )
    assert matrix_component.shape == (3, 2)
    assert matrix_component.create_input_vector().getSize() == 2
    assert matrix_component.create_output_vector().getSize() == 3


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "preconditioner_type", [PETSc.PC.Type.JACOBI, PETSc.PC.Type.GAMG], ids=["jacobi", "gamg"]
)
def test_inverse_matrix_solver_apply_solves_system(preconditioner_type: str) -> None:
    rng = np.random.default_rng(0)
    spd_matrix = helpers.random_spd_matrix(rng, SPD_MATRIX_DIM)
    helpers.assert_condition_number_bounded(spd_matrix)
    right_hand_side = rng.random(SPD_MATRIX_DIM)
    solver_component = components.InverseMatrixSolver(
        solver_settings(preconditioner_type), helpers.petsc_matrix_from_dense(spd_matrix)
    )

    solution = apply_component(solver_component, right_hand_side)

    helpers.assert_allclose_normwise(
        solution, np.linalg.solve(spd_matrix, right_hand_side), helpers.KRYLOV_TOLERANCE
    )


# --------------------------------------------------------------------------------------------------
def test_inverse_matrix_solver_raises_on_non_convergence() -> None:
    """CG needs many iterations for a 1D Laplacian, so a limit of two iterations cannot converge."""
    laplacian_matrix = helpers.petsc_matrix_from_dense(laplacian_1d(LAPLACIAN_DIM))
    settings = solver_settings(PETSc.PC.Type.JACOBI, relative_tolerance=1e-12, max_num_iterations=2)
    solver_component = components.InverseMatrixSolver(settings, laplacian_matrix)

    with pytest.raises(RuntimeError, match="DIVERGED_MAX_IT"):
        apply_component(solver_component, np.ones(LAPLACIAN_DIM))


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "invalid_setting",
    [{"relative_tolerance": -1.0}, {"absolute_tolerance": 0.0}, {"max_num_iterations": 0}],
    ids=["relative_tolerance", "absolute_tolerance", "max_num_iterations"],
)
def test_inverse_matrix_solver_settings_reject_non_positive_values(invalid_setting: dict) -> None:
    with pytest.raises(BeartypeCallHintViolation):
        components.InverseMatrixSolverSettings(
            PETSc.KSP.Type.CG, PETSc.PC.Type.JACOBI, **invalid_setting
        )


# ==================================================================================================
@pytest.mark.parametrize("num_components", [2, 3])
def test_composition_applies_components_in_given_order(num_components: int) -> None:
    """Non-symmetric rectangular factors of distinct shapes expose any change of order."""
    rng = np.random.default_rng(0)
    factor_shapes = [(2, 3), (4, 2), (5, 4)][:num_components]
    dense_factors = [rng.random(shape) for shape in factor_shapes]
    composition = components.PETScComponentComposition(
        *(components.Matrix(helpers.petsc_matrix_from_dense(factor)) for factor in dense_factors)
    )
    input_array = rng.random(3)

    output_array = apply_component(composition, input_array)

    expected = input_array
    for factor in dense_factors:
        expected = factor @ expected
    helpers.assert_allclose_normwise(output_array, expected, helpers.ROUNDOFF_TOLERANCE)
    assert composition.shape == (factor_shapes[-1][0], 3)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("factor_shapes", "message"),
    [([(2, 3)], "At least two components"), ([(2, 3), (2, 3)], "Component 0 output dimension")],
    ids=["single_component", "mismatched_dimensions"],
)
def test_composition_rejects_invalid_components(
    factor_shapes: list[tuple[int, int]], message: str
) -> None:
    factors = [
        components.Matrix(helpers.petsc_matrix_from_dense(np.ones(s))) for s in factor_shapes
    ]

    with pytest.raises(ValueError, match=message):
        components.PETScComponentComposition(*factors)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("component_type", ["matrix", "inverse_matrix_solver", "composition"])
@pytest.mark.parametrize(
    ("wrong_size_vector", "message"),
    [("input", "Input vector size"), ("output", "Output vector size")],
    ids=["input", "output"],
)
def test_component_apply_rejects_wrong_vector_size(
    component_type: str, wrong_size_vector: str, message: str
) -> None:
    rng = np.random.default_rng(0)
    petsc_matrix = helpers.petsc_matrix_from_dense(helpers.random_spd_matrix(rng, SPD_MATRIX_DIM))
    matrix_component = components.Matrix(petsc_matrix)
    component = {
        "matrix": matrix_component,
        "inverse_matrix_solver": components.InverseMatrixSolver(
            solver_settings(PETSc.PC.Type.JACOBI), petsc_matrix
        ),
        "composition": components.PETScComponentComposition(matrix_component, matrix_component),
    }[component_type]
    wrong_size_vector_array = helpers.petsc_vector_from_array(np.ones(SPD_MATRIX_DIM + 1))
    input_vector = (
        wrong_size_vector_array if wrong_size_vector == "input" else component.create_input_vector()
    )
    output_vector = (
        wrong_size_vector_array
        if wrong_size_vector == "output"
        else component.create_output_vector()
    )

    with pytest.raises(ValueError, match=message):
        component.apply(input_vector, output_vector)


# ==================================================================================================
def test_interface_component_apply_equals_dense_product() -> None:
    rng = np.random.default_rng(0)
    dense_matrix = rng.random((3, 4))
    input_array = rng.random(4)
    interface_component = helpers.dense_interface_component(dense_matrix)

    output_array = interface_component.apply(input_array)

    helpers.assert_allclose_normwise(
        output_array, dense_matrix @ input_array, helpers.ROUNDOFF_TOLERANCE
    )


# --------------------------------------------------------------------------------------------------
def test_interface_component_selects_input_indices() -> None:
    rng = np.random.default_rng(0)
    dense_matrix = rng.random((3, 4))
    input_indices = np.array([2, 0, 3, 1])
    complete_input = rng.random(4)
    interface_component = components.InterfaceComponent(
        components.Matrix(helpers.petsc_matrix_from_dense(dense_matrix)), input_indices
    )

    output_array = interface_component.apply(complete_input)

    helpers.assert_allclose_normwise(
        output_array, dense_matrix @ complete_input[input_indices], helpers.ROUNDOFF_TOLERANCE
    )


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("with_input_indices", [False, True], ids=["owned_input", "full_input"])
def test_interface_component_rejects_wrong_input_shape(with_input_indices: bool) -> None:
    input_indices = np.arange(4) if with_input_indices else None
    interface_component = components.InterfaceComponent(
        components.Matrix(helpers.petsc_matrix_from_dense(np.ones((3, 4)))), input_indices
    )

    with pytest.raises(ValueError, match="Input vector shape"):
        interface_component.apply(np.ones(5))


# --------------------------------------------------------------------------------------------------
def test_interface_component_rejects_wrong_index_shape() -> None:
    matrix_component = components.Matrix(helpers.petsc_matrix_from_dense(np.ones((3, 4))))

    with pytest.raises(ValueError, match="Input indices shape"):
        components.InterfaceComponent(matrix_component, np.arange(3))


# --------------------------------------------------------------------------------------------------
def test_interface_component_returns_independent_copy() -> None:
    rng = np.random.default_rng(0)
    interface_component = helpers.dense_interface_component(rng.random((3, 4)))
    first_output = interface_component.apply(rng.random(4))
    first_output_copy = first_output.copy()

    interface_component.apply(rng.random(4))

    np.testing.assert_array_equal(first_output, first_output_copy)
