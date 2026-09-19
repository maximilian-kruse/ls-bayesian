r"""Integration tests of the Bilaplacian prior assembled by the builder.

Operators are checked in vertex space against closed-form values, mutual consistency, and dense
counterparts computed from dolfinx-assembled matrices. For P1 spaces, the vertex-space operators
are similar to the DoF-space operators $P = A M^{-1} A$, $C = A^{-1} M A^{-1}$ and
$F = A^{-1} L^T \widehat{M}_e$, via the permutation from
[`vertex_to_dof_selection_matrix`][tests.spde_prior.helpers.vertex_to_dof_selection_matrix].
"""

import dolfinx as dlx
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation
from mpi4py import MPI

from ls_bayesian.spde_prior import builder
from tests.spde_prior import helpers

pytestmark = pytest.mark.integration

P1_CASES = pytest.mark.parametrize("fem_case", helpers.P1_CASE_IDS, indirect=True)
SINGLE_CASE = pytest.mark.parametrize("fem_case", ["1d_p1"], indirect=True)
NEUMANN_ONLY = pytest.mark.parametrize("robin_const", [None], ids=["neumann"], indirect=True)
CONSTANT_SHIFT = 1.7
# Solver settings that cannot converge: a single iteration towards a residual reduction of 1e-14.
NON_CONVERGING_MAX_ITERATIONS = 1
NON_CONVERGING_RELATIVE_TOLERANCE = 1e-14

# Matern variance test: kappa gives a correlation length sqrt(8 nu) / kappa ~ 0.14 for nu = 1, so
# that boundary effects at the domain center, 3.5 correlation lengths away, are negligible. The
# mesh resolutions resolve 1 / kappa with at least 3 cells.
MATERN_KAPPA = 20.0
MATERN_COARSE_RESOLUTION = 64
MATERN_FINE_RESOLUTION = 128


# ==================================================================================================
def dense_spde_matrix(
    assembled_matrices: helpers.AssembledMatrices, robin_const: float | None
) -> np.ndarray:
    if robin_const is None:
        return assembled_matrices.neumann_spde_matrix
    return assembled_matrices.robin_spde_matrix


def vertex_space_operator(
    fem_space_setup: helpers.FEMSpaceSetup, dof_operator: np.ndarray
) -> np.ndarray:
    selection_matrix = helpers.vertex_to_dof_selection_matrix(fem_space_setup.function_space)
    return selection_matrix.T @ dof_operator @ selection_matrix


def materialize_precision_and_covariance(
    setup: helpers.BuiltPriorSetup,
) -> tuple[np.ndarray, np.ndarray]:
    num_vertices = setup.mean_vector.size
    precision = helpers.materialize_operator(setup.prior.apply_precision_operator, num_vertices)
    covariance = helpers.materialize_operator(setup.prior.apply_covariance_operator, num_vertices)
    return precision, covariance


def matern_pointwise_variance(kappa: float, tau: float) -> float:
    r"""Marginal variance of $\tau(\kappa^2 - \Delta) m = \mathcal{W}$ in $\mathbb{R}^2$.

    For $\alpha = 2$ in $d = 2$, the smoothness is $\nu = \alpha - d/2 = 1$, and the variance
    $\Gamma(\nu) / (\Gamma(\alpha) (4\pi)^{d/2} \kappa^{2\nu} \tau^2)$ reduces to
    $1 / (4 \pi \kappa^2 \tau^2)$ (Lindgren, Rue, Lindström, JRSS-B 73(4), 2011).
    """
    return 1.0 / (4 * np.pi * kappa**2 * tau**2)


def center_vertex_variance(resolution: int) -> float:
    mesh = dlx.mesh.create_unit_square(MPI.COMM_SELF, resolution, resolution)
    num_vertices = mesh.geometry.x.shape[0]
    settings = builder.BilaplacianPriorSettings(
        mesh,
        np.zeros(num_vertices),
        kappa=MATERN_KAPPA,
        tau=helpers.TAU,
        robin_const=helpers.ROBIN_CONSTANT,
    )
    prior = builder.BilaplacianPriorBuilder(settings).build()
    vertex_coordinates = helpers.input_ordered_vertex_coordinates(mesh)
    center_index = np.argmin(np.linalg.norm(vertex_coordinates[:, :2] - 0.5, axis=1))
    return float(prior.apply_covariance_operator(np.eye(num_vertices)[center_index])[center_index])


# ==================================================================================================
@NEUMANN_ONLY
def test_bilaplacian_prior_cost_of_constant_shift_matches_closed_form(
    built_prior_setup: helpers.BuiltPriorSetup,
) -> None:
    r"""For Neumann conditions, $A 1 = \kappa^2 \tau M 1$.

    Hence $1^T A M^{-1} A 1 = \kappa^4 \tau^2 1^T M 1 = \kappa^4 \tau^2 |\Omega|$.
    """
    cost = built_prior_setup.prior.evaluate_cost(built_prior_setup.mean_vector + CONSTANT_SHIFT)

    domain_measure = built_prior_setup.fem_space_setup.fem_case.domain_measure
    expected_cost = 0.5 * CONSTANT_SHIFT**2 * helpers.KAPPA**4 * helpers.TAU**2 * domain_measure
    helpers.assert_allclose_normwise(cost, expected_cost, helpers.KRYLOV_TOLERANCE)


# --------------------------------------------------------------------------------------------------
@P1_CASES
def test_bilaplacian_precision_matches_dense_operator(
    built_prior_setup: helpers.BuiltPriorSetup, assembled_matrices: helpers.AssembledMatrices
) -> None:
    precision, _ = materialize_precision_and_covariance(built_prior_setup)

    spde_matrix = dense_spde_matrix(assembled_matrices, built_prior_setup.robin_const)
    expected_precision = vertex_space_operator(
        built_prior_setup.fem_space_setup,
        spde_matrix @ np.linalg.solve(assembled_matrices.mass_matrix, spde_matrix),
    )
    helpers.assert_allclose_normwise(precision, expected_precision, helpers.KRYLOV_TOLERANCE)


# --------------------------------------------------------------------------------------------------
@P1_CASES
def test_bilaplacian_covariance_inverts_precision(
    built_prior_setup: helpers.BuiltPriorSetup,
) -> None:
    precision, covariance = materialize_precision_and_covariance(built_prior_setup)
    helpers.assert_condition_number_bounded(precision)

    identity = np.eye(built_prior_setup.mean_vector.size)
    helpers.assert_allclose_normwise(covariance @ precision, identity, helpers.KRYLOV_TOLERANCE)


# --------------------------------------------------------------------------------------------------
@P1_CASES
def test_bilaplacian_factorization_reproduces_covariance(
    built_prior_setup: helpers.BuiltPriorSetup,
) -> None:
    prior = built_prior_setup.prior

    factorization = helpers.materialize_operator(
        prior.apply_covariance_factorization, prior.random_vector_size
    )
    _, covariance = materialize_precision_and_covariance(built_prior_setup)

    helpers.assert_allclose_normwise(
        factorization @ factorization.T, covariance, helpers.KRYLOV_TOLERANCE
    )


# --------------------------------------------------------------------------------------------------
@P1_CASES
def test_bilaplacian_precision_is_symmetric_positive_definite(
    built_prior_setup: helpers.BuiltPriorSetup,
) -> None:
    precision, _ = materialize_precision_and_covariance(built_prior_setup)

    helpers.assert_allclose_normwise(precision, precision.T, helpers.KRYLOV_TOLERANCE)
    assert np.linalg.eigvalsh(0.5 * (precision + precision.T)).min() > 0


# --------------------------------------------------------------------------------------------------
@SINGLE_CASE
@pytest.mark.parametrize("comparison", ["same_seed_identical", "different_seeds_differ"])
def test_bilaplacian_builder_forwards_seed(
    fem_space_setup: helpers.FEMSpaceSetup, comparison: str
) -> None:
    first_prior = helpers.build_bilaplacian_prior(fem_space_setup, None, seed=0).prior
    second_seed = 0 if comparison == "same_seed_identical" else 1
    second_prior = helpers.build_bilaplacian_prior(fem_space_setup, None, seed=second_seed).prior

    first_sample, second_sample = first_prior.generate_sample(), second_prior.generate_sample()

    if comparison == "same_seed_identical":
        np.testing.assert_array_equal(first_sample, second_sample)
    else:
        assert not np.allclose(first_sample, second_sample)


# --------------------------------------------------------------------------------------------------
@SINGLE_CASE
@NEUMANN_ONLY
def test_bilaplacian_whitened_samples_are_standard_normal(
    built_prior_setup: helpers.BuiltPriorSetup, assembled_matrices: helpers.AssembledMatrices
) -> None:
    prior = built_prior_setup.prior

    samples = np.stack([prior.generate_sample() for _ in range(helpers.NUM_SAMPLES)])

    spde_matrix = assembled_matrices.neumann_spde_matrix
    inverse_spde_matrix = np.linalg.inv(spde_matrix)
    covariance = vertex_space_operator(
        built_prior_setup.fem_space_setup,
        inverse_spde_matrix @ assembled_matrices.mass_matrix @ inverse_spde_matrix,
    )
    mean_z_score, variance_z_score = helpers.whitened_sample_z_scores(
        samples, built_prior_setup.mean_vector, covariance
    )
    assert abs(mean_z_score) < helpers.STATISTICAL_Z_THRESHOLD
    assert abs(variance_z_score) < helpers.STATISTICAL_Z_THRESHOLD


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fem_case", ["2d_p2"], indirect=True)
@pytest.mark.parametrize(
    ("solver_prefix", "operator_name"),
    [("cg", "apply_precision_operator"), ("amg", "apply_covariance_operator")],
    ids=["cg_precision", "amg_covariance"],
)
def test_bilaplacian_builder_forwards_solver_settings(
    fem_space_setup: helpers.FEMSpaceSetup, solver_prefix: str, operator_name: str
) -> None:
    setup = helpers.build_bilaplacian_prior(
        fem_space_setup,
        None,
        **{
            f"{solver_prefix}_max_iterations": NON_CONVERGING_MAX_ITERATIONS,
            f"{solver_prefix}_relative_tolerance": NON_CONVERGING_RELATIVE_TOLERANCE,
        },
    )

    with pytest.raises(RuntimeError, match="Krylov solver did not converge"):
        getattr(setup.prior, operator_name)(setup.mean_vector)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "invalid_setting",
    [
        {"kappa": 0.0},
        {"tau": -1.0},
        {"robin_const": -0.1},
        {"fe_data": ("Lagrange", 0)},
        {"fe_data": ("DG", 1)},
        {"cg_relative_tolerance": 0.0},
        {"cg_absolute_tolerance": -1.0},
        {"cg_max_iterations": 0},
        {"amg_relative_tolerance": -1.0},
        {"amg_absolute_tolerance": 0.0},
        {"amg_max_iterations": 0},
    ],
    ids=lambda setting: "-".join(f"{key}={value}" for key, value in setting.items()),
)
def test_bilaplacian_settings_reject_invalid_values(invalid_setting: dict) -> None:
    mesh = helpers.create_unit_interval_mesh(helpers.MESH_COMMUNICATOR)
    settings_arguments = {
        "mesh": mesh,
        "mean_vector": np.zeros(mesh.geometry.x.shape[0]),
        "kappa": helpers.KAPPA,
        "tau": helpers.TAU,
    }

    with pytest.raises(BeartypeCallHintViolation):
        builder.BilaplacianPriorSettings(**(settings_arguments | invalid_setting))


# --------------------------------------------------------------------------------------------------
def test_bilaplacian_builder_rejects_wrong_mean_length() -> None:
    mesh = helpers.create_unit_interval_mesh(helpers.MESH_COMMUNICATOR)
    settings = builder.BilaplacianPriorSettings(
        mesh, np.zeros(mesh.geometry.x.shape[0] + 1), kappa=helpers.KAPPA, tau=helpers.TAU
    )

    with pytest.raises(ValueError, match="Expected vertex_values to have shape"):
        builder.BilaplacianPriorBuilder(settings).build()


# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_bilaplacian_marginal_variance_matches_matern() -> None:
    """The discretization error of the fine mesh is bounded by the change under refinement.

    For a discretization error that at least halves under refinement, the error of the fine
    solution is at most the difference between the coarse and fine solutions.
    """
    coarse_variance = center_vertex_variance(MATERN_COARSE_RESOLUTION)
    fine_variance = center_vertex_variance(MATERN_FINE_RESOLUTION)

    refinement_difference = abs(fine_variance - coarse_variance)
    np.testing.assert_allclose(
        fine_variance,
        matern_pointwise_variance(MATERN_KAPPA, helpers.TAU),
        rtol=0,
        atol=refinement_difference,
    )
