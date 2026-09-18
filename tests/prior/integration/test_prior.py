import numpy as np
import pytest

import tests.prior.conftest as main_config
from ls_bayesian.prior import prior

pytestmark = pytest.mark.unit


# ==================================================================================================
def test_prior_cost(
    parametrized_bilaplacian_component_setup: main_config.PriorComponentSetup,
) -> None:
    mean_array = parametrized_bilaplacian_component_setup.mean_vector
    precision_array = parametrized_bilaplacian_component_setup.precision_array
    fem_converter = parametrized_bilaplacian_component_setup.fem_converter
    prior_object = prior.Prior(
        mean_array,
        parametrized_bilaplacian_component_setup.precision_interface,
        parametrized_bilaplacian_component_setup.covariance_interface,
        parametrized_bilaplacian_component_setup.sampling_factor_interface,
        fem_converter,
        seed=0,
    )

    rng = np.random.default_rng(1)
    state_array = rng.random(mean_array.shape)
    cost = prior_object.evaluate_cost(state_array)

    diff = state_array - mean_array
    projected_diff = fem_converter.convert_vertex_values_to_dofs(diff)
    expected_cost = 0.5 * projected_diff @ (precision_array @ projected_diff)
    assert np.isclose(cost, expected_cost)


# --------------------------------------------------------------------------------------------------
def test_prior_grad(
    parametrized_bilaplacian_component_setup: main_config.PriorComponentSetup,
) -> None:
    mean_array = parametrized_bilaplacian_component_setup.mean_vector
    precision_array = parametrized_bilaplacian_component_setup.precision_array
    fem_converter = parametrized_bilaplacian_component_setup.fem_converter
    prior_object = prior.Prior(
        mean_array,
        parametrized_bilaplacian_component_setup.precision_interface,
        parametrized_bilaplacian_component_setup.covariance_interface,
        parametrized_bilaplacian_component_setup.sampling_factor_interface,
        fem_converter,
        seed=0,
    )

    rng = np.random.default_rng(1)
    state_array = rng.random(mean_array.shape)
    grad = prior_object.evaluate_gradient(state_array)

    diff = state_array - mean_array
    projected_diff = fem_converter.convert_vertex_values_to_dofs(diff)
    expected_grad_dof = precision_array @ projected_diff
    expected_grad = fem_converter.convert_dofs_to_vertex_values(expected_grad_dof)
    assert np.allclose(grad, expected_grad)


# --------------------------------------------------------------------------------------------------
def test_prior_hvp(
    parametrized_bilaplacian_component_setup: main_config.PriorComponentSetup,
) -> None:
    mean_array = parametrized_bilaplacian_component_setup.mean_vector
    precision_array = parametrized_bilaplacian_component_setup.precision_array
    fem_converter = parametrized_bilaplacian_component_setup.fem_converter
    prior_object = prior.Prior(
        mean_array,
        parametrized_bilaplacian_component_setup.precision_interface,
        parametrized_bilaplacian_component_setup.covariance_interface,
        parametrized_bilaplacian_component_setup.sampling_factor_interface,
        fem_converter,
        seed=0,
    )

    rng = np.random.default_rng(1)
    direction_array = rng.random(mean_array.shape)
    hvp = prior_object.evaluate_hessian_vector_product(direction_array)
    direction_dof = fem_converter.convert_vertex_values_to_dofs(direction_array)
    expected_hvp_dof = precision_array @ direction_dof
    expected_hvp = fem_converter.convert_dofs_to_vertex_values(expected_hvp_dof)
    assert np.allclose(hvp, expected_hvp)


# --------------------------------------------------------------------------------------------------
def test_prior_sample(
    parametrized_bilaplacian_component_setup: main_config.PriorComponentSetup,
) -> None:
    mean_array = parametrized_bilaplacian_component_setup.mean_vector
    sampling_factor_array = parametrized_bilaplacian_component_setup.sampling_factor_array
    fem_converter = parametrized_bilaplacian_component_setup.fem_converter
    prior_object = prior.Prior(
        mean_array,
        parametrized_bilaplacian_component_setup.precision_interface,
        parametrized_bilaplacian_component_setup.covariance_interface,
        parametrized_bilaplacian_component_setup.sampling_factor_interface,
        fem_converter,
        seed=0,
    )

    sample = prior_object.generate_sample()

    rng = np.random.default_rng(0)
    random_vector = rng.normal(loc=0.0, scale=1.0, size=(sampling_factor_array.shape[1],))
    sample_dof = sampling_factor_array @ random_vector
    expected_sample = fem_converter.convert_dofs_to_vertex_values(sample_dof) + mean_array
    assert np.allclose(sample, expected_sample)
