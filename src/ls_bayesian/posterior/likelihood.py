r"""Concrete likelihood implementations.

Classes:
    VertexObservationSettings: Settings for point observations at mesh vertices.
    GaussianLogLikelihood: Negative log-likelihood for additive Gaussian noise and a linear
        observation operator.

Functions:
    assemble_vertex_observation_matrix: Assemble the observation operator for point observations
        at mesh vertices.
    assemble_diagonal_precision_matrix: Assemble a diagonal noise precision matrix.
"""

from dataclasses import dataclass
from typing import Self, override

import numpy as np
import scipy.sparse as sp

from ls_bayesian.posterior import interfaces


# ==================================================================================================
def assemble_vertex_observation_matrix(
    num_vertices: int, observed_vertex_indices: np.ndarray[tuple[int], np.dtype[np.integer]]
) -> sp.coo_matrix:
    r"""Assemble the observation operator $\mathcal{B}$ for point observations at mesh vertices.

    Row $i$ of $\mathcal{B}$ holds a single unit entry in column $j_i$, the index of the $i$-th
    observed vertex, so that $(\mathcal{B}u)_i = u_{j_i}$ for a solution $u$ given on the mesh
    vertices. Each vertex may be observed at most once.

    Args:
        num_vertices (int): Number of mesh vertices, i.e. the solution dimension.
        observed_vertex_indices (np.ndarray[tuple[int], np.dtype[np.integer]]): Indices $j_i$ of the
            observed vertices, shape `(num_observations,)`.

    Raises:
        ValueError: If the number of vertices is not positive.
        ValueError: If the indices are not one-dimensional.
        ValueError: If an index lies outside of `[0, num_vertices)`.
        ValueError: If an index occurs more than once.

    Returns:
        sp.coo_matrix: Observation operator, shape `(num_observations, num_vertices)`.
    """
    if num_vertices <= 0:
        raise ValueError(f"Number of vertices must be positive, but is {num_vertices}.")
    if observed_vertex_indices.ndim != 1:
        raise ValueError(
            "Observed vertex indices must be one-dimensional, "
            f"but have shape {observed_vertex_indices.shape}."
        )
    invalid_indices = observed_vertex_indices[
        (observed_vertex_indices < 0) | (observed_vertex_indices >= num_vertices)
    ]
    if invalid_indices.size > 0:
        raise ValueError(
            f"Observed vertex indices {invalid_indices} lie outside of [0, {num_vertices})."
        )
    unique_indices, index_counts = np.unique(observed_vertex_indices, return_counts=True)
    duplicate_indices = unique_indices[index_counts > 1]
    if duplicate_indices.size > 0:
        raise ValueError(
            f"Observed vertex indices must be unique, but {duplicate_indices} occur more than once."
        )
    num_observations = observed_vertex_indices.shape[0]
    row_inds = np.arange(num_observations, dtype=np.int64)
    data = np.ones(num_observations, dtype=np.float64)
    return sp.coo_matrix(
        (data, (row_inds, observed_vertex_indices)), shape=(num_observations, num_vertices)
    )


# --------------------------------------------------------------------------------------------------
def assemble_diagonal_precision_matrix(
    precision_values: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> sp.coo_matrix:
    r"""Assemble a diagonal noise precision matrix $\Gamma^{-1}$.

    This corresponds to independent Gaussian noise per observation, with
    $(\Gamma^{-1})_{ii} = 1 / \sigma_i^2$ for the noise standard deviations $\sigma_i$.

    Args:
        precision_values (np.ndarray[tuple[int], np.dtype[np.float64]]): Diagonal entries
            $1 / \sigma_i^2$, shape `(num_observations,)`.

    Raises:
        ValueError: If the precision values are not one-dimensional.
        ValueError: If a precision value is not strictly positive and finite.

    Returns:
        sp.coo_matrix: Diagonal precision matrix, shape `(num_observations, num_observations)`.
    """
    if precision_values.ndim != 1:
        raise ValueError(
            f"Precision values must be one-dimensional, but have shape {precision_values.shape}."
        )
    invalid_values = precision_values[~((precision_values > 0) & np.isfinite(precision_values))]
    if invalid_values.size > 0:
        raise ValueError(
            f"Precision values must be strictly positive and finite, but contain {invalid_values}."
        )
    num_observations = precision_values.shape[0]
    diagonal_inds = np.arange(num_observations, dtype=np.int64)
    return sp.coo_matrix(
        (precision_values, (diagonal_inds, diagonal_inds)),
        shape=(num_observations, num_observations),
    )


# ==================================================================================================
@dataclass(frozen=True)
class VertexObservationSettings:
    r"""Settings for point observations at mesh vertices.

    Collects the inputs required to assemble a
    [`GaussianLogLikelihood`][ls_bayesian.posterior.likelihood.GaussianLogLikelihood] via
    [`from_vertex_observations`][ls_bayesian.posterior.likelihood.GaussianLogLikelihood.from_vertex_observations].
    The field constraints are validated by the assembly functions and the constructor.

    Attributes:
        data_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Observed data $d$, shape
            `(num_observations,)`.
        num_vertices (int): Number of mesh vertices, i.e. the solution dimension.
        observed_vertex_indices (np.ndarray[tuple[int], np.dtype[np.integer]]): Indices of the
            observed vertices, shape `(num_observations,)`. Each vertex may be observed at most
            once.
        precision_values (np.ndarray[tuple[int], np.dtype[np.float64]]): Noise precisions
            $1 / \sigma_i^2$ per observation, shape `(num_observations,)`.
    """

    data_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    num_vertices: int
    observed_vertex_indices: np.ndarray[tuple[int], np.dtype[np.integer]]
    precision_values: np.ndarray[tuple[int], np.dtype[np.float64]]


# ==================================================================================================
class GaussianLogLikelihood(interfaces.Likelihood):
    r"""Negative log-likelihood for additive Gaussian noise and a linear observation operator.

    Observations are modelled as $d = \mathcal{B}u + \eta$ with a linear observation operator
    $\mathcal{B}$ and noise $\eta \sim \mathcal{N}(0, \Gamma)$. Up to an additive constant, the
    negative log-likelihood is the quadratic misfit

    $$
    \begin{equation*}
        \Phi(u) = \frac{1}{2} (\mathcal{B}u - d)^T \Gamma^{-1} (\mathcal{B}u - d).
    \end{equation*}
    $$

    As $\Phi$ is quadratic in $u$, its Hessian $\mathcal{B}^T \Gamma^{-1} \mathcal{B}$ does not
    depend on the solution. For point observations at mesh vertices with independent noise, the
    likelihood is created with the factory
    [`from_vertex_observations`][ls_bayesian.posterior.likelihood.GaussianLogLikelihood.from_vertex_observations].

    Methods:
        from_vertex_observations: Create the likelihood for independent point observations at
            mesh vertices.
        evaluate_cost: Evaluate $\Phi(u)$.
        evaluate_gradient: Evaluate $\mathcal{B}^T \Gamma^{-1} (\mathcal{B}u - d)$.
        evaluate_hessian_vector_product: Evaluate $\mathcal{B}^T \Gamma^{-1} \mathcal{B} \hat{u}$.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        data_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        observation_matrix: sp.coo_matrix,
        precision_matrix: sp.coo_matrix,
    ) -> None:
        r"""Initialize the likelihood.

        Args:
            data_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Observed data $d$, shape
                `(num_observations,)`.
            observation_matrix (sp.coo_matrix): Observation operator $\mathcal{B}$, shape
                `(num_observations, solution_dim)`.
            precision_matrix (sp.coo_matrix): Noise precision $\Gamma^{-1}$, shape
                `(num_observations, num_observations)`. Must be symmetric positive definite.

        Raises:
            ValueError: If the data vector is not one-dimensional.
            ValueError: If the data vector contains non-finite values.
            ValueError: If the observation matrix row count does not match the data dimension.
            ValueError: If the precision matrix shape does not match the data dimension.
        """
        if data_vector.ndim != 1:
            raise ValueError(
                f"Data vector must be one-dimensional, but has shape {data_vector.shape}."
            )
        non_finite_values = data_vector[~np.isfinite(data_vector)]
        if non_finite_values.size > 0:
            raise ValueError(f"Data vector must be finite, but contains {non_finite_values}.")
        num_observations = data_vector.shape[0]
        if observation_matrix.shape[0] != num_observations:
            raise ValueError(
                f"Observation matrix shape {observation_matrix.shape} does not match "
                f"the data dimension {num_observations}."
            )
        if precision_matrix.shape != (num_observations, num_observations):
            raise ValueError(
                f"Precision matrix shape {precision_matrix.shape} does not match "
                f"the data dimension {num_observations}."
            )
        self._data_vector = data_vector
        self._observation_matrix = observation_matrix
        self._precision_matrix = precision_matrix

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def from_vertex_observations(cls, settings: VertexObservationSettings) -> Self:
        r"""Create the likelihood for independent point observations at mesh vertices.

        Assembles the observation operator with
        [`assemble_vertex_observation_matrix`][ls_bayesian.posterior.likelihood.assemble_vertex_observation_matrix]
        and the diagonal noise precision with
        [`assemble_diagonal_precision_matrix`][ls_bayesian.posterior.likelihood.assemble_diagonal_precision_matrix].

        Args:
            settings (VertexObservationSettings): Settings for the point observations.

        Raises:
            ValueError: If the inputs are invalid, see the assembly functions and the constructor.

        Returns:
            Self: Likelihood with observation operator $\mathcal{B}$ and precision
                $\Gamma^{-1} = \operatorname{diag}(1 / \sigma_i^2)$.
        """
        observation_matrix = assemble_vertex_observation_matrix(
            settings.num_vertices, settings.observed_vertex_indices
        )
        precision_matrix = assemble_diagonal_precision_matrix(settings.precision_values)
        return cls(settings.data_vector, observation_matrix, precision_matrix)

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_cost(self, solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        r"""Evaluate $\frac{1}{2} (\mathcal{B}u - d)^T \Gamma^{-1} (\mathcal{B}u - d)$.

        Args:
            solution_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Solution $u$.

        Returns:
            float: Negative log-likelihood, up to an additive constant.
        """
        difference_vector = self._compute_misfit(solution_vector)
        cost = 0.5 * difference_vector @ (self._precision_matrix @ difference_vector)
        return float(cost)

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_gradient(
        self, solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Evaluate $\mathcal{B}^T \Gamma^{-1} (\mathcal{B}u - d)$.

        Args:
            solution_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Solution $u$.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Gradient, same shape as the solution.
        """
        difference_vector = self._compute_misfit(solution_vector)
        return self._observation_matrix.T @ (self._precision_matrix @ difference_vector)

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_hessian_vector_product(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Evaluate $\mathcal{B}^T \Gamma^{-1} \mathcal{B} \hat{u}$.

        The Hessian is constant, so the solution only enters through the dimension check.

        Args:
            solution_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Solution $u$.
            direction_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Direction $\hat{u}$.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Hessian-vector product, same shape as the
                solution.
        """
        self._check_solution_dimension(solution_vector)
        self._check_solution_dimension(direction_vector)
        observed_direction = self._observation_matrix @ direction_vector
        return self._observation_matrix.T @ (self._precision_matrix @ observed_direction)

    # ----------------------------------------------------------------------------------------------
    def _compute_misfit(
        self, solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Compute the data misfit $\mathcal{B}u - d$."""
        self._check_solution_dimension(solution_vector)
        return self._observation_matrix @ solution_vector - self._data_vector

    # ----------------------------------------------------------------------------------------------
    def _check_solution_dimension(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> None:
        """Check that a vector matches the observation operator's input dimension."""
        expected_shape = (self._observation_matrix.shape[1],)
        if vector.shape != expected_shape:
            raise ValueError(
                f"Solution-space vector shape {vector.shape} does not match "
                f"the expected shape {expected_shape}."
            )
