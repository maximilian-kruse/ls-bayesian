import numpy as np
import scipy.sparse as sp


# ==================================================================================================
def assemble_vertex_observation_matrix(
    num_vertices: int, observed_vertex_indices: np.ndarray[tuple[int], np.dtype[np.int64]]
) -> sp.coo_matrix:
    row_inds = np.arange(len(observed_vertex_indices), dtype=np.int64)
    col_inds = observed_vertex_indices
    data = np.ones(len(observed_vertex_indices), dtype=np.float64)
    observation_matrix = sp.coo_matrix(
        (data, (row_inds, col_inds)), shape=(len(observed_vertex_indices), num_vertices)
    )
    return observation_matrix


# --------------------------------------------------------------------------------------------------
def assemble_diagonal_precision_matrix(
    precision_values: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> sp.coo_matrix:
    num_observations = precision_values.shape[0]
    row_inds = np.arange(num_observations, dtype=np.int64)
    col_inds = np.arange(num_observations, dtype=np.int64)
    data = precision_values
    precision_matrix = sp.coo_matrix(
        (data, (row_inds, col_inds)), shape=(num_observations, num_observations)
    )
    return precision_matrix
