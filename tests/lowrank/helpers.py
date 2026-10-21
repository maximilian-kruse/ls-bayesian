"""Constants, setup objects and pure helper functions for the `lowrank` tests.

This is a regular module, imported by test modules and `conftest.py` files alike. Fixtures live in
the `conftest.py` files, everything that is imported by name lives here.
"""

from dataclasses import dataclass
from typing import override

import numpy as np
import scipy.linalg

from ls_bayesian.lowrank.eigensolvers import GeneralizedEigenpairs, GeneralizedEigensolver
from ls_bayesian.lowrank.interfaces import ReferenceCovariance

PARAMETER_SIZE = 30
HESSIAN_RANK = 6
NUM_OBSERVATIONS = HESSIAN_RANK


# ==================================================================================================
class DenseReferenceCovariance(ReferenceCovariance):
    r"""Dense reference covariance with a non-symmetric factorization (lower Cholesky factor), as
    only $\widehat{\Gamma}_0\widehat{\Gamma}_0^T = \Gamma_0$ is guaranteed by the interface."""

    def __init__(self, covariance_matrix: np.ndarray) -> None:
        self.covariance_matrix = covariance_matrix
        self.precision_matrix = np.linalg.inv(covariance_matrix)
        self.covariance_factor = np.linalg.cholesky(covariance_matrix)

    @property
    @override
    def random_vector_size(self) -> int:
        return self.covariance_factor.shape[1]

    @override
    def apply_covariance_factorization(self, random_vector: np.ndarray) -> np.ndarray:
        return self.covariance_factor @ random_vector

    @override
    def apply_covariance_operator(self, vector: np.ndarray) -> np.ndarray:
        return self.covariance_matrix @ vector

    @override
    def apply_precision_operator(self, vector: np.ndarray) -> np.ndarray:
        return self.precision_matrix @ vector


# ==================================================================================================
class FixedEigensolver(GeneralizedEigensolver):
    """Solver returning given eigenpairs, to test post-processing independent of an algorithm."""

    def __init__(self, eigenpairs: GeneralizedEigenpairs) -> None:
        self._eigenpairs = eigenpairs

    @override
    def solve(self, hessian_action, precision_action, covariance_action, size, num_eigenvalues):
        return self._eigenpairs


# ==================================================================================================
@dataclass(frozen=True)
class DenseLowRankSetup:
    r"""Dense Gauss-Newton problem $H = J^TO^TPOJ$ of rank `HESSIAN_RANK` with SPD reference
    precision $Q$, and its exact generalized eigenpairs and posterior.

    Attributes:
        reference: Reference covariance with dense matrices.
        hessian: Gauss-Newton Hessian $H$.
        eigenvalues: The `HESSIAN_RANK` positive eigenvalues of $Hv = \lambda Qv$, descending.
        eigenvectors: Corresponding $Q$-orthonormal eigenvectors.
        posterior_covariance: $(Q + H)^{-1}$.
        posterior_precision: $Q + H$.
    """

    reference: DenseReferenceCovariance
    hessian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    posterior_covariance: np.ndarray
    posterior_precision: np.ndarray

    @property
    def exact_eigenpairs(self) -> GeneralizedEigenpairs:
        return GeneralizedEigenpairs(self.eigenvalues, self.eigenvectors)

    def apply_hessian(self, vector: np.ndarray) -> np.ndarray:
        return self.hessian @ vector


# --------------------------------------------------------------------------------------------------
def create_dense_low_rank_setup(seed: int) -> DenseLowRankSetup:
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((PARAMETER_SIZE, PARAMETER_SIZE))
    precision = factor @ factor.T / PARAMETER_SIZE + 0.1 * np.eye(PARAMETER_SIZE)
    jacobian = rng.standard_normal((NUM_OBSERVATIONS, PARAMETER_SIZE))
    noise_precision = np.diag(rng.uniform(1.0, 5.0, NUM_OBSERVATIONS))
    hessian = jacobian.T @ noise_precision @ jacobian

    eigenvalues, eigenvectors = scipy.linalg.eigh(hessian, precision)
    top = np.argsort(eigenvalues)[::-1][:HESSIAN_RANK]
    return DenseLowRankSetup(
        reference=DenseReferenceCovariance(np.linalg.inv(precision)),
        hessian=hessian,
        eigenvalues=eigenvalues[top],
        eigenvectors=eigenvectors[:, top],
        posterior_covariance=np.linalg.inv(precision + hessian),
        posterior_precision=precision + hessian,
    )
