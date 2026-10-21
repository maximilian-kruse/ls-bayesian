r"""Solvers for the generalized eigenvalue problem of a Hessian relative to a precision operator.

Given a symmetric positive semi-definite Hessian $H$ (e.g. the Gauss-Newton Hessian of a data
misfit) and a symmetric positive definite precision $Q = \Gamma_0^{-1}$, we seek the dominant
eigenpairs of

$$
H v_i = \lambda_i Q v_i, \qquad v_i^T Q v_j = \delta_{ij}.
$$

They determine the low-rank Laplace approximation of the posterior (Bui-Thanh, Ghattas, Martin,
Stadler, 2013; Villa, Petra, Ghattas, 2021), see
[`LowRankGaussian`][ls_bayesian.lowrank.gaussian.LowRankGaussian]. All solvers work matrix-free and
require $H$, $Q$ and $Q^{-1}$ as callables.

Classes:
    GeneralizedEigenpairs: Eigenvalues and $Q$-orthonormal eigenvectors.
    GeneralizedEigensolver: ABC strategy interface for generalized eigensolvers.
    LanczosSolver: Implicitly restarted Lanczos method (ARPACK).
    RandomizedDoublePassSolver: Randomized double-pass algorithm (Saibaba et al., 2016).

Functions:
    compute_generalized_eigenpairs: Solve the problem with a given solver and discard eigenvalues
        that are not positive.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, override

import numpy as np
import scipy.sparse.linalg as sla
from beartype.vale import Is

VectorAction = Callable[
    [np.ndarray[tuple[int], np.dtype[np.float64]]], np.ndarray[tuple[int], np.dtype[np.float64]]
]
PositiveInt = Annotated[int, Is[lambda value: value > 0]]
NonNegativeInt = Annotated[int, Is[lambda value: value >= 0]]
PositiveFloat = Annotated[float, Is[lambda value: value > 0]]

# Eigenvalues of a Gram matrix are only accurate relative to the largest one, to about machine
# precision. Directions with smaller Gram eigenvalues are numerical noise, i.e. the sketched
# subspace is rank deficient (e.g. if the Hessian has a lower rank than the sketch size).
_RELATIVE_RANK_TOLERANCE = 1e-12


# ==================================================================================================
@dataclass(frozen=True)
class GeneralizedEigenpairs:
    r"""Eigenpairs of $Hv = \lambda Qv$ with $V^TQV = I$.

    Attributes:
        eigenvalues (np.ndarray[tuple[int], np.dtype[np.float64]]): Eigenvalues $\lambda_i$ in
            descending order, shape `(rank,)`.
        eigenvectors (np.ndarray[tuple[int, int], np.dtype[np.float64]]): Corresponding
            $Q$-orthonormal eigenvectors as columns, shape `(parameter_size, rank)`.
    """

    eigenvalues: np.ndarray[tuple[int], np.dtype[np.float64]]
    eigenvectors: np.ndarray[tuple[int, int], np.dtype[np.float64]]

    # ----------------------------------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.eigenvalues.ndim != 1:
            raise ValueError(f"Eigenvalues must be 1D, but have shape {self.eigenvalues.shape}.")
        if self.eigenvectors.ndim != 2 or self.eigenvectors.shape[1] != self.eigenvalues.size:
            raise ValueError(
                f"Eigenvectors of shape {self.eigenvectors.shape} do not match "
                f"{self.eigenvalues.size} eigenvalues."
            )


# ==================================================================================================
class GeneralizedEigensolver(ABC):
    r"""ABC strategy interface for solvers of $Hv = \lambda Qv$ for the dominant eigenpairs.

    Methods:
        solve: Compute the dominant eigenpairs.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def solve(
        self,
        hessian_action: VectorAction,
        precision_action: VectorAction,
        covariance_action: VectorAction,
        size: PositiveInt,
        num_eigenvalues: PositiveInt,
    ) -> GeneralizedEigenpairs:
        r"""Compute the dominant eigenpairs of $Hv = \lambda Qv$.

        Args:
            hessian_action (VectorAction): Product with the symmetric positive semi-definite
                Hessian $H$.
            precision_action (VectorAction): Product with the precision $Q$.
            covariance_action (VectorAction): Product with $Q^{-1}$.
            size (int): Dimension $n$ of the parameter space.
            num_eigenvalues (int): Number of eigenpairs $k$ to compute.

        Returns:
            GeneralizedEigenpairs: Eigenpairs with eigenvalues in descending order. A solver may
                return fewer than `num_eigenvalues` pairs if it detects a lower numerical rank.
        """


# ==================================================================================================
@dataclass(frozen=True)
class LanczosSolver(GeneralizedEigensolver):
    r"""Implicitly restarted Lanczos method for the dominant eigenpairs, via ARPACK (Lehoucq,
    Sorensen, Yang, 1998) as wrapped by `scipy.sparse.linalg.eigsh`.

    The problem is solved in regular inverse mode with operator $Q^{-1}H$ in the $Q$-inner
    product, so that the eigenvectors are $Q$-orthonormal. Products are computed one at a time and
    sequentially, with a cost of typically a few multiples of `num_eigenvalues` evaluations of each
    action.

    Attributes:
        tolerance (float): Relative convergence tolerance of the Ritz values. Applications of
            $Q$ and $Q^{-1}$ that are only accurate up to the tolerance of an inner iterative
            solver limit the achievable accuracy, so that tolerance should be tighter. Defaults
            to `1e-8`.
        seed (int): Seed of the random starting vector. Defaults to `0`.
    """

    tolerance: PositiveFloat = 1e-8
    seed: int = 0

    # ----------------------------------------------------------------------------------------------
    @override
    def solve(
        self,
        hessian_action: VectorAction,
        precision_action: VectorAction,
        covariance_action: VectorAction,
        size: PositiveInt,
        num_eigenvalues: PositiveInt,
    ) -> GeneralizedEigenpairs:
        if num_eigenvalues >= size:
            raise ValueError(
                f"Lanczos solver requires fewer eigenvalues than the parameter size {size}, "
                f"but {num_eigenvalues} were requested."
            )
        shape = (size, size)
        eigenvalues, eigenvectors = sla.eigsh(
            sla.LinearOperator(shape, matvec=hessian_action, dtype=np.float64),
            k=num_eigenvalues,
            M=sla.LinearOperator(shape, matvec=precision_action, dtype=np.float64),
            Minv=sla.LinearOperator(shape, matvec=covariance_action, dtype=np.float64),
            which="LA",
            tol=self.tolerance,
            v0=np.random.default_rng(self.seed).standard_normal(size),
        )
        descending_order = np.argsort(eigenvalues)[::-1]
        return GeneralizedEigenpairs(
            eigenvalues[descending_order], eigenvectors[:, descending_order]
        )


# ==================================================================================================
@dataclass(frozen=True)
class RandomizedDoublePassSolver(GeneralizedEigensolver):
    r"""Randomized double-pass algorithm for the dominant eigenpairs (Saibaba, Lee, Kitanidis,
    2016, Algorithm 3; implemented as `doublePassG` in hIPPYlib, Villa, Petra, Ghattas, 2021).

    1. Sketch the range of $Q^{-1}H$ with a Gaussian matrix $\Omega \in \mathbb{R}^{n\times(k+p)}$,
       $Y = Q^{-1}H\Omega$ (first pass).
    2. $Q$-orthonormalize, $Z^TQZ = I$ with $\mathrm{span}(Z) = \mathrm{span}(Y)$.
    3. Project, $T = Z^THZ$, and diagonalize $T = U\Lambda U^T$ (second pass). The eigenpairs are
       $(\lambda_i, ZU_i)$.

    Unlike [`LanczosSolver`][ls_bayesian.lowrank.eigensolvers.LanczosSolver], the products are
    independent and can be evaluated in batches by the caller, at a fixed cost of $2(k+p)$ products
    with $H$. The result is accurate for the eigenvalues that are well separated from the
    discarded tail of the spectrum, so oversampling should be chosen accordingly.

    Deviation: the $Q$-orthonormalization uses an eigendecomposition of the Gram matrix $Y^TQY$,
    applied twice for stability, instead of a Cholesky factorization. This tolerates a sketch of
    lower numerical rank (e.g. `num_eigenvalues + oversampling` exceeding the rank of $H$), where
    the Cholesky factorization breaks down, by dropping the numerically null directions.

    Attributes:
        oversampling (int): Number $p$ of sketch vectors beyond `num_eigenvalues`. Defaults to
            `10`, the value suggested by Saibaba et al. (2016).
        seed (int): Seed of the Gaussian sketch. Defaults to `0`.
    """

    oversampling: NonNegativeInt = 10
    seed: int = 0

    # ----------------------------------------------------------------------------------------------
    @override
    def solve(
        self,
        hessian_action: VectorAction,
        precision_action: VectorAction,
        covariance_action: VectorAction,
        size: PositiveInt,
        num_eigenvalues: PositiveInt,
    ) -> GeneralizedEigenpairs:
        sketch_size = num_eigenvalues + self.oversampling
        if sketch_size > size:
            raise ValueError(
                f"Sketch size {sketch_size} (num_eigenvalues + oversampling) exceeds the "
                f"parameter size {size}."
            )
        gaussian_matrix = np.random.default_rng(self.seed).standard_normal((size, sketch_size))
        sketch = _apply_columnwise(
            lambda column: covariance_action(hessian_action(column)), gaussian_matrix
        )
        basis = _orthonormalize_in_precision_inner_product(sketch, precision_action)
        basis = _orthonormalize_in_precision_inner_product(basis, precision_action)
        projected_hessian = basis.T @ _apply_columnwise(hessian_action, basis)
        projected_hessian = 0.5 * (projected_hessian + projected_hessian.T)
        eigenvalues, rotation = np.linalg.eigh(projected_hessian)
        descending_order = np.argsort(eigenvalues)[::-1][:num_eigenvalues]
        return GeneralizedEigenpairs(
            eigenvalues[descending_order], basis @ rotation[:, descending_order]
        )


# ==================================================================================================
def compute_generalized_eigenpairs(
    hessian_action: VectorAction,
    precision_action: VectorAction,
    covariance_action: VectorAction,
    size: PositiveInt,
    num_eigenvalues: PositiveInt,
    solver: GeneralizedEigensolver,
    minimum_eigenvalue: float = 0.0,
) -> GeneralizedEigenpairs:
    r"""Compute the dominant eigenpairs of $Hv = \lambda Qv$, discarding non-positive eigenvalues.

    A Gauss-Newton Hessian is positive semi-definite, but numerical errors, or Hessians of
    rank lower than `num_eigenvalues`, lead to eigenvalues that are zero up to rounding errors,
    possibly with negative sign. They carry no information on the data and are removed, such that
    the remaining pairs can be used in a
    [`LowRankGaussian`][ls_bayesian.lowrank.gaussian.LowRankGaussian].

    Args:
        hessian_action (VectorAction): Product with the symmetric positive semi-definite Hessian
            $H$, e.g. Gauss-Newton, in the same (coefficient) space as the precision.
        precision_action (VectorAction): Product with the precision $Q = \Gamma_0^{-1}$ of the
            reference Gaussian.
        covariance_action (VectorAction): Product with its covariance $\Gamma_0 = Q^{-1}$.
        size (int): Dimension $n$ of the parameter space.
        num_eigenvalues (int): Number of eigenpairs $k$ to compute.
        solver (GeneralizedEigensolver): Eigensolver strategy.
        minimum_eigenvalue (float): Eigenpairs with $\lambda \le$ `minimum_eigenvalue` are
            discarded. Defaults to `0.0`.

    Returns:
        GeneralizedEigenpairs: Retained eigenpairs, in descending order of eigenvalues.
    """
    if num_eigenvalues > size:
        raise ValueError(
            f"Number of eigenvalues {num_eigenvalues} exceeds the parameter size {size}."
        )
    eigenpairs = solver.solve(
        hessian_action, precision_action, covariance_action, size, num_eigenvalues
    )
    retained = eigenpairs.eigenvalues > minimum_eigenvalue
    return GeneralizedEigenpairs(
        eigenpairs.eigenvalues[retained], eigenpairs.eigenvectors[:, retained]
    )


# ==================================================================================================
def _apply_columnwise(
    action: VectorAction, matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]]
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Apply a vector action to each column of a matrix. The loop is at the boundary to the
    matrix-free callables, which are defined on single vectors."""
    return np.column_stack([action(column) for column in matrix.T])


# --------------------------------------------------------------------------------------------------
def _orthonormalize_in_precision_inner_product(
    vectors: np.ndarray[tuple[int, int], np.dtype[np.float64]], precision_action: VectorAction
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    r"""Return a basis $Z$ of the span of the columns, with $Z^TQZ = I$.

    With the eigendecomposition $Y^TQY = G\Sigma G^T$, $Z = YG\Sigma^{-1/2}$ for the eigenvalues
    above `_RELATIVE_RANK_TOLERANCE`, relative to the largest one.
    """
    gram_matrix = vectors.T @ _apply_columnwise(precision_action, vectors)
    gram_eigenvalues, gram_eigenvectors = np.linalg.eigh(0.5 * (gram_matrix + gram_matrix.T))
    retained = gram_eigenvalues > _RELATIVE_RANK_TOLERANCE * np.max(gram_eigenvalues, initial=0.0)
    return vectors @ (gram_eigenvectors[:, retained] / np.sqrt(gram_eigenvalues[retained]))
