r"""Interfaces for the components the posterior is composed of.

The negative log-posterior for a parameter $m$ reads

$$
\begin{equation*}
    J(m) = \Phi(F(m)) + R(m),
\end{equation*}
$$

where $F$ maps parameters to (PDE) solutions $u = F(m)$, $\Phi$ is the negative log-likelihood
of the solution given observed data, and $R$ is the negative log-prior of a Gaussian prior
measure. Each of the three terms is represented by its own interface, so that concrete
implementations can be exchanged independently and composed in
[`LogPosterior`][ls_bayesian.posterior.posterior.LogPosterior].

Implementations inherit from these abstract base classes. Implementations from other
subpackages are not coupled to this module directly. Instead, they are connected through thin
adapters in the application layer, which inherit from the respective interface and delegate to
the implementation.

Likelihood and parameter-to-solution map are formulated for the general, non-linear and
non-Gaussian case. Subclasses have to accept arguments they do not require. Adapters absorb such
arguments, so the wrapped implementations keep their own signatures.

Classes:
    Likelihood: Interface for the negative log-likelihood $\Phi$ on the solution space.
    ParameterToSolutionMap: Interface for the parameter-to-solution map $F$.
    GaussianPrior: Interface for a Gaussian prior measure on the parameter space.
"""

from abc import ABC, abstractmethod

import numpy as np


# ==================================================================================================
class Likelihood(ABC):
    r"""Interface for the negative log-likelihood $\Phi(u)$, defined on the solution space.

    Methods:
        evaluate_cost: Evaluate $\Phi(u)$.
        evaluate_gradient: Evaluate $\nabla_u \Phi(u)$.
        evaluate_hessian_vector_product: Evaluate $\nabla_u^2 \Phi(u)\, \hat{u}$.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_cost(self, solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        r"""Evaluate the negative log-likelihood $\Phi(u)$.

        Args:
            solution_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Solution $u$.

        Returns:
            float: Negative log-likelihood, up to an additive constant.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_gradient(
        self, solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Evaluate the gradient $\nabla_u \Phi(u)$ with respect to the solution.

        Args:
            solution_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Solution $u$.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Gradient, same shape as the solution.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_hessian_vector_product(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Evaluate the Hessian-vector product $\nabla_u^2 \Phi(u)\, \hat{u}$.

        Args:
            solution_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Solution $u$ at which
                the Hessian is evaluated.
            direction_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Direction $\hat{u}$.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Hessian-vector product, same shape as the
                solution.
        """


# ==================================================================================================
class ParameterToSolutionMap(ABC):
    r"""Interface for the parameter-to-solution map $u = F(m)$.

    Typically, $F$ involves the solution of a PDE, making it the most expensive part of a posterior
    evaluation. Derivatives are provided in the form of gradients, Jacobian-vector products, and
    Hessian-vector products

    Methods:
        evaluate_forward: Evaluate $F(m)$.
        evaluate_gradient: Apply the transposed Jacobian $(\nabla_m F)^T$ to a solution vector.
        evaluate_jacobian_vector_product: Apply the Jacobian $\nabla_m F$ to a direction.
        evaluate_hessian_vector_product: Evaluate the second-order contribution of $F$.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_forward(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Evaluate the forward map $u = F(m)$.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter $m$.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Solution $u$.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_gradient(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        adjoint_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply the transposed Jacobian, $(\nabla_m F(m))^T v$.

        Within the posterior, $v = \nabla_u \Phi(u)$ is the likelihood gradient, so that the result
        is the likelihood contribution to the gradient with respect to the parameter (chain rule).

        Args:
            solution_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Solution $u = F(m)$.
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter $m$.
            adjoint_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Solution-space vector
                $v$ the transposed Jacobian is applied to.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Parameter-space vector.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_jacobian_vector_product(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply the Jacobian, $\nabla_m F(m)\, \hat{m}$.

        Args:
            solution_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Solution $u = F(m)$.
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter $m$.
            direction_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter-space
                direction $\hat{m}$.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Solution-space vector.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_hessian_vector_product(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        adjoint_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        gradient_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Evaluate the second-order contribution of $F$ to the posterior Hessian-vector product.

        Args:
            solution_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Solution $u = F(m)$.
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter $m$.
            direction_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter-space
                direction $\hat{m}$.
            adjoint_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Solution-space vector
                from the gradient evaluation at $m$.
            gradient_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter-space
                gradient from the gradient evaluation at $m$.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Parameter-space vector.
        """


# ==================================================================================================
class GaussianPrior(ABC):
    r"""Interface for a Gaussian prior measure $\mathcal{N}(\overline{m}, \mathcal{C})$.

    The negative log-prior is $R(m) = \frac{1}{2}(m - \overline{m})^T \mathcal{C}^{-1}
    (m - \overline{m})$, up to an additive constant. Its Hessian is the constant precision operator
    $\mathcal{C}^{-1}$, so Hessian-vector products do not depend on the parameter. Samples are
    generated as $\overline{m} + \widehat{\mathcal{C}}\xi$ with a factorization
    $\widehat{\mathcal{C}}\widehat{\mathcal{C}}^T = \mathcal{C}$ and an i.i.d. standard normal
    vector $\xi$.

    Methods:
        evaluate_cost: Evaluate $R(m)$.
        evaluate_gradient: Evaluate $\nabla_m R(m) = \mathcal{C}^{-1}(m - \overline{m})$.
        evaluate_hessian_vector_product: Evaluate $\mathcal{C}^{-1} \hat{m}$.
        generate_sample: Draw a sample from the prior distribution.
        apply_covariance_operator: Apply the covariance operator $\mathcal{C}$.
        apply_covariance_factorization: Apply the covariance factorization
            $\widehat{\mathcal{C}}$.
        apply_precision_operator: Apply the precision operator $\mathcal{C}^{-1}$.

    Attributes:
        random_vector_size: Size of the i.i.d. normal vector $\xi$ required for sampling.
    """

    # ----------------------------------------------------------------------------------------------
    @property
    @abstractmethod
    def random_vector_size(self) -> int:
        r"""Return the size of the i.i.d. normal vector $\xi$ required for sampling.

        Returns:
            int: Number of columns of the covariance factorization $\widehat{\mathcal{C}}$.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_cost(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        r"""Evaluate the negative log-prior $R(m)$.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter $m$.

        Returns:
            float: Negative log-prior, up to an additive constant.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_gradient(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Evaluate the gradient $\nabla_m R(m) = \mathcal{C}^{-1}(m - \overline{m})$.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter $m$.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Gradient, same shape as the parameter.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_hessian_vector_product(
        self, direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Evaluate the Hessian-vector product $\mathcal{C}^{-1} \hat{m}$.

        Args:
            direction_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Direction $\hat{m}$.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Hessian-vector product, same shape as the
                parameter.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def generate_sample(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Draw a sample $\overline{m} + \widehat{\mathcal{C}}\xi$ from the prior distribution.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Sample, same shape as the parameter.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def apply_covariance_operator(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply the covariance operator $\mathcal{C}$.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter-space
                vector to apply the covariance operator to.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Result, same shape as the parameter.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def apply_covariance_factorization(
        self, random_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply the covariance factorization $\widehat{\mathcal{C}}$.

        Args:
            random_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Vector to apply the
                factorization to, shape `(random_vector_size,)`.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Result, same shape as the parameter.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def apply_precision_operator(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply the precision operator $\mathcal{C}^{-1}$.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter-space
                vector to apply the precision operator to.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Result, same shape as the parameter.
        """
