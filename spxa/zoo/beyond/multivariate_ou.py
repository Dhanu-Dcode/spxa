"""Multivariate Ornstein-Uhlenbeck (Vector OU) Process.

A d-dimensional mean-reverting Gaussian diffusion:
    dX(t) = -K * (X(t) - theta) dt + Sigma * dW(t)

The process possesses exact Gaussian transitions X(t) ~ N(mu(t), V(t)), where the stationary
covariance V_inf solves the continuous Lyapunov equation:
    K * V_inf + V_inf * K^T = Sigma * Sigma^T
and the finite-time conditional covariance is:
    V(t) = V_inf - exp(-K*t) * V_inf * exp(-K^T * t)

References:
    Gardiner, C. W. (2004). Handbook of Stochastic Methods for Physics, Chemistry and the Natural Sciences.
        Springer, Sec. 4.4.
    Uhlenbeck, G. E., & Ornstein, L. S. (1930). On the theory of the Brownian motion.
        Physical Review, 36(5), 823-841.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy import linalg

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties


class MultivariateOUProcess(Process):
    r"""d-dimensional Multivariate Ornstein-Uhlenbeck vector diffusion process.

    .. math::
        d\boldsymbol{X}_t = -\mathbf{K} (\boldsymbol{X}_t - \boldsymbol{\theta}) dt + \boldsymbol{\Sigma} d\boldsymbol{W}_t
    """

    def __init__(
        self,
        k_matrix: Union[list[list[float]], np.ndarray],
        theta: Union[list[float], np.ndarray],
        sigma_matrix: Union[list[list[float]], np.ndarray],
        x0: Optional[Union[list[float], np.ndarray]] = None,
    ) -> None:
        self.k_mat = np.array(k_matrix, dtype=float)
        self.theta_vec = np.array(theta, dtype=float)
        self.sigma_mat = np.array(sigma_matrix, dtype=float)

        if self.k_mat.ndim != 2 or self.k_mat.shape[0] != self.k_mat.shape[1]:
            raise ValueError(f"K matrix must be square (d x d), got {self.k_mat.shape}")

        d = self.k_mat.shape[0]
        if self.theta_vec.shape != (d,):
            raise ValueError(f"theta must have shape ({d},), got {self.theta_vec.shape}")
        if self.sigma_mat.shape != (d, d):
            raise ValueError(f"sigma matrix must have shape ({d}, {d}), got {self.sigma_mat.shape}")

        eigvals = np.linalg.eigvals(self.k_mat)
        if np.any(np.real(eigvals) <= 0):
            raise ValueError(
                f"All eigenvalues of K must have strictly positive real parts for stability, got {eigvals}"
            )

        self.dim = d
        self.x0_vec = np.zeros(d) if x0 is None else np.array(x0, dtype=float)
        if self.x0_vec.shape != (d,):
            raise ValueError(f"x0 must have shape ({d},), got {self.x0_vec.shape}")

        # Compute stationary covariance via Lyapunov equation: K * V_inf + V_inf * K^T = Q
        q = self.sigma_mat @ self.sigma_mat.T
        self.v_inf = linalg.solve_continuous_lyapunov(self.k_mat, q)
        # Ensure symmetric positive semi-definite
        self.v_inf = 0.5 * (self.v_inf + self.v_inf.T)

        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _triplet(self):
        raise ExactnessError(
            "MultivariateOUProcess has state-dependent drift -K*(X - theta) and cannot be represented by a Levy triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            notes=["Multivariate Ornstein-Uhlenbeck vector diffusion (Vector OU)"],
        )

    def mean_vector(self, t: float) -> np.ndarray:
        r"""Conditional mean vector E[X(t) | X(0)] = theta + exp(-K*t) * (X_0 - theta)."""
        if t <= 0:
            return self.x0_vec.copy()
        exp_kt = linalg.expm(-self.k_mat * t)
        return self.theta_vec + exp_kt @ (self.x0_vec - self.theta_vec)

    def covariance_matrix(self, t: float) -> np.ndarray:
        r"""Conditional covariance matrix Var[X(t) | X(0)] = V_inf - exp(-K*t) * V_inf * exp(-K^T * t)."""
        if t <= 0:
            return np.zeros((self.dim, self.dim))
        exp_kt = linalg.expm(-self.k_mat * t)
        cov = self.v_inf - exp_kt @ self.v_inf @ exp_kt.T
        return 0.5 * (cov + cov.T)

    def stationary_mean(self) -> np.ndarray:
        """Long-run stationary mean vector."""
        return self.theta_vec.copy()

    def stationary_covariance(self) -> np.ndarray:
        """Long-run stationary covariance matrix."""
        return self.v_inf.copy()

    def characteristic_function(self, u: np.ndarray, t: float) -> complex:
        r"""Characteristic function E[exp(i u^T X_t)] = exp(i u^T mu(t) - 0.5 u^T V(t) u)."""
        u_vec = np.asarray(u, dtype=float)
        mu = self.mean_vector(t)
        cov = self.covariance_matrix(t)
        quad = float(u_vec.T @ cov @ u_vec)
        lin = float(u_vec.T @ mu)
        return complex(math.cos(lin), math.sin(lin)) * math.exp(-0.5 * quad)

    def simulate(
        self,
        t_max: float,
        num_steps: int,
        num_paths: int = 1,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate sample paths of the vector process using exact Gaussian transitions.

        Returns array of shape (num_paths, dim, num_steps + 1).
        """
        if seed is not None:
            np.random.seed(seed)

        dt = float(t_max) / num_steps
        d = self.dim
        paths = np.zeros((num_paths, d, num_steps + 1))
        paths[:, :, 0] = self.x0_vec

        exp_kdt = linalg.expm(-self.k_mat * dt)
        cov_step = self.v_inf - exp_kdt @ self.v_inf @ exp_kdt.T
        cov_step = 0.5 * (cov_step + cov_step.T)

        # Cholesky factor of step covariance
        chol_step = np.linalg.cholesky(cov_step + 1e-12 * np.eye(d))

        for step in range(num_steps):
            x_prev = paths[:, :, step]
            # Deterministic drift update
            mean_step = self.theta_vec[None, :] + (x_prev - self.theta_vec[None, :]) @ exp_kdt.T
            # Stochastic Gaussian update
            z = np.random.normal(0.0, 1.0, (num_paths, d))
            delta = z @ chol_step.T
            paths[:, :, step + 1] = mean_step + delta

        return paths


# Aliases
MultivariateOU = MultivariateOUProcess
VectorOU = MultivariateOUProcess
