"""
Fisher Information Metric (FIM) and statistical manifold geometry for stochastic processes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class FisherInformationMetric:
    r"""Computes Fisher Information Metrics on statistical manifolds of probability densities and stochastic processes.

    Given a parameterized log-likelihood :math:`\ell(\theta) = \log p(x; \theta)`, the Fisher Information Metric is:
    .. math::
        g_{ij}(\theta) = \mathbb{E}_\theta\left[ \frac{\partial \ell}{\partial \theta_i} \frac{\partial \ell}{\partial \theta_j} \right]
        = - \mathbb{E}_\theta\left[ \frac{\partial^2 \ell}{\partial \theta_i \partial \theta_j} \right]
    """

    @staticmethod
    def gaussian_fisher(mu: float, sigma: float) -> np.ndarray:
        r"""Exact Fisher Information Matrix for the univariate Gaussian distribution :math:`\mathcal{N}(\mu, \sigma^2)`.

        Coordinates: :math:`\theta = (\mu, \sigma)`.
        .. math::
            g(\mu, \sigma) = \begin{pmatrix} \frac{1}{\sigma^2} & 0 \\ 0 & \frac{2}{\sigma^2} \end{pmatrix}
        """
        if sigma <= 0.0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")
        sig2 = sigma**2
        return np.array([
            [1.0 / sig2, 0.0],
            [0.0, 2.0 / sig2],
        ], dtype=np.float64)

    @staticmethod
    def ou_transition_fisher(
        x0: float,
        dt: float,
        theta: float,
        mu: float,
        sigma: float,
    ) -> np.ndarray:
        r"""Fisher Information Matrix for the transition density of the Ornstein-Uhlenbeck process:

        .. math::
            dX_t = \theta (\mu - X_t) dt + \sigma dW_t
        with :math:`X_{\Delta t} | X_0 \sim \mathcal{N}(m, v)` where
        :math:`m = x_0 e^{-\theta \Delta t} + \mu (1 - e^{-\theta \Delta t})`,
        :math:`v = \frac{\sigma^2}{2\theta} (1 - e^{-2\theta \Delta t})`.

        Coordinates: :math:`(\theta, \mu, \sigma)`.
        """
        if theta <= 0.0 or sigma <= 0.0 or dt <= 0.0:
            raise ValueError("theta, sigma, and dt must be strictly positive")

        exp_neg = np.exp(-theta * dt)
        one_minus_exp = 1.0 - exp_neg
        exp_neg_2 = np.exp(-2.0 * theta * dt)
        one_minus_exp_2 = 1.0 - exp_neg_2

        m = x0 * exp_neg + mu * one_minus_exp
        v = (sigma**2 / (2.0 * theta)) * one_minus_exp_2

        # Derivatives of mean m with respect to (theta, mu, sigma)
        dm_dtheta = -dt * x0 * exp_neg + dt * mu * exp_neg
        dm_dmu = one_minus_exp
        dm_dsigma = 0.0
        grad_m = np.array([dm_dtheta, dm_dmu, dm_dsigma])

        # Derivatives of variance v with respect to (theta, mu, sigma)
        dv_dtheta = (sigma**2 / (2.0 * theta**2)) * (-one_minus_exp_2 + 2.0 * theta * dt * exp_neg_2)
        dv_dmu = 0.0
        dv_dsigma = (sigma / theta) * one_minus_exp_2
        grad_v = np.array([dv_dtheta, dv_dmu, dv_dsigma])

        # For Gaussian: g_ij = (1/v) * (dm/dtheta_i)(dm/dtheta_j) + (1/(2 v^2)) * (dv/dtheta_i)(dv/dtheta_j)
        g = (1.0 / v) * np.outer(grad_m, grad_m) + (1.0 / (2.0 * v**2)) * np.outer(grad_v, grad_v)
        return g

    @staticmethod
    def empirical_fisher(scores: np.ndarray) -> np.ndarray:
        r"""Compute empirical Fisher Information Matrix from score vectors.

        :param scores: Array of shape (n_samples, n_params) containing :math:`\nabla_\theta \log p(x_k; \theta)`.
        :return: Covariance of scores, shape (n_params, n_params).
        """
        if scores.ndim != 2:
            raise ValueError(f"scores must be 2D, got shape {scores.shape}")
        n_samples = scores.shape[0]
        if n_samples < 2:
            raise ValueError("Need at least 2 samples to compute empirical Fisher matrix")
        return (scores.T @ scores) / float(n_samples)

    @staticmethod
    def numerical_hessian_fisher(
        neg_log_lik_fn: Callable[[np.ndarray], float],
        theta: np.ndarray,
        eps: float = 1e-4,
    ) -> np.ndarray:
        r"""Computes expected / observed Fisher Information via central finite difference Hessian of negative log-likelihood:

        .. math::
            g_{ij}(\theta) \approx \frac{\partial^2 (-\ell)}{\partial \theta_i \partial \theta_j}
        """
        theta = np.asarray(theta, dtype=np.float64)
        n = len(theta)
        hessian = np.zeros((n, n), dtype=np.float64)

        f0 = neg_log_lik_fn(theta)
        f_plus = np.zeros(n)
        f_minus = np.zeros(n)

        for i in range(n):
            step_i = np.zeros(n)
            step_i[i] = eps
            f_plus[i] = neg_log_lik_fn(theta + step_i)
            f_minus[i] = neg_log_lik_fn(theta - step_i)
            hessian[i, i] = (f_plus[i] - 2.0 * f0 + f_minus[i]) / (eps**2)

        for i in range(n):
            for j in range(i + 1, n):
                step_ij = np.zeros(n)
                step_ij[i] = eps
                step_ij[j] = eps
                f_pp = neg_log_lik_fn(theta + step_ij)

                step_imj = np.zeros(n)
                step_imj[i] = eps
                step_imj[j] = -eps
                f_pm = neg_log_lik_fn(theta + step_imj)

                step_mij = np.zeros(n)
                step_mij[i] = -eps
                step_mij[j] = eps
                f_mp = neg_log_lik_fn(theta + step_mij)

                step_mm = np.zeros(n)
                step_mm[i] = -eps
                step_mm[j] = -eps
                f_mm = neg_log_lik_fn(theta + step_mm)

                val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps**2)
                hessian[i, j] = val
                hessian[j, i] = val

        return hessian

    @staticmethod
    def is_positive_definite(g: np.ndarray, tol: float = 1e-8) -> bool:
        r"""Checks whether the metric tensor :math:`g` is symmetric positive-definite."""
        if g.ndim != 2 or g.shape[0] != g.shape[1]:
            return False
        if not np.allclose(g, g.T, atol=1e-6):
            return False
        eigvals = np.linalg.eigvalsh(g)
        return bool(np.all(eigvals > tol))

    @staticmethod
    def is_positive_semidefinite(g: np.ndarray, tol: float = -1e-8) -> bool:
        r"""Checks whether the metric tensor :math:`g` is symmetric positive semi-definite."""
        if g.ndim != 2 or g.shape[0] != g.shape[1]:
            return False
        if not np.allclose(g, g.T, atol=1e-6):
            return False
        eigvals = np.linalg.eigvalsh(g)
        return bool(np.all(eigvals >= tol))
