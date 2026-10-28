"""
Affine connections, Christoffel symbols, and Amari alpha-connections on statistical manifolds.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class AffineConnection:
    r"""Computes Christoffel symbols, Levi-Civita Riemannian connection, and Amari's :math:`\alpha`-connections.

    On a Riemannian manifold with metric tensor :math:`g_{ij}(\theta)`:
    .. math::
        \Gamma^k_{ij} = \frac{1}{2} g^{kl} \left( \frac{\partial g_{jl}}{\partial \theta^i} + \frac{\partial g_{il}}{\partial \theta^j} - \frac{\partial g_{ij}}{\partial \theta^l} \right)
    """

    def __init__(
        self,
        metric_fn: Callable[[np.ndarray], np.ndarray],
        dim: int,
        eps: float = 1e-4,
    ) -> None:
        self.metric_fn = metric_fn
        self.dim = int(dim)
        self.eps = float(eps)

    def metric_derivatives(self, theta: np.ndarray) -> np.ndarray:
        r"""Compute partial derivatives of the metric tensor: :math:`\partial_k g_{ij}`.

        :return: Array of shape (dim, dim, dim) where index (k, i, j) represents :math:`\frac{\partial g_{ij}}{\partial \theta^k}`.
        """
        n = self.dim
        eps = self.eps
        dg = np.zeros((n, n, n), dtype=np.float64)

        for k in range(n):
            step = np.zeros(n)
            step[k] = eps
            g_plus = self.metric_fn(theta + step)
            g_minus = self.metric_fn(theta - step)
            dg[k, :, :] = (g_plus - g_minus) / (2.0 * eps)

        return dg

    def christoffel_symbols(self, theta: np.ndarray) -> np.ndarray:
        r"""Compute Christoffel symbols of the second kind :math:`\Gamma^k_{ij}(\theta)` (Levi-Civita connection).

        :return: Array of shape (dim, dim, dim) where index (k, i, j) is :math:`\Gamma^k_{ij}`.
        """
        n = self.dim
        theta = np.asarray(theta, dtype=np.float64)
        g = self.metric_fn(theta)
        g_inv = np.linalg.inv(g)
        dg = self.metric_derivatives(theta)

        # Gamma^k_ij = 0.5 * g^{kl} * (dg_jl/dtheta_i + dg_il/dtheta_j - dg_ij/dtheta_l)
        gamma = np.zeros((n, n, n), dtype=np.float64)

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    term = 0.0
                    for l in range(n):
                        # dg[i, j, l] is dg_jl / dtheta_i
                        # dg[j, i, l] is dg_il / dtheta_j
                        # dg[l, i, j] is dg_ij / dtheta_l
                        bracket = dg[i, j, l] + dg[j, i, l] - dg[l, i, j]
                        term += g_inv[k, l] * bracket
                    gamma[k, i, j] = 0.5 * term

        return gamma

    def amari_alpha_connection(
        self,
        theta: np.ndarray,
        alpha: float = 0.0,
        skewness_tensor: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Compute Amari's :math:`\alpha`-connection:

        .. math::
            \Gamma^{(\alpha), k}_{ij} = \Gamma^k_{ij} - \frac{\alpha}{2} g^{kl} T_{ijl}

        where :math:`T_{ijl} = \mathbb{E}[\partial_i \ell \partial_j \ell \partial_l \ell]` is the Amari skewness tensor.
        - :math:`\alpha = 0`: Riemannian Levi-Civita connection.
        - :math:`\alpha = 1`: Exponential (:math:`e`)-connection (flat for exponential families).
        - :math:`\alpha = -1`: Mixture (:math:`m`)-connection (flat for mixture families).
        """
        gamma_0 = self.christoffel_symbols(theta)
        if abs(alpha) < 1e-12 or skewness_tensor is None:
            return gamma_0

        n = self.dim
        g = self.metric_fn(theta)
        g_inv = np.linalg.inv(g)

        gamma_alpha = np.copy(gamma_0)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    correction = 0.0
                    for l in range(n):
                        correction += g_inv[k, l] * skewness_tensor[i, j, l]
                    gamma_alpha[k, i, j] -= 0.5 * alpha * correction

        return gamma_alpha

    def riemann_curvature_tensor(self, theta: np.ndarray) -> np.ndarray:
        r"""Compute Riemann curvature tensor :math:`R^l_{ijk}(\theta)`."""
        n = self.dim
        eps = self.eps
        gamma0 = self.christoffel_symbols(theta)

        # Numerical derivative of Gamma: dGamma[m, l, i, k] = d Gamma^l_ik / d theta_m
        d_gamma = np.zeros((n, n, n, n), dtype=np.float64)
        for m in range(n):
            step = np.zeros(n)
            step[m] = eps
            gamma_plus = self.christoffel_symbols(theta + step)
            gamma_minus = self.christoffel_symbols(theta - step)
            d_gamma[m, :, :, :] = (gamma_plus - gamma_minus) / (2.0 * eps)

        # R^l_ijk = d_j Gamma^l_ik - d_k Gamma^l_ij + Gamma^m_ik Gamma^l_jm - Gamma^m_ij Gamma^l_km
        riemann = np.zeros((n, n, n, n), dtype=np.float64)
        for l in range(n):
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        val = d_gamma[j, l, i, k] - d_gamma[k, l, i, j]
                        for m in range(n):
                            val += gamma0[m, i, k] * gamma0[l, j, m] - gamma0[m, i, j] * gamma0[l, k, m]
                        riemann[l, i, j, k] = val

        return riemann

    def ricci_scalar(self, theta: np.ndarray) -> float:
        r"""Compute Ricci scalar curvature :math:`R = g^{ik} R_{ik}`."""
        n = self.dim
        g = self.metric_fn(theta)
        g_inv = np.linalg.inv(g)
        r_tensor = self.riemann_curvature_tensor(theta)

        # Ricci tensor: R_ik = sum_l R^l_ilk
        ricci = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for k in range(n):
                ricci[i, k] = np.sum([r_tensor[l, i, l, k] for l in range(n)])

        scalar = float(np.sum(g_inv * ricci))
        return scalar
