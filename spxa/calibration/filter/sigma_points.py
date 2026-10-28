r"""Merwe-Wan Scaled Unscented Transformation for generating sigma points.

Generates 2*L + 1 sigma points and weights for nonlinear Kalman filtering:
    \lambda = \alpha^2 * (L + \kappa) - L
    \mathcal{X}_0 = \bar{x}
    \mathcal{X}_i = \bar{x} + (\sqrt{(L + \lambda) * P})_i,      i = 1, ..., L
    \mathcal{X}_{i+L} = \bar{x} - (\sqrt{(L + \lambda) * P})_i,  i = 1, ..., L
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy import linalg


class MerweScaledSigmaPoints:
    r"""Van der Merwe scaled unscented transformation."""

    def __init__(
        self,
        dim: int,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"Dimension must be positive integer, got {dim}")
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)

        # Scale parameter lambda
        self.lam = (self.alpha**2) * (self.dim + self.kappa) - self.dim
        self.gamma = math.sqrt(self.dim + self.lam)

        # Compute weights
        self.n_points = 2 * self.dim + 1
        self.wm = np.empty(self.n_points, dtype=float)
        self.wc = np.empty(self.n_points, dtype=float)

        self.wm[0] = self.lam / (self.dim + self.lam)
        self.wc[0] = self.wm[0] + (1.0 - (self.alpha**2) + self.beta)

        w_rest = 0.5 / (self.dim + self.lam)
        self.wm[1:] = w_rest
        self.wc[1:] = w_rest

    def generate(self, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        r"""Generate (2*dim + 1, dim) array of sigma points."""
        m = np.asarray(mean, dtype=float)
        p = np.asarray(cov, dtype=float)

        # Cholesky factor of (dim + lambda) * P
        p_scaled = (self.dim + self.lam) * p
        try:
            chol = linalg.cholesky(p_scaled, lower=True)
        except linalg.LinAlgError:
            # Fallback to eigendecomposition if slightly non-positive definite
            eigvals, eigvecs = np.linalg.eigh(p_scaled)
            eigvals_clamped = np.maximum(1e-12, eigvals)
            chol = eigvecs @ np.diag(np.sqrt(eigvals_clamped))

        sigma_points = np.empty((self.n_points, self.dim), dtype=float)
        sigma_points[0] = m

        for i in range(self.dim):
            sigma_points[i + 1] = m + chol[:, i]
            sigma_points[i + 1 + self.dim] = m - chol[:, i]

        return sigma_points

    def reconstruct(
        self, sigma_points: np.ndarray, noise_cov: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Reconstruct mean and covariance from transformed sigma points."""
        pts = np.asarray(sigma_points, dtype=float)
        # Weighted mean
        mean = np.sum(self.wm[:, None] * pts, axis=0)

        # Weighted covariance
        diff = pts - mean[None, :]
        cov = diff.T @ (self.wc[:, None] * diff)

        if noise_cov is not None:
            cov += np.asarray(noise_cov, dtype=float)

        return mean, cov
