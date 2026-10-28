"""
Fractional Brownian motion, Riemann-Liouville kernels, and hybrid convolution schemes for rough volatility.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.special import gamma


class FractionalKernel:
    r"""Evaluates fractional singular power kernels and simulates fractional Brownian motion :math:`B_t^H`."""

    def __init__(self, hurst: float = 0.1) -> None:
        if not (0.0 < hurst < 1.0):
            raise ValueError(f"Hurst exponent H must be in (0, 1), got {hurst}")
        self.H = float(hurst)
        self.alpha = self.H - 0.5  # In (-0.5, 0) for rough regime H < 0.5
        self.prefactor = np.sqrt(2.0 * self.H) / gamma(self.H + 0.5)

    def kernel_value(self, t: float) -> float:
        r"""Evaluates Riemann-Liouville fractional kernel :math:`K_H(t) = \frac{\sqrt{2H}}{\Gamma(H + 1/2)} t^{H - 1/2}`."""
        if t <= 0.0:
            return 0.0
        return float(self.prefactor * (t**self.alpha))

    def sample_fbm_cholesky(
        self,
        t_mat: float = 1.0,
        n_steps: int = 100,
        n_paths: int = 1000,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        r"""Exact simulation of fractional Brownian motion :math:`B_t^H` via Cholesky decomposition of covariance:

        .. math::
            \mathbb{E}[B_t^H B_s^H] = \frac{1}{2} \left( t^{2H} + s^{2H} - |t - s|^{2H} \right)
        """
        rng = np.random.default_rng(seed)
        time_grid = np.linspace(0.0, t_mat, n_steps + 1)
        t_grid = time_grid[1:]  # Exclude t=0 where variance is 0
        n = len(t_grid)

        # Covariance matrix
        t_col = t_grid[:, np.newaxis]
        t_row = t_grid[np.newaxis, :]
        cov = 0.5 * (t_col**(2.0 * self.H) + t_row**(2.0 * self.H) - np.abs(t_col - t_row)**(2.0 * self.H))

        # Regularize diagonal slightly for numerical stability when H is very small
        cov += 1e-10 * np.eye(n)
        chol = np.linalg.cholesky(cov)

        z = rng.normal(0.0, 1.0, size=(n, n_paths))
        paths = chol @ z  # Shape (n, n_paths)

        full_paths = np.zeros((n_paths, n_steps + 1))
        full_paths[:, 1:] = paths.T

        return {
            "paths": full_paths,
            "time_grid": time_grid,
        }

    def hybrid_convolution(
        self,
        dw: np.ndarray,
        dt: float,
        b_cutoff: int = 3,
    ) -> np.ndarray:
        r"""Bennedsen-Lunde-Pakkanen (2017) Hybrid scheme for evaluating the Volterra integral :math:`\int_0^t (t - s)^\alpha dW_s`.

        Splits into:
        1. Exact local integration for the singular part :math:`s \in [t - b \Delta t, t]`.
        2. Fast discrete convolution for the regular tail.
        """
        n_steps = len(dw)
        res = np.zeros(n_steps + 1)
        alpha = self.alpha

        # Local singular weights: exact integrals of (k dt - s)^alpha ds
        local_weights = np.zeros(b_cutoff)
        for k in range(1, b_cutoff + 1):
            w_k = ((k * dt)**(alpha + 1.0) - ((k - 1) * dt)**(alpha + 1.0)) / (alpha + 1.0)
            local_weights[k - 1] = w_k / dt

        # Convolution weights for tail
        tail_len = max(0, n_steps - b_cutoff)
        if tail_len > 0:
            k_tail = np.arange(b_cutoff, n_steps)
            tail_weights = (k_tail * dt)**alpha
            weights = np.concatenate([local_weights, tail_weights])
        else:
            weights = local_weights[:n_steps]

        # Time-stepping convolution
        for i in range(1, n_steps + 1):
            w_slice = weights[:i][::-1]
            dw_slice = dw[:i]
            res[i] = np.sum(w_slice * dw_slice)

        return res
