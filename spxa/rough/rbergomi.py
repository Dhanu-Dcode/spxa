"""
Rough Bergomi (rBergomi) stochastic volatility model with power-law short-time skew.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from spxa.rough.kernel import FractionalKernel


class RoughBergomi:
    r"""Rough Bergomi (rBergomi) model of Bayer, Friz, and Gatheral (2016).

    .. math::
        S_t = S_0 \exp\left( \int_0^t \sqrt{v_s} dW_s - \frac{1}{2} \int_0^t v_s ds \right)
        v_t = \xi_0 \exp\left( \eta \sqrt{2H} \tilde{W}_t^H - \frac{1}{2} \eta^2 t^{2H} \right)
    where :math:`\tilde{W}_t^H = \int_0^t (t - s)^{H - 1/2} dZ_s` and :math:`d\langle W, Z \rangle_t = \rho dt`.
    """

    def __init__(
        self,
        hurst: float = 0.1,
        eta: float = 1.9,
        rho: float = -0.7,
        xi0: float = 0.04,
        s0: float = 100.0,
    ) -> None:
        self.H = float(hurst)
        self.eta = float(eta)
        self.rho = float(rho)
        self.xi0 = float(xi0)
        self.s0 = float(s0)
        self.kernel = FractionalKernel(hurst=hurst)

    def theoretical_short_time_skew(self, tau: float) -> float:
        r"""Evaluates leading-order short-time ATM volatility skew:

        .. math::
            \left| \frac{\partial \sigma_{BS}}{\partial k} \right|_{k=0} \approx \frac{|\rho| \eta}{2} \frac{\sqrt{2H}}{H + 1/2} \tau^{H - 1/2}
        """
        if tau <= 0.0:
            return 0.0
        c = (abs(self.rho) * self.eta / 2.0) * (np.sqrt(2.0 * self.H) / (self.H + 0.5))
        return float(c * (tau**(self.H - 0.5)))

    def simulate_paths(
        self,
        t_mat: float = 1.0,
        n_steps: int = 100,
        n_paths: int = 1000,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        r"""Simulates joint paths of instantaneous variance :math:`v_t` and spot price :math:`S_t`."""
        rng = np.random.default_rng(seed)
        dt = t_mat / n_steps
        time_grid = np.linspace(0.0, t_mat, n_steps + 1)

        # 1. Simulate fractional driver W_tilde_H using Cholesky
        fbm_res = self.kernel.sample_fbm_cholesky(t_mat=t_mat, n_steps=n_steps, n_paths=n_paths, seed=seed)
        # Note: B_t^H ~ sqrt(2H)/(H+1/2) W_tilde_H scaling
        w_tilde = fbm_res["paths"]  # Shape (n_paths, n_steps + 1)

        # 2. Instantaneous variance paths
        # v_t = xi0 * exp(eta * sqrt(2H) * W_tilde_t - 0.5 * eta^2 * t^{2H})
        t_pow = time_grid**(2.0 * self.H)
        exponent = self.eta * np.sqrt(2.0 * self.H) * w_tilde - 0.5 * (self.eta**2) * t_pow
        v_paths = self.xi0 * np.exp(exponent)

        # 3. Correlated Brownian increments for spot price
        # dW_t = rho dZ_t + sqrt(1 - rho^2) dZ_perp_t
        # where dZ_t is approximated by differences of driving BM
        dz = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
        dz_perp = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
        dw = self.rho * dz + np.sqrt(1.0 - self.rho**2) * dz_perp

        # 4. Spot price integration
        s_paths = np.zeros((n_paths, n_steps + 1))
        s_paths[:, 0] = self.s0
        log_s = np.log(self.s0) * np.ones(n_paths)

        for s in range(n_steps):
            v_cur = v_paths[:, s]
            sqrt_v = np.sqrt(v_cur)
            d_log_s = -0.5 * v_cur * dt + sqrt_v * dw[:, s]
            log_s += d_log_s
            s_paths[:, s + 1] = np.exp(log_s)

        return {
            "spot_paths": s_paths,
            "variance_paths": v_paths,
            "time_grid": time_grid,
        }
