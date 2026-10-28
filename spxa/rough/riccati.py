"""
Fractional Adams-Bashforth-Moulton Riccati solver and rough Heston characteristic functions.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.special import gamma


class FractionalRiccatiSolver:
    r"""Solves fractional Riccati Volterra integral equations using the Adams-Bashforth-Moulton (ABM) scheme:

    .. math::
        h(t) = \int_0^t \frac{(t - s)^{\alpha - 1}}{\Gamma(\alpha)} F(u, h(s)) ds
    where :math:`\alpha = H + 1/2 \in (0.5, 1)`.
    """

    def __init__(
        self,
        hurst: float = 0.1,
        lambda_param: float = 1.0,
        rho: float = -0.7,
        nu: float = 0.3,
    ) -> None:
        if not (0.0 < hurst < 1.0):
            raise ValueError(f"Hurst exponent H must be in (0, 1), got {hurst}")
        self.H = float(hurst)
        self.alpha = self.H + 0.5  # In (0.5, 1.0)
        self.lam = float(lambda_param)
        self.rho = float(rho)
        self.nu = float(nu)

    def riccati_rhs(self, u: complex, h: complex) -> complex:
        r"""Evaluates :math:`F(u, h) = -\frac{1}{2}(u^2 + i u) + (\lambda - i \rho \nu u) h + \frac{1}{2}\nu^2 h^2`."""
        term1 = -0.5 * (u**2 + 1j * u)
        term2 = (self.lam - 1j * self.rho * self.nu * u) * h
        term3 = 0.5 * (self.nu**2) * (h**2)
        return term1 + term2 + term3

    def solve(
        self,
        u: complex,
        t_mat: float = 1.0,
        n_steps: int = 100,
    ) -> Dict[str, Any]:
        r"""Solves the fractional Riccati equation via fractional ABM predictor-corrector algorithm."""
        dt = t_mat / n_steps
        alpha = self.alpha
        t_grid = np.linspace(0.0, t_mat, n_steps + 1)

        h_vals = np.zeros(n_steps + 1, dtype=complex)
        f_vals = np.zeros(n_steps + 1, dtype=complex)

        # Initial condition: h(0) = 0
        h_vals[0] = 0.0 + 0.0j
        f_vals[0] = self.riccati_rhs(u, h_vals[0])

        c_pred = (dt**alpha) / gamma(alpha + 1.0)
        c_corr = (dt**alpha) / gamma(alpha + 2.0)

        for n in range(1, n_steps + 1):
            # 1. Predictor step (Adams-Bashforth)
            j_arr = np.arange(n)
            b_jn = (n - j_arr)**alpha - (n - 1 - j_arr)**alpha
            h_pred = c_pred * np.sum(b_jn * f_vals[:n])

            f_pred = self.riccati_rhs(u, h_pred)

            # 2. Corrector step (Adams-Moulton)
            a_0n = (n - 1.0)**(alpha + 1.0) - (n - 1.0 - alpha) * (n**alpha)
            if n > 1:
                j_mid = np.arange(1, n)
                a_jn = (n - j_mid + 1.0)**(alpha + 1.0) - 2.0 * ((n - j_mid)**(alpha + 1.0)) + (n - j_mid - 1.0)**(alpha + 1.0)
                sum_mid = np.sum(a_jn * f_vals[1:n])
            else:
                sum_mid = 0.0 + 0.0j

            h_corr = c_corr * (f_pred + a_0n * f_vals[0] + sum_mid)
            h_vals[n] = h_corr
            f_vals[n] = self.riccati_rhs(u, h_corr)

        return {
            "h_vals": h_vals,
            "t_grid": t_grid,
            "final_h": h_vals[-1],
        }

    def rough_heston_char_func(
        self,
        u: complex,
        t_mat: float = 1.0,
        v0: float = 0.04,
        theta: float = 0.04,
        n_steps: int = 100,
    ) -> complex:
        r"""Evaluates rough Heston characteristic function :math:`\mathbb{E}[e^{i u X_T}]`."""
        sol = self.solve(u, t_mat=t_mat, n_steps=n_steps)
        h_vals = sol["h_vals"]
        t_grid = sol["t_grid"]
        dt = t_mat / n_steps

        # Fractional integral I^{H+1/2} h(T)
        alpha = self.alpha
        weights = ((t_mat - t_grid[:-1])**(alpha - 1.0)) / gamma(alpha)
        # Handle singularity by replacing last weight with integrated weight
        weights[-1] = (dt**(alpha - 1.0)) / gamma(alpha)
        frac_int = np.sum(weights * h_vals[:-1]) * dt

        # Standard Riemann integral theta * lambda * int_0^T h(s) ds
        riemann_int = np.sum(h_vals) * dt

        log_cf = v0 * frac_int + theta * self.lam * riemann_int
        return complex(np.exp(log_cf))
