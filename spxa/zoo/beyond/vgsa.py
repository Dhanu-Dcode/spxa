r"""Variance Gamma with Stochastic Arrival (VGSA) Process.

The VGSA model (Carr, Geman, Madan, & Yor, 2003) extends the Variance Gamma process by
subordinating it with a stochastic clock Y(t) = \int_0^t y(s) ds, where the clock rate y(s)
follows a mean-reverting Cox-Ingersoll-Ross (CIR) diffusion:
    dy(t) = k * (eta - y(t)) dt + lambda_v * sqrt(y(t)) dW(t)
    X(t) = VG(Y(t); sigma, nu, theta)

References:
    Carr, P., Geman, H., Madan, D. B., & Yor, M. (2003). Stochastic volatility for Lévy processes.
        Mathematical Finance, 13(3), 345-382.
    Schoutens, W. (2003). Lévy Processes in Finance: Pricing Financial Derivatives. Wiley.
"""

from __future__ import annotations

import cmath
import math
from typing import Optional, Union

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties


class VGSAProcess(Process):
    r"""Variance Gamma with Stochastic Arrival (VGSA) process.

    .. math::
        X_t = \text{VG}(Y_t; \sigma, \nu, \theta)
        dy_t = \kappa (\eta - y_t) dt + \lambda \sqrt{y_t} dW_t, \quad Y_t = \int_0^t y_s ds
    """

    def __init__(
        self,
        sigma: float,
        nu: float,
        theta: float,
        k: float,
        eta: float,
        lambd: float,
        y0: float = 1.0,
    ) -> None:
        if sigma <= 0 or nu <= 0:
            raise ValueError(f"VG parameters sigma and nu must be positive: sigma={sigma}, nu={nu}")
        if k <= 0 or eta <= 0 or lambd <= 0:
            raise ValueError(f"CIR clock parameters k, eta, lambd must be positive: k={k}, eta={eta}, lambd={lambd}")
        if y0 <= 0:
            raise ValueError(f"Initial clock rate y0 must be positive: y0={y0}")

        self.sigma = float(sigma)
        self.nu = float(nu)
        self.theta = float(theta)
        self.k = float(k)
        self.eta = float(eta)
        self.lambd = float(lambd)
        self.y0 = float(y0)
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _triplet(self):
        raise ExactnessError(
            "VGSAProcess has stochastic time-change and cannot be represented by a standard Levy triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=(self.theta == 0.0),
            has_finite_variance=True,
            has_finite_mean=True,
            notes=["Variance Gamma with CIR stochastic arrival rate (VGSA)"],
        )

    def mean_integrated_clock(self, t: float) -> float:
        """Expected integrated time E[Y(t)] = E[int_0^t y(s) ds]."""
        if t <= 0:
            return 0.0
        return self.eta * t + (self.y0 - self.eta) * (1.0 - math.exp(-self.k * t)) / self.k

    def var_integrated_clock(self, t: float) -> float:
        """Variance of integrated time Var[Y(t)]."""
        if t <= 0:
            return 0.0
        k = self.k
        lam = self.lambd
        eta = self.eta
        y0 = self.y0
        e_kt = math.exp(-k * t)

        term1 = (2.0 * eta * lam**2 / (k**3)) * (k * t - 2.0 * (1.0 - e_kt) + 0.5 * (1.0 - e_kt**2))
        term2 = (y0 * lam**2 / (k**2)) * ((1.0 - e_kt) ** 2)
        return float(max(0.0, term1 + term2))

    def mean(self, t: float) -> float:
        """Mean E[X(t)] = theta * E[Y(t)]."""
        return self.theta * self.mean_integrated_clock(t)

    def variance(self, t: float) -> float:
        """Variance Var[X(t)] = (sigma^2 + nu * theta^2) * E[Y(t)] + theta^2 * Var[Y(t)]."""
        e_y = self.mean_integrated_clock(t)
        var_y = self.var_integrated_clock(t)
        return (self.sigma**2 + self.nu * self.theta**2) * e_y + (self.theta**2) * var_y

    def vg_characteristic_exponent(self, u: complex) -> complex:
        r"""Variance Gamma characteristic exponent \psi(u) = - (1/nu) ln(1 - i u theta nu + 0.5 sigma^2 u^2 nu)."""
        term = 1.0 - 1j * u * self.theta * self.nu + 0.5 * (self.sigma**2) * (u**2) * self.nu
        return -(1.0 / self.nu) * cmath.log(term)

    def characteristic_function(self, u: Union[float, complex], t: float) -> complex:
        r"""Characteristic function E[exp(i u X_t)] via CIR Laplace transform of integrated time."""
        u_c = complex(u)
        psi_vg = self.vg_characteristic_exponent(u_c)
        gamma = -psi_vg

        if abs(gamma) < 1e-15:
            return 1.0 + 0j

        k = self.k
        lam = self.lambd
        eta = self.eta
        y0 = self.y0

        # CIR bond formula: E[exp(-gamma int_0^t y_s ds)]
        d = cmath.sqrt(k**2 + 2.0 * (lam**2) * gamma)
        exp_dt = cmath.exp(-d * t)
        denom = (d + k) + (d - k) * exp_dt

        b = 2.0 * gamma * (1.0 - exp_dt) / denom
        a = ((2.0 * d * cmath.exp(0.5 * (k - d) * t)) / denom) ** (2.0 * k * eta / (lam**2))

        return a * cmath.exp(-b * y0)

    def simulate(
        self,
        t_max: float,
        num_steps: int,
        num_paths: int = 1,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate VGSA sample paths using CIR rate clock and Gamma-subordinated BM."""
        if seed is not None:
            np.random.seed(seed)

        dt = float(t_max) / num_steps
        y_curr = np.full(num_paths, self.y0)
        x_paths = np.zeros((num_paths, num_steps + 1))

        k, eta, lam = self.k, self.eta, self.lambd
        sigma, nu, theta = self.sigma, self.nu, self.theta

        # Exact CIR transition factors
        d = 4.0 * k * eta / (lam**2)
        exp_kdt = math.exp(-k * dt)
        c = (lam**2 / (4.0 * k)) * (1.0 - exp_kdt)

        for step in range(num_steps):
            # Sample CIR clock rate at next step
            nonc = (4.0 * k * y_curr * exp_kdt) / (lam**2 * (1.0 - exp_kdt))
            y_next = c * np.random.noncentral_chisquare(df=d, nonc=np.maximum(0.0, nonc))
            y_next = np.maximum(0.0, y_next)

            # Integrated time increment by trapezoidal rule
            delta_y = 0.5 * (y_curr + y_next) * dt

            # Subordinated Variance Gamma increment for delta_y
            # Gamma subordinator increment tau ~ Gamma(shape=delta_y/nu, scale=nu)
            shape = np.maximum(1e-8, delta_y / nu)
            tau = np.random.gamma(shape=shape, scale=nu)

            # Gaussian increment given tau: N(theta * tau, sigma^2 * tau)
            z = np.random.normal(0.0, 1.0, num_paths)
            dx = theta * tau + sigma * np.sqrt(tau) * z

            x_paths[:, step + 1] = x_paths[:, step] + dx
            y_curr = y_next

        return x_paths


# Aliases
VGSA = VGSAProcess
VarianceGammaStochasticArrival = VGSAProcess
