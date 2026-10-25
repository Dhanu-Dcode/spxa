"""Two-Factor Cox-Ingersoll-Ross (CIR2++) Short-Rate Process.

Two independent Cox-Ingersoll-Ross diffusion factors driving an additive short rate:
    r(t) = x(t) + y(t) + phi(t)
    dx(t) = k_x * (theta_x - x(t)) dt + sigma_x * sqrt(x(t)) dW_x(t)
    dy(t) = k_y * (theta_y - y(t)) dt + sigma_y * sqrt(y(t)) dW_y(t)

References:
    Cox, J. C., Ingersoll, J. E., & Ross, S. A. (1985). A theory of the term structure of interest rates.
        Econometrica, 53(2), 385-407.
    Brigo, D., & Mercurio, F. (2006). Interest Rate Models - Theory and Practice. Springer, Sec. 4.3.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties


class TwoFactorCIRProcess(Process):
    r"""Two-Factor Cox-Ingersoll-Ross (CIR2) short rate model.

    .. math::
        r_t = x_t + y_t + \phi
        dx_t = \kappa_x (\theta_x - x_t) dt + \sigma_x \sqrt{x_t} dW_t^x
        dy_t = \kappa_y (\theta_y - y_t) dt + \sigma_y \sqrt{y_t} dW_t^y
    """

    def __init__(
        self,
        k_x: float,
        theta_x: float,
        sigma_x: float,
        k_y: float,
        theta_y: float,
        sigma_y: float,
        x0: float = 0.02,
        y0: float = 0.02,
        phi: float = 0.0,
    ) -> None:
        if k_x <= 0 or theta_x <= 0 or sigma_x <= 0:
            raise ValueError(f"Factor X parameters must be positive: k={k_x}, theta={theta_x}, sigma={sigma_x}")
        if k_y <= 0 or theta_y <= 0 or sigma_y <= 0:
            raise ValueError(f"Factor Y parameters must be positive: k={k_y}, theta={theta_y}, sigma={sigma_y}")
        if x0 < 0 or y0 < 0:
            raise ValueError(f"Initial factor values must be non-negative: x0={x0}, y0={y0}")

        self.k_x = float(k_x)
        self.theta_x = float(theta_x)
        self.sigma_x = float(sigma_x)
        self.k_y = float(k_y)
        self.theta_y = float(theta_y)
        self.sigma_y = float(sigma_y)
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.phi = float(phi)
        self.r0 = self.x0 + self.y0 + self.phi
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            notes=["Two-Factor CIR short-rate process (CIR2++)"],
        )

    def _triplet(self):
        raise ExactnessError(
            "TwoFactorCIRProcess has state-dependent diffusion coefficients and cannot be represented by a Levy triplet."
        )

    @property
    def feller_ratio_x(self) -> float:
        """Feller ratio 2*k_x*theta_x / sigma_x^2."""
        return 2.0 * self.k_x * self.theta_x / (self.sigma_x**2)

    @property
    def feller_ratio_y(self) -> float:
        """Feller ratio 2*k_y*theta_y / sigma_y^2."""
        return 2.0 * self.k_y * self.theta_y / (self.sigma_y**2)

    def mean_x(self, t: float) -> float:
        """Conditional mean E[x(t) | x(0)]."""
        return self.theta_x + (self.x0 - self.theta_x) * math.exp(-self.k_x * t)

    def mean_y(self, t: float) -> float:
        """Conditional mean E[y(t) | y(0)]."""
        return self.theta_y + (self.y0 - self.theta_y) * math.exp(-self.k_y * t)

    def mean(self, t: float) -> float:
        """Conditional mean of short rate E[r(t)]."""
        return self.mean_x(t) + self.mean_y(t) + self.phi

    def var_x(self, t: float) -> float:
        """Conditional variance Var[x(t) | x(0)]."""
        e_kt = math.exp(-self.k_x * t)
        term1 = self.x0 * (self.sigma_x**2 / self.k_x) * (e_kt - e_kt**2)
        term2 = self.theta_x * (self.sigma_x**2 / (2.0 * self.k_x)) * ((1.0 - e_kt) ** 2)
        return term1 + term2

    def var_y(self, t: float) -> float:
        """Conditional variance Var[y(t) | y(0)]."""
        e_kt = math.exp(-self.k_y * t)
        term1 = self.y0 * (self.sigma_y**2 / self.k_y) * (e_kt - e_kt**2)
        term2 = self.theta_y * (self.sigma_y**2 / (2.0 * self.k_y)) * ((1.0 - e_kt) ** 2)
        return term1 + term2

    def variance(self, t: float) -> float:
        """Conditional variance Var[r(t)]."""
        return self.var_x(t) + self.var_y(t)

    def stationary_mean(self) -> float:
        """Stationary unconditional mean of short rate."""
        return self.theta_x + self.theta_y + self.phi

    def stationary_variance(self) -> float:
        """Stationary unconditional variance of short rate."""
        return (self.theta_x * self.sigma_x**2) / (2.0 * self.k_x) + (
            self.theta_y * self.sigma_y**2
        ) / (2.0 * self.k_y)

    def _cir_bond_coeffs(
        self, tau: float, k: float, theta: float, sigma: float
    ) -> tuple[float, float]:
        """Compute A(tau), B(tau) for 1D CIR bond pricing P(t, T) = A(tau) exp(-B(tau)*x)."""
        gamma = math.sqrt(k**2 + 2.0 * sigma**2)
        exp_gt = math.exp(gamma * tau)
        denom = (gamma + k) * (exp_gt - 1.0) + 2.0 * gamma
        b = 2.0 * (exp_gt - 1.0) / denom
        num_a = 2.0 * gamma * math.exp((k + gamma) * tau / 2.0)
        power = 2.0 * k * theta / (sigma**2)
        a = (num_a / denom) ** power
        return a, b

    def zero_coupon_bond(self, tau: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Closed-form zero-coupon bond price P(t, t+tau)."""
        is_scalar = np.isscalar(tau)
        taus = np.atleast_1d(tau).astype(float)
        prices = np.zeros_like(taus)

        for i, t in enumerate(taus):
            if t <= 0.0:
                prices[i] = 1.0
                continue
            a_x, b_x = self._cir_bond_coeffs(t, self.k_x, self.theta_x, self.sigma_x)
            a_y, b_y = self._cir_bond_coeffs(t, self.k_y, self.theta_y, self.sigma_y)
            p = a_x * a_y * math.exp(-b_x * self.x0 - b_y * self.y0 - self.phi * t)
            prices[i] = p

        return float(prices[0]) if is_scalar else prices

    def yield_curve(self, tau: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Continuously compounded zero-coupon yield y(t, t+tau) = -ln(P(t, t+tau))/tau."""
        is_scalar = np.isscalar(tau)
        taus = np.atleast_1d(tau).astype(float)
        p = np.atleast_1d(self.zero_coupon_bond(taus))
        y = np.zeros_like(taus)
        pos = taus > 0.0
        y[pos] = -np.log(p[pos]) / taus[pos]
        y[~pos] = self.r0
        return float(y[0]) if is_scalar else y

    def simulate(
        self,
        t_max: float,
        num_steps: int,
        num_paths: int = 1,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate short-rate paths using exact non-central chi-square increments for each factor."""
        if seed is not None:
            np.random.seed(seed)

        dt = float(t_max) / num_steps
        x_paths = np.zeros((num_paths, num_steps + 1))
        y_paths = np.zeros((num_paths, num_steps + 1))
        x_paths[:, 0] = self.x0
        y_paths[:, 0] = self.y0

        # Exact transition factors for x
        d_x = 4.0 * self.k_x * self.theta_x / (self.sigma_x**2)
        c_x = (self.sigma_x**2 / (4.0 * self.k_x)) * (1.0 - math.exp(-self.k_x * dt))
        exp_kx = math.exp(-self.k_x * dt)

        # Exact transition factors for y
        d_y = 4.0 * self.k_y * self.theta_y / (self.sigma_y**2)
        c_y = (self.sigma_y**2 / (4.0 * self.k_y)) * (1.0 - math.exp(-self.k_y * dt))
        exp_ky = math.exp(-self.k_y * dt)

        for step in range(num_steps):
            x_prev = x_paths[:, step]
            lambda_x = (4.0 * self.k_x * x_prev * exp_kx) / (
                self.sigma_x**2 * (1.0 - exp_kx)
            )
            # Sample non-central chi-squared
            x_next = c_x * np.random.noncentral_chisquare(df=d_x, nonc=np.maximum(0.0, lambda_x))
            x_paths[:, step + 1] = np.maximum(0.0, x_next)

            y_prev = y_paths[:, step]
            lambda_y = (4.0 * self.k_y * y_prev * exp_ky) / (
                self.sigma_y**2 * (1.0 - exp_ky)
            )
            y_next = c_y * np.random.noncentral_chisquare(df=d_y, nonc=np.maximum(0.0, lambda_y))
            y_paths[:, step + 1] = np.maximum(0.0, y_next)

        return x_paths + y_paths + self.phi


# Aliases
TwoFactorCIR = TwoFactorCIRProcess
CIR2PlusPlus = TwoFactorCIRProcess
