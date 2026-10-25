"""Double Heston Two-Factor Stochastic Volatility Model.

The Double Heston model (Christoffersen, Heston, & Jacobs, 2009) introduces two independent
stochastic variance factors with separate mean-reversion speeds, volatility-of-volatilities,
and leverage correlations to capture both short-term volatility bursts and long-term skew.

Dynamics:
    dS(t) = (r - q) * S(t) dt + sqrt(V_1(t)) * S(t) dW_1^S(t) + sqrt(V_2(t)) * S(t) dW_2^S(t)
    dV_1(t) = k_1 * (theta_1 - V_1(t)) dt + sigma_1 * sqrt(V_1(t)) dW_1^V(t)
    dV_2(t) = k_2 * (theta_2 - V_2(t)) dt + sigma_2 * sqrt(V_2(t)) dW_2^V(t)
    Corr(dW_1^S, dW_1^V) = rho_1, Corr(dW_2^S, dW_2^V) = rho_2

References:
    Christoffersen, P., Heston, S., & Jacobs, K. (2009). The shape and term structure of the index
        normal implied volatility curve. European Financial Management, 15(4), 748-777.
    Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007). The little Heston trap.
        Wilmott Magazine, 83-92.
"""

from __future__ import annotations

import cmath
import math
from typing import Optional, Union

import numpy as np
from scipy import integrate

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties


class DoubleHestonProcess(Process):
    r"""Two-Factor Double Heston stochastic volatility model.

    .. math::
        dS_t = (r - q) S_t dt + \sqrt{V_{1,t}} S_t dW_{1,t}^S + \sqrt{V_{2,t}} S_t dW_{2,t}^S
        dV_{j,t} = \kappa_j (\theta_j - V_{j,t}) dt + \sigma_j \sqrt{V_{j,t}} dW_{j,t}^V, \quad j \in \{1, 2\}
    """

    def __init__(
        self,
        s0: float,
        v1_0: float,
        v2_0: float,
        k1: float,
        theta1: float,
        sigma1: float,
        rho1: float,
        k2: float,
        theta2: float,
        sigma2: float,
        rho2: float,
        r: float = 0.0,
        q: float = 0.0,
    ) -> None:
        if s0 <= 0:
            raise ValueError(f"Initial spot price s0 must be positive: {s0}")
        if v1_0 < 0 or v2_0 < 0:
            raise ValueError(f"Initial variances must be non-negative: v1_0={v1_0}, v2_0={v2_0}")
        if k1 <= 0 or theta1 <= 0 or sigma1 <= 0:
            raise ValueError("Factor 1 parameters k1, theta1, sigma1 must be positive")
        if k2 <= 0 or theta2 <= 0 or sigma2 <= 0:
            raise ValueError("Factor 2 parameters k2, theta2, sigma2 must be positive")
        if not (-1.0 <= rho1 <= 1.0) or not (-1.0 <= rho2 <= 1.0):
            raise ValueError("Correlation coefficients rho1 and rho2 must be in [-1, 1]")

        self.s0 = float(s0)
        self.v1_0 = float(v1_0)
        self.v2_0 = float(v2_0)
        self.k1 = float(k1)
        self.theta1 = float(theta1)
        self.sigma1 = float(sigma1)
        self.rho1 = float(rho1)
        self.k2 = float(k2)
        self.theta2 = float(theta2)
        self.sigma2 = float(sigma2)
        self.rho2 = float(rho2)
        self.r = float(r)
        self.q = float(q)
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _triplet(self):
        raise ExactnessError(
            "DoubleHestonProcess has stochastic volatility and cannot be represented by a Levy triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=(self.r == self.q),
            has_finite_variance=True,
            has_finite_mean=True,
            notes=["Double Heston two-factor stochastic volatility model"],
        )

    @property
    def total_initial_variance(self) -> float:
        """Total initial variance V_0 = V_{1,0} + V_{2,0}."""
        return self.v1_0 + self.v2_0

    @property
    def total_long_run_variance(self) -> float:
        """Total long-run stationary variance theta = theta_1 + theta_2."""
        return self.theta1 + self.theta2

    def _heston_factor_riccati(
        self, u: complex, t: float, k: float, theta: float, sigma: float, rho: float, v0: float
    ) -> complex:
        """Compute the Little Heston trap (Albrecher et al.) continuous Riccati component."""
        i = 1j
        d = cmath.sqrt((rho * sigma * i * u - k) ** 2 + sigma**2 * (i * u + u**2))
        g = (k - rho * sigma * i * u - d) / (k - rho * sigma * i * u + d)
        exp_dt = cmath.exp(-d * t)

        c = (k * theta / (sigma**2)) * (
            (k - rho * sigma * i * u - d) * t - 2.0 * cmath.log((1.0 - g * exp_dt) / (1.0 - g))
        )
        d_val = ((k - rho * sigma * i * u - d) / (sigma**2)) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
        return cmath.exp(c + d_val * v0)

    def characteristic_function(self, u: Union[float, complex], t: float) -> complex:
        r"""Characteristic function of log-asset price ln(S_t)."""
        u_c = complex(u)
        drift = 1j * u_c * (math.log(self.s0) + (self.r - self.q) * t)
        phi1 = self._heston_factor_riccati(
            u_c, t, self.k1, self.theta1, self.sigma1, self.rho1, self.v1_0
        )
        phi2 = self._heston_factor_riccati(
            u_c, t, self.k2, self.theta2, self.sigma2, self.rho2, self.v2_0
        )
        return cmath.exp(drift) * phi1 * phi2

    def mean_spot(self, t: float) -> float:
        """Expected spot price E[S(t)] = S_0 * exp((r - q) * t)."""
        return self.s0 * math.exp((self.r - self.q) * t)

    def mean_variance(self, t: float) -> float:
        """Expected total instantaneous variance E[V_1(t) + V_2(t)]."""
        m1 = self.theta1 + (self.v1_0 - self.theta1) * math.exp(-self.k1 * t)
        m2 = self.theta2 + (self.v2_0 - self.theta2) * math.exp(-self.k2 * t)
        return m1 + m2

    def european_call(self, strike: float, maturity: float) -> float:
        r"""Price European call option via Heston / Gil-Pelaez inversion."""
        if strike <= 0 or maturity <= 0:
            raise ValueError("Strike and maturity must be strictly positive")

        t = maturity
        k = strike
        s0 = self.s0
        r, q = self.r, self.q
        ln_k_s0 = math.log(k / s0)

        def char_return(u: complex) -> complex:
            drift = 1j * u * (r - q) * t
            phi1 = self._heston_factor_riccati(
                u, t, self.k1, self.theta1, self.sigma1, self.rho1, self.v1_0
            )
            phi2 = self._heston_factor_riccati(
                u, t, self.k2, self.theta2, self.sigma2, self.rho2, self.v2_0
            )
            return cmath.exp(drift) * phi1 * phi2

        def integrand1(u: float) -> float:
            if u < 1e-12:
                return 0.0
            cf1 = math.exp(-(r - q) * t) * char_return(u - 1j)
            val = cmath.exp(-1j * u * ln_k_s0) * cf1 / (1j * u)
            return float(val.real)

        def integrand2(u: float) -> float:
            if u < 1e-12:
                return 0.0
            cf2 = char_return(u)
            val = cmath.exp(-1j * u * ln_k_s0) * cf2 / (1j * u)
            return float(val.real)

        p1_int, _ = integrate.quad(integrand1, 1e-8, 100.0, limit=200)
        p2_int, _ = integrate.quad(integrand2, 1e-8, 100.0, limit=200)

        p1 = float(np.clip(0.5 + (1.0 / math.pi) * p1_int, 0.0, 1.0))
        p2 = float(np.clip(0.5 + (1.0 / math.pi) * p2_int, 0.0, 1.0))

        price = s0 * math.exp(-q * t) * p1 - k * math.exp(-r * t) * p2
        intrinsic = max(0.0, s0 * math.exp(-q * t) - k * math.exp(-r * t))
        return float(max(price, intrinsic))

    def european_put(self, strike: float, maturity: float) -> float:
        """Price European put option via exact put-call parity."""
        call = self.european_call(strike, maturity)
        parity_forward = self.s0 * math.exp(-self.q * maturity) - strike * math.exp(
            -self.r * maturity
        )
        put = call - parity_forward
        return float(max(0.0, put))

    def simulate(
        self,
        t_max: float,
        num_steps: int,
        num_paths: int = 1,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate spot and variance paths using full-truncation Euler-Maruyama discretization."""
        if seed is not None:
            np.random.seed(seed)

        dt = float(t_max) / num_steps
        sqrt_dt = math.sqrt(dt)

        s_paths = np.zeros((num_paths, num_steps + 1))
        v1_paths = np.zeros((num_paths, num_steps + 1))
        v2_paths = np.zeros((num_paths, num_steps + 1))

        s_paths[:, 0] = self.s0
        v1_paths[:, 0] = self.v1_0
        v2_paths[:, 0] = self.v2_0

        rho1 = self.rho1
        rho2 = self.rho2
        k1, theta1, sigma1 = self.k1, self.theta1, self.sigma1
        k2, theta2, sigma2 = self.k2, self.theta2, self.sigma2
        r, q = self.r, self.q

        for step in range(num_steps):
            s_curr = s_paths[:, step]
            v1_curr = np.maximum(0.0, v1_paths[:, step])
            v2_curr = np.maximum(0.0, v2_paths[:, step])

            # Correlated Gaussian increments for Factor 1
            z1_v = np.random.normal(0.0, 1.0, num_paths)
            z1_perp = np.random.normal(0.0, 1.0, num_paths)
            z1_s = rho1 * z1_v + math.sqrt(1.0 - rho1**2) * z1_perp

            # Correlated Gaussian increments for Factor 2
            z2_v = np.random.normal(0.0, 1.0, num_paths)
            z2_perp = np.random.normal(0.0, 1.0, num_paths)
            z2_s = rho2 * z2_v + math.sqrt(1.0 - rho2**2) * z2_perp

            # Variance updates with full truncation
            v1_next = (
                v1_paths[:, step]
                + k1 * (theta1 - v1_curr) * dt
                + sigma1 * np.sqrt(v1_curr) * sqrt_dt * z1_v
            )
            v2_next = (
                v2_paths[:, step]
                + k2 * (theta2 - v2_curr) * dt
                + sigma2 * np.sqrt(v2_curr) * sqrt_dt * z2_v
            )
            v1_paths[:, step + 1] = np.maximum(0.0, v1_next)
            v2_paths[:, step + 1] = np.maximum(0.0, v2_next)

            # Spot update
            total_v = v1_curr + v2_curr
            log_s_next = (
                np.log(s_curr)
                + (r - q - 0.5 * total_v) * dt
                + (np.sqrt(v1_curr) * z1_s + np.sqrt(v2_curr) * z2_s) * sqrt_dt
            )
            s_paths[:, step + 1] = np.exp(log_s_next)

        return s_paths, v1_paths, v2_paths


# Aliases
DoubleHeston = DoubleHestonProcess
TwoFactorHeston = DoubleHestonProcess
