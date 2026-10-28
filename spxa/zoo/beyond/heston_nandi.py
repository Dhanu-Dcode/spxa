"""Heston-Nandi GARCH Diffusion Process.

Continuous-time diffusion limit of the Heston-Nandi (2000) affine GARCH option pricing model.
The model incorporates asymmetric leverage through the parameter gamma and admits an exact
closed-form moment generating function and characteristic function for log-returns.

Dynamics:
    dS(t) = r * S(t) dt + sqrt(V(t)) * S(t) dW_S(t)
    dV(t) = (k * (theta - V(t)) + alpha * gamma^2 * V(t)) dt + alpha * sqrt(V(t)) dW_V(t)
    Corr(dW_S, dW_V) = -1.0 (or parameterized rho)

References:
    Heston, S. L., & Nandi, S. (2000). A closed-form GARCH option valuation model.
        The Review of Financial Studies, 13(3), 585-625.
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


class HestonNandiProcess(Process):
    r"""Heston-Nandi continuous-time GARCH diffusion model.

    .. math::
        dS_t = r S_t dt + \sqrt{V_t} S_t dW_t^S
        dV_t = \kappa (\theta - V_t) dt + \alpha \sqrt{V_t} dW_t^V
    """

    def __init__(
        self,
        s0: float,
        v0: float,
        k: float,
        theta: float,
        alpha: float,
        gamma: float,
        r: float = 0.0,
        rho: float = -1.0,
    ) -> None:
        if s0 <= 0:
            raise ValueError(f"Spot price s0 must be positive: {s0}")
        if v0 < 0:
            raise ValueError(f"Initial variance v0 must be non-negative: {v0}")
        if k <= 0 or theta <= 0 or alpha <= 0:
            raise ValueError(
                f"Parameters k, theta, alpha must be positive: k={k}, theta={theta}, alpha={alpha}"
            )
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"Correlation rho must be in [-1, 1]: {rho}")

        self.s0 = float(s0)
        self.v0 = float(v0)
        self.k = float(k)
        self.theta = float(theta)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.r = float(r)
        self.rho = float(rho)
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _triplet(self):
        raise ExactnessError(
            "HestonNandiProcess has stochastic volatility and cannot be represented by a Levy triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=(self.r == 0.0),
            has_finite_variance=True,
            has_finite_mean=True,
            notes=["Heston-Nandi continuous GARCH diffusion model"],
        )

    @property
    def feller_ratio(self) -> float:
        """Feller condition ratio 2*k*theta / alpha^2."""
        return 2.0 * self.k * self.theta / (self.alpha**2)

    def mean_spot(self, t: float) -> float:
        """Expected spot price E[S(t)] = S_0 * exp(r * t)."""
        return self.s0 * math.exp(self.r * t)

    def mean_variance(self, t: float) -> float:
        """Conditional expected variance E[V(t) | V(0)]."""
        return self.theta + (self.v0 - self.theta) * math.exp(-self.k * t)

    def characteristic_function(self, u: Union[float, complex], t: float) -> complex:
        r"""Characteristic function of log-price ln(S_t)."""
        u_c = complex(u)
        i = 1j

        # Affine Riccati ODE coefficients
        # dV = (k*theta - (k - alpha*gamma*rho)*V) dt + alpha*sqrt(V) dW
        a = self.alpha
        k_eff = self.k - self.alpha * self.gamma * self.rho
        d = cmath.sqrt((self.rho * a * i * u_c - k_eff) ** 2 + a**2 * (i * u_c + u_c**2))
        g = (k_eff - self.rho * a * i * u_c - d) / (k_eff - self.rho * a * i * u_c + d)
        exp_dt = cmath.exp(-d * t)

        c = (self.k * self.theta / (a**2)) * (
            (k_eff - self.rho * a * i * u_c - d) * t
            - 2.0 * cmath.log((1.0 - g * exp_dt) / (1.0 - g))
        )
        d_val = ((k_eff - self.rho * a * i * u_c - d) / (a**2)) * (
            (1.0 - exp_dt) / (1.0 - g * exp_dt)
        )

        drift = i * u_c * (math.log(self.s0) + self.r * t)
        return cmath.exp(drift + c + d_val * self.v0)

    def european_call(self, strike: float, maturity: float) -> float:
        """Price European call option via Heston / Gil-Pelaez inversion."""
        if strike <= 0 or maturity <= 0:
            raise ValueError("Strike and maturity must be strictly positive")

        t = maturity
        k = strike
        s0 = self.s0
        r = self.r
        ln_k_s0 = math.log(k / s0)

        def char_return(u: complex) -> complex:
            drift = 1j * u * r * t
            a = self.alpha
            k_eff = self.k - self.alpha * self.gamma * self.rho
            d = cmath.sqrt((self.rho * a * 1j * u - k_eff) ** 2 + a**2 * (1j * u + u**2))
            g = (k_eff - self.rho * a * 1j * u - d) / (k_eff - self.rho * a * 1j * u + d)
            exp_dt = cmath.exp(-d * t)

            c = (self.k * self.theta / (a**2)) * (
                (k_eff - self.rho * a * 1j * u - d) * t
                - 2.0 * cmath.log((1.0 - g * exp_dt) / (1.0 - g))
            )
            d_val = ((k_eff - self.rho * a * 1j * u - d) / (a**2)) * (
                (1.0 - exp_dt) / (1.0 - g * exp_dt)
            )
            return cmath.exp(drift + c + d_val * self.v0)

        def integrand1(u: float) -> float:
            if u < 1e-12:
                return 0.0
            cf1 = math.exp(-r * t) * char_return(u - 1j)
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

        price = s0 * p1 - k * math.exp(-r * t) * p2
        intrinsic = max(0.0, s0 - k * math.exp(-r * t))
        return float(max(price, intrinsic))

    def european_put(self, strike: float, maturity: float) -> float:
        """Price European put option via put-call parity."""
        call = self.european_call(strike, maturity)
        forward = self.s0 - strike * math.exp(-self.r * maturity)
        put = call - forward
        return float(max(0.0, put))

    def simulate(
        self,
        t_max: Optional[float] = None,
        num_steps: Optional[int] = None,
        num_paths: int = 1,
        seed: Optional[int] = None,
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        n_paths: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate spot and variance paths using Euler-Maruyama discretization."""
        if seed is not None:
            np.random.seed(seed)

        final_t = t_max if t_max is not None else (T if T is not None else 1.0)
        steps = num_steps if num_steps is not None else (n_steps if n_steps is not None else 100)
        paths_count = n_paths if n_paths is not None else num_paths

        dt = float(final_t) / steps
        sqrt_dt = math.sqrt(dt)

        s_paths = np.zeros((paths_count, steps + 1))
        v_paths = np.zeros((paths_count, steps + 1))

        s_paths[:, 0] = self.s0
        v_paths[:, 0] = self.v0

        k, theta, alpha = self.k, self.theta, self.alpha
        rho = self.rho
        r = self.r

        for step in range(steps):
            s_curr = s_paths[:, step]
            v_curr = np.maximum(0.0, v_paths[:, step])

            z_v = np.random.normal(0.0, 1.0, paths_count)
            z_perp = np.random.normal(0.0, 1.0, paths_count)
            z_s = rho * z_v + math.sqrt(max(0.0, 1.0 - rho**2)) * z_perp

            v_next = (
                v_paths[:, step]
                + k * (theta - v_curr) * dt
                + alpha * np.sqrt(v_curr) * sqrt_dt * z_v
            )
            v_paths[:, step + 1] = np.maximum(0.0, v_next)

            log_s_next = np.log(s_curr) + (r - 0.5 * v_curr) * dt + np.sqrt(v_curr) * sqrt_dt * z_s
            s_paths[:, step + 1] = np.exp(log_s_next)

        return s_paths, v_paths


# Aliases
HestonNandi = HestonNandiProcess
HNGARCH = HestonNandiProcess
