"""CEV-Heston local-stochastic volatility process.

Combines the Constant Elasticity of Variance (CEV) local volatility backbone
with the Heston (1993) square-root mean-reverting stochastic variance process.

SDE:
    dS_t = (r - q) S_t dt + sqrt(V_t) * S_t^gamma * dW_t^S
    dV_t = kappa * (theta - V_t) dt + sigma_v * sqrt(V_t) * dW_t^V
    d<W^S, W^V>_t = rho dt

Special cases:
- gamma = 1: standard Heston (1993) stochastic volatility model.
- sigma_v -> 0: Constant Elasticity of Variance (CEV) local volatility model.

References:
    Andersen, L. & Brotherton-Ratcliffe, R. (2005). Extended LIBOR market models
        with stochastic volatility. Journal of Computational Finance, 9(1), 1-40.
    Heston, S. L. (1993). A closed-form solution for options with stochastic
        volatility with applications to bond and currency options.
        Review of Financial Studies, 6(2), 327-343.
    Lord, R., Koekkoek, R., & van Dijk, D. (2010). A comparison of biased simulation
        schemes for stochastic volatility models. Quantitative Finance, 10(2), 177-194.
"""

from __future__ import annotations

import cmath
import math
from typing import Optional

import numpy as np
from scipy import integrate

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet


class CEVHestonProcess(Process):
    r"""CEV-Heston local-stochastic volatility process.

    Parameters
    ----------
    s0 : float
        Initial asset spot price (must be positive).
    v0 : float
        Initial instantaneous variance (must be positive).
    gamma : float
        CEV elasticity exponent (must be positive). When gamma = 1.0, reduces to Heston.
    kappa : float
        Mean-reversion rate of the variance process (must be positive).
    theta : float
        Long-term mean variance (must be positive).
    sigma_v : float
        Volatility of variance parameter (must be positive).
    rho : float
        Correlation between the spot and variance Brownian motions (-1 <= rho <= 1).
    r : float
        Continuously compounded risk-free interest rate (default: 0.0).
    q : float
        Continuous dividend or convenience yield (default: 0.0).
    """

    def __init__(
        self,
        s0: float = 100.0,
        v0: float = 0.04,
        gamma: float = 0.8,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma_v: float = 0.3,
        rho: float = -0.6,
        r: float = 0.0,
        q: float = 0.0,
    ) -> None:
        if s0 <= 0.0:
            raise ValueError(f"Initial spot price s0 must be positive, got {s0}")
        if v0 <= 0.0:
            raise ValueError(f"Initial variance v0 must be positive, got {v0}")
        if gamma <= 0.0:
            raise ValueError(f"CEV elasticity gamma must be positive, got {gamma}")
        if kappa <= 0.0 or theta <= 0.0 or sigma_v <= 0.0:
            raise ValueError(
                f"Variance parameters kappa, theta, sigma_v must be positive: "
                f"kappa={kappa}, theta={theta}, sigma_v={sigma_v}"
            )
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"Correlation rho must lie in [-1, 1], got {rho}")

        self.s0 = float(s0)
        self.v0 = float(v0)
        self.gamma = float(gamma)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma_v = float(sigma_v)
        self.rho = float(rho)
        self.r = float(r)
        self.q = float(q)

        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "CEVHestonProcess has state-dependent local volatility and stochastic variance; "
            "it does not possess a stationary Levy triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=(self.r == 0.0 and self.q == 0.0),
            has_finite_mean=True,
            has_finite_variance=True,
            notes=[
                "CEV-Heston local-stochastic volatility model (Andersen & Brotherton-Ratcliffe 2005)",
                f"CEV elasticity gamma = {self.gamma}",
                f"Feller ratio 2*kappa*theta / sigma_v^2 = {self.feller_ratio:.4f}",
            ],
        )

    @property
    def feller_ratio(self) -> float:
        r"""Compute the Feller ratio 2 * kappa * theta / sigma_v^2."""
        return float(2.0 * self.kappa * self.theta / (self.sigma_v**2))

    @property
    def satisfies_feller(self) -> bool:
        r"""Check if variance strictly avoids zero: 2 * kappa * theta > sigma_v^2."""
        return self.feller_ratio > 1.0

    @property
    def is_heston(self) -> bool:
        r"""Check if model coincides with standard lognormal Heston (gamma == 1)."""
        return math.isclose(self.gamma, 1.0, rel_tol=1e-9)

    def local_volatility_factor(self, s: float) -> float:
        r"""Compute the CEV local volatility backbone factor: S^{gamma - 1}."""
        if s <= 0.0:
            return 0.0
        return float(s ** (self.gamma - 1.0))

    def expected_variance(self, t: float) -> float:
        r"""Exact conditional variance expectation: E[V_t | V_0]."""
        if t <= 0.0:
            return self.v0
        return float(self.theta + (self.v0 - self.theta) * math.exp(-self.kappa * t))

    def variance_of_variance(self, t: float) -> float:
        r"""Exact conditional variance of variance: Var[V_t | V_0]."""
        if t <= 0.0:
            return 0.0
        k = self.kappa
        th = self.theta
        sig2 = self.sigma_v**2
        e_kt = math.exp(-k * t)
        term1 = self.v0 * (sig2 / k) * (e_kt - e_kt**2)
        term2 = th * (sig2 / (2.0 * k)) * (1.0 - e_kt) ** 2
        return float(term1 + term2)

    def stationary_variance_mean(self) -> float:
        r"""Stationary mean of the CIR variance process: theta."""
        return self.theta

    def stationary_variance_var(self) -> float:
        r"""Stationary variance of the CIR variance process: theta * sigma_v^2 / (2 * kappa)."""
        return float(self.theta * (self.sigma_v**2) / (2.0 * self.kappa))

    def characteristic_function_heston(self, u: complex, t: float) -> complex:
        r"""Exact continuous characteristic function for the Heston limit (gamma = 1.0).

        Uses Albrecher et al. (2007) numerically stable formulation without branch cuts:
        phi(u, t) = E[exp(i * u * ln(S_t / S_0))]
        """
        if not self.is_heston:
            raise ValueError(
                f"Closed-form affine characteristic function is only available for Heston limit (gamma=1.0), "
                f"got gamma={self.gamma}."
            )
        if t <= 0.0:
            return complex(1.0, 0.0)

        u_c = complex(u)
        i = 1j
        k = self.kappa
        th = self.theta
        sig = self.sigma_v
        rho = self.rho
        r = self.r
        q = self.q

        d = cmath.sqrt((k - i * rho * sig * u_c) ** 2 + (sig**2) * (i * u_c + u_c**2))
        g = (k - i * rho * sig * u_c - d) / (k - i * rho * sig * u_c + d)
        exp_dt = cmath.exp(-d * t)

        c = (k * th / (sig**2)) * (
            (k - i * rho * sig * u_c - d) * t - 2.0 * cmath.log((1.0 - g * exp_dt) / (1.0 - g))
        )
        d_val = ((k - i * rho * sig * u_c - d) / (sig**2)) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))

        drift = i * u_c * (r - q) * t
        return cmath.exp(drift + c + d_val * self.v0)

    def european_call(self, strike: float, maturity: float) -> float:
        r"""Price European call option via Fourier inversion (gamma = 1) or Monte Carlo."""
        if strike <= 0.0:
            raise ValueError(f"Strike must be positive, got {strike}")
        if maturity <= 0.0:
            return max(0.0, self.s0 - strike)

        if self.is_heston:
            # Exact Fourier inversion via Gil-Pelaez representation
            s0 = self.s0
            k_val = strike
            r = self.r
            q = self.q
            t = maturity

            def integrand_p1(u: float) -> float:
                cf = self.characteristic_function_heston(u - 1j, t)
                phi0 = self.characteristic_function_heston(-1j, t)
                val = cmath.exp(-1j * u * math.log(k_val / s0)) * cf / (1j * u * phi0)
                return float(val.real)

            def integrand_p2(u: float) -> float:
                cf = self.characteristic_function_heston(u, t)
                val = cmath.exp(-1j * u * math.log(k_val / s0)) * cf / (1j * u)
                return float(val.real)

            int1, _ = integrate.quad(integrand_p1, 1e-8, 200.0, limit=200)
            int2, _ = integrate.quad(integrand_p2, 1e-8, 200.0, limit=200)

            p1 = 0.5 + int1 / math.pi
            p2 = 0.5 + int2 / math.pi

            call = s0 * math.exp(-q * t) * p1 - k_val * math.exp(-r * t) * p2
            intrinsic = max(0.0, s0 * math.exp(-q * t) - k_val * math.exp(-r * t))
            return float(max(intrinsic, call))

        # For general gamma != 1, evaluate via Monte Carlo
        paths = self.simulate(t_max=maturity, num_steps=100, num_paths=10000, seed=42)
        payoffs = np.maximum(0.0, paths[:, -1] - strike)
        return float(math.exp(-self.r * maturity) * np.mean(payoffs))

    def european_put(self, strike: float, maturity: float) -> float:
        r"""Price European put option via put-call parity."""
        call = self.european_call(strike, maturity)
        parity_forward = self.s0 * math.exp(-self.q * maturity) - strike * math.exp(
            -self.r * maturity
        )
        return float(max(0.0, call - parity_forward))

    def simulate(
        self,
        t_max: float = 1.0,
        num_steps: int = 100,
        num_paths: int = 1000,
        seed: Optional[int] = None,
        return_variance: bool = False,
    ) -> np.ndarray:
        r"""Simulate spot price (and optionally variance) paths using full-truncation Euler scheme.

        Parameters
        ----------
        t_max : float
            Simulation time horizon.
        num_steps : int
            Number of discrete time steps.
        num_paths : int
            Number of simulated sample paths.
        seed : int, optional
            Random number generator seed.
        return_variance : bool
            If True, returns a tuple (s_paths, v_paths). Otherwise returns s_paths.
        """
        if t_max <= 0.0 or num_steps <= 0 or num_paths <= 0:
            raise ValueError("t_max, num_steps, and num_paths must be strictly positive.")

        rng = np.random.default_rng(seed)
        dt = t_max / num_steps
        sqrt_dt = math.sqrt(dt)

        s_paths = np.empty((num_paths, num_steps + 1), dtype=float)
        v_paths = np.empty((num_paths, num_steps + 1), dtype=float)

        s_paths[:, 0] = self.s0
        v_paths[:, 0] = self.v0

        rho = self.rho
        sqrt_one_minus_rho2 = math.sqrt(max(0.0, 1.0 - rho**2))
        k = self.kappa
        th = self.theta
        sig_v = self.sigma_v
        r = self.r
        q = self.q
        gamma = self.gamma

        for step in range(num_steps):
            s_curr = s_paths[:, step]
            v_curr = np.maximum(0.0, v_paths[:, step])

            z_v = rng.standard_normal(num_paths)
            z_indep = rng.standard_normal(num_paths)
            z_s = rho * z_v + sqrt_one_minus_rho2 * z_indep

            # Full truncation Euler step for CIR variance
            v_next = (
                v_paths[:, step] + k * (th - v_curr) * dt + sig_v * np.sqrt(v_curr) * sqrt_dt * z_v
            )
            v_paths[:, step + 1] = np.maximum(0.0, v_next)

            # Spot step with CEV local volatility backbone
            sqrt_v = np.sqrt(v_curr)
            drift_term = (r - q) * s_curr * dt
            diff_term = sqrt_v * (np.maximum(0.0, s_curr) ** gamma) * sqrt_dt * z_s
            s_next = s_curr + drift_term + diff_term
            s_paths[:, step + 1] = np.maximum(0.0, s_next)

        if return_variance:
            return s_paths, v_paths
        return s_paths


CEVHeston = CEVHestonProcess
HestonCEV = CEVHestonProcess
SLVCEV = CEVHestonProcess
