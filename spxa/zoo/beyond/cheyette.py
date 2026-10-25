"""Cheyette Quasi-Gaussian Markovian Heath-Jarrow-Morton (HJM) process.

The Cheyette (1992) model is the finite-dimensional Markovian representation of
the Heath-Jarrow-Morton framework with an exponentially decaying volatility structure:
    sigma_f(t, T) = sigma * exp(-kappa * (T - t))

State dynamics:
    dx_t = (y_t - kappa * x_t) dt + sigma * dW_t,   x_0 = 0
    dy_t = (sigma^2 - 2 * kappa * y_t) dt,          y_0 = 0
    r_t  = r0 + x_t

Bond pricing formula:
    P(t, T) = exp(-r0 * (T - t) - G(t, T) * x_t - 0.5 * G(t, T)^2 * y_t)
    where G(t, T) = (1 - exp(-kappa * (T - t))) / kappa

References:
    Cheyette, O. (1992). Markov representation of the Heath-Jarrow-Morton model.
        BARRA working paper.
    Andreasen, J. & Andersen, L. (2002). Volatilities in the Cheyette model.
        Working paper.
    Beyna, I. (2013). Interest Rate Models: An Infinite Dimensional Stochastic
        Analysis Perspective. Springer.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import stats

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet


class CheyetteProcess(Process):
    r"""Cheyette Quasi-Gaussian HJM Markovian short-rate process.

    Parameters
    ----------
    kappa : float
        Mean-reversion rate of the forward rate volatility decay (must be positive).
    sigma : float
        Volatility scaling of the forward rate curve (must be positive).
    r0 : float
        Initial flat forward / short rate (default: 0.03).
    x0 : float
        Initial state variable x_0 (default: 0.0).
    y0 : float
        Initial auxiliary state variable y_0 (default: 0.0).
    """

    def __init__(
        self,
        kappa: float = 0.5,
        sigma: float = 0.015,
        r0: float = 0.03,
        x0: float = 0.0,
        y0: float = 0.0,
    ) -> None:
        if kappa <= 0.0:
            raise ValueError(f"Mean-reversion parameter kappa must be positive, got {kappa}")
        if sigma <= 0.0:
            raise ValueError(f"Volatility parameter sigma must be positive, got {sigma}")
        if y0 < 0.0:
            raise ValueError(f"Auxiliary state variable y0 must be non-negative, got {y0}")

        self.kappa = float(kappa)
        self.sigma = float(sigma)
        self.r0 = float(r0)
        self.x0 = float(x0)
        self.y0 = float(y0)

        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "CheyetteProcess is a non-stationary Markovian HJM term-structure process; "
            "it does not possess a stationary Levy triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_mean=True,
            has_finite_variance=True,
            notes=[
                "Cheyette Quasi-Gaussian Markovian HJM model (Cheyette 1992)",
                f"Decay rate kappa = {self.kappa}, sigma = {self.sigma}",
            ],
        )

    def g_factor(self, t: float, big_t: float) -> float:
        r"""Compute the HJM volatility integration factor G(t, T) = (1 - exp(-kappa*(T - t))) / kappa."""
        if big_t <= t:
            return 0.0
        tau = big_t - t
        return float((1.0 - math.exp(-self.kappa * tau)) / self.kappa)

    def y_deterministic(self, t: float) -> float:
        r"""Exact deterministic value of the auxiliary state variable y_t:

        y_t = y_0 * exp(-2*kappa*t) + (sigma^2 / (2*kappa)) * (1 - exp(-2*kappa*t)).
        """
        if t <= 0.0:
            return self.y0
        k = self.kappa
        sig2 = self.sigma**2
        e_2kt = math.exp(-2.0 * k * t)
        return float(self.y0 * e_2kt + (sig2 / (2.0 * k)) * (1.0 - e_2kt))

    def stationary_y(self) -> float:
        r"""Long-run stationary level of y_t: sigma^2 / (2 * kappa)."""
        return float(self.sigma**2 / (2.0 * self.kappa))

    def mean_x(self, t: float) -> float:
        r"""Exact conditional expectation of state variable x_t under the risk-neutral measure.

        E[x_t | x_0, y_0] = x_0 * exp(-kappa*t) + (sigma^2 / (2*kappa^2)) * (1 - exp(-kappa*t))^2.
        """
        if t <= 0.0:
            return self.x0
        k = self.kappa
        sig2 = self.sigma**2
        e_kt = math.exp(-k * t)
        hjm_drift = (sig2 / (2.0 * k**2)) * ((1.0 - e_kt) ** 2)
        return float(self.x0 * e_kt + hjm_drift)

    def variance_x(self, t: float) -> float:
        r"""Exact conditional variance of state variable x_t:

        Var[x_t] = (sigma^2 / (2*kappa)) * (1 - exp(-2*kappa*t)) = y_t (when y_0 = 0).
        """
        if t <= 0.0:
            return 0.0
        return float((self.sigma**2 / (2.0 * self.kappa)) * (1.0 - math.exp(-2.0 * self.kappa * t)))

    def mean_short_rate(self, t: float) -> float:
        r"""Exact expected instantaneous short rate E[r_t] = r_0 + E[x_t]."""
        return float(self.r0 + self.mean_x(t))

    def variance_short_rate(self, t: float) -> float:
        r"""Exact variance of the instantaneous short rate Var[r_t] = Var[x_t]."""
        return self.variance_x(t)

    def zero_coupon_bond(
        self, tau: float, x_val: Optional[float] = None, t_now: float = 0.0
    ) -> float:
        r"""Zero-coupon bond price P(t, t + tau).

        At t_now = 0: P(0, tau) = exp(-r_0 * tau - G(0, tau) * x_0 - 0.5 * G(0, tau)^2 * y_0).
        """
        if tau <= 0.0:
            return 1.0
        x = self.x0 if x_val is None else x_val
        y = self.y_deterministic(t_now)
        g = self.g_factor(0.0, tau)
        exponent = -self.r0 * tau - g * x - 0.5 * (g**2) * y
        return float(math.exp(exponent))

    def forward_rate(self, tau: float, x_val: Optional[float] = None, t_now: float = 0.0) -> float:
        r"""Instantaneous forward rate f(t, t + tau)."""
        x = self.x0 if x_val is None else x_val
        y = self.y_deterministic(t_now)
        k = self.kappa
        e_kt = math.exp(-k * tau)
        g = self.g_factor(0.0, tau)
        return float(self.r0 + e_kt * x + e_kt * g * y)

    def yield_curve(self, tau: float) -> float:
        r"""Continuously compounded zero yield y(0, tau) = -ln P(0, tau) / tau."""
        if tau <= 0.0:
            return self.r0 + self.x0
        p = self.zero_coupon_bond(tau)
        return float(-math.log(p) / tau)

    def european_bond_call(self, maturity: float, bond_maturity: float, strike: float) -> float:
        r"""Closed-form European call option on a zero-coupon bond P(T_opt, T_bond) via Jamshidian Black-76.

        Parameters
        ----------
        maturity : float
            Option expiry T_opt.
        bond_maturity : float
            Bond maturity T_bond (must satisfy T_bond > T_opt).
        strike : float
            Strike price K.
        """
        if maturity >= bond_maturity:
            raise ValueError(
                f"Option maturity {maturity} must be strictly less than bond maturity {bond_maturity}"
            )
        if strike <= 0.0:
            raise ValueError(f"Strike must be positive, got {strike}")

        # Forward bond price
        p_opt = self.zero_coupon_bond(maturity)
        p_bond = self.zero_coupon_bond(bond_maturity)
        forward_bond = p_bond / p_opt

        # Integrated variance of the forward bond price
        # sigma_P(t) = G(t, T_bond) - G(t, T_opt) = exp(-kappa*(T_opt - t)) * G(T_opt, T_bond) * sigma
        k = self.kappa
        tau_diff = bond_maturity - maturity
        g_diff = (1.0 - math.exp(-k * tau_diff)) / k
        sig = self.sigma

        # Integrated vol over [0, maturity]
        # Integral_0^{T_opt} [g_diff * exp(-kappa*(T_opt - s)) * sig]^2 ds
        int_var = (sig**2) * (g_diff**2) * (1.0 - math.exp(-2.0 * k * maturity)) / (2.0 * k)
        sigma_total = math.sqrt(max(1e-12, int_var))

        d1 = (math.log(forward_bond / strike) + 0.5 * (sigma_total**2)) / sigma_total
        d2 = d1 - sigma_total

        call_fwd = forward_bond * stats.norm.cdf(d1) - strike * stats.norm.cdf(d2)
        return float(p_opt * call_fwd)

    def simulate(
        self,
        t_max: float = 1.0,
        num_steps: int = 100,
        num_paths: int = 1000,
        seed: Optional[int] = None,
        return_states: bool = False,
    ) -> np.ndarray:
        r"""Simulate short rate r_t and state paths (x_t, y_t).

        Parameters
        ----------
        t_max : float
            Simulation horizon.
        num_steps : int
            Number of time steps.
        num_paths : int
            Number of sample paths.
        seed : int, optional
            RNG seed.
        return_states : bool
            If True, returns tuple (r_paths, x_paths, y_paths). Otherwise returns r_paths.
        """
        if t_max <= 0.0 or num_steps <= 0 or num_paths <= 0:
            raise ValueError("t_max, num_steps, and num_paths must be strictly positive.")

        rng = np.random.default_rng(seed)
        dt = t_max / num_steps
        sqrt_dt = math.sqrt(dt)

        x_paths = np.empty((num_paths, num_steps + 1), dtype=float)
        y_paths = np.empty((num_paths, num_steps + 1), dtype=float)

        x_paths[:, 0] = self.x0
        y_paths[:, 0] = self.y0

        k = self.kappa
        sig = self.sigma

        for step in range(num_steps):
            t_curr = step * dt
            # Exact deterministic y_curr and y_next
            y_curr = y_paths[:, step]
            y_next = self.y_deterministic(t_curr + dt)
            y_paths[:, step + 1] = y_next

            # x_t update with drift (y_t - kappa * x_t)
            z = rng.standard_normal(num_paths)
            dx = (y_curr - k * x_paths[:, step]) * dt + sig * sqrt_dt * z
            x_paths[:, step + 1] = x_paths[:, step] + dx

        r_paths = self.r0 + x_paths

        if return_states:
            return r_paths, x_paths, y_paths
        return r_paths


Cheyette = CheyetteProcess
CheyetteModel = CheyetteProcess
QuasiGaussianHJM = CheyetteProcess
