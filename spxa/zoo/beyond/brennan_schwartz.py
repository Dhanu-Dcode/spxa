"""Brennan-Schwartz (1985) Two-Factor Commodity Process.

A two-factor commodity valuation model where spot price S(t) and proportional convenience
yield delta(t) evolve jointly:
    dS(t) = (mu - delta(t)) * S(t) dt + sigma_1 * S(t) dW_1(t)
    ddelta(t) = k * (delta_bar - delta(t)) dt + sigma_2 * delta(t) dW_2(t)
    Corr(dW_1, dW_2) = rho

References:
    Brennan, M. J., & Schwartz, E. S. (1985). Evaluating natural resource investments.
        The Journal of Business, 58(2), 135-157.
    Gibson, R., & Schwartz, E. S. (1990). Stochastic convenience yield and the pricing of oil contingent claims.
        The Journal of Finance, 45(3), 959-976.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties


class BrennanSchwartzProcess(Process):
    r"""Brennan-Schwartz (1985) two-factor commodity diffusion model.

    .. math::
        dS_t = (\mu - \delta_t) S_t dt + \sigma_1 S_t dW_{1,t}
        d\delta_t = \kappa (\bar{\delta} - \delta_t) dt + \sigma_2 \delta_t dW_{2,t}
    """

    def __init__(
        self,
        s0: float,
        delta0: float,
        mu: float,
        k: float,
        delta_bar: float,
        sigma1: float,
        sigma2: float,
        rho: float = 0.0,
        r: float = 0.03,
    ) -> None:
        if s0 <= 0:
            raise ValueError(f"Spot price s0 must be positive: {s0}")
        if delta0 <= 0:
            raise ValueError(f"Convenience yield delta0 must be positive: {delta0}")
        if k <= 0 or delta_bar <= 0:
            raise ValueError(
                f"Mean-reversion parameters k and delta_bar must be positive: k={k}, delta_bar={delta_bar}"
            )
        if sigma1 <= 0 or sigma2 <= 0:
            raise ValueError(
                f"Volatilities sigma1 and sigma2 must be positive: sigma1={sigma1}, sigma2={sigma2}"
            )
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"Correlation rho must be in [-1, 1]: {rho}")

        self.s0 = float(s0)
        self.delta0 = float(delta0)
        self.mu = float(mu)
        self.k = float(k)
        self.delta_bar = float(delta_bar)
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)
        self.rho = float(rho)
        self.r = float(r)
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _triplet(self):
        raise ExactnessError(
            "BrennanSchwartzProcess has stochastic convenience yield drift and cannot be represented by a Levy triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            notes=["Brennan-Schwartz (1985) two-factor commodity and convenience yield model"],
        )

    def mean_convenience_yield(self, t: float) -> float:
        """Expected convenience yield E[delta(t) | delta(0)]."""
        if t <= 0:
            return self.delta0
        return self.delta_bar + (self.delta0 - self.delta_bar) * math.exp(-self.k * t)

    def stationary_mean_convenience_yield(self) -> float:
        """Stationary unconditional mean of convenience yield."""
        return self.delta_bar

    def stationary_var_convenience_yield(self) -> float:
        """Stationary unconditional variance of convenience yield (requires 2*k > sigma2^2)."""
        denom = 2.0 * self.k - self.sigma2**2
        if denom <= 0:
            return float("inf")
        return (self.delta_bar**2 * self.sigma2**2) / denom

    def futures_price(self, tau: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Futures price term structure F(0, tau)."""
        is_scalar = np.isscalar(tau)
        taus = np.atleast_1d(tau).astype(float)
        prices = np.zeros_like(taus)

        for i, t in enumerate(taus):
            if t <= 0:
                prices[i] = self.s0
                continue
            int_exp = (1.0 - math.exp(-self.k * t)) / self.k
            drift_term = (self.r - self.delta_bar) * t
            yield_term = (self.delta_bar - self.delta0) * int_exp
            prices[i] = self.s0 * math.exp(drift_term + yield_term)

        return float(prices[0]) if is_scalar else prices

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
        """Simulate joint spot and convenience yield paths."""
        if seed is not None:
            np.random.seed(seed)

        final_t = t_max if t_max is not None else (T if T is not None else 1.0)
        steps = num_steps if num_steps is not None else (n_steps if n_steps is not None else 100)
        paths_count = n_paths if n_paths is not None else num_paths

        dt = float(final_t) / steps
        sqrt_dt = math.sqrt(dt)

        s_paths = np.zeros((paths_count, steps + 1))
        d_paths = np.zeros((paths_count, steps + 1))

        s_paths[:, 0] = self.s0
        d_paths[:, 0] = self.delta0

        rho = self.rho
        sig1 = self.sigma1
        sig2 = self.sigma2
        k = self.k
        d_bar = self.delta_bar
        mu = self.mu

        for step in range(steps):
            s_curr = s_paths[:, step]
            d_curr = np.maximum(1e-6, d_paths[:, step])

            # Correlated Gaussians
            z1 = np.random.normal(0.0, 1.0, paths_count)
            z_perp = np.random.normal(0.0, 1.0, paths_count)
            z2 = rho * z1 + math.sqrt(max(0.0, 1.0 - rho**2)) * z_perp

            # Convenience yield update
            d_next = d_curr + k * (d_bar - d_curr) * dt + sig2 * d_curr * sqrt_dt * z2
            d_paths[:, step + 1] = np.maximum(1e-6, d_next)

            # Spot price log-update
            log_s_next = np.log(s_curr) + (mu - d_curr - 0.5 * sig1**2) * dt + sig1 * sqrt_dt * z1
            s_paths[:, step + 1] = np.exp(log_s_next)

        return s_paths, d_paths


# Aliases
BrennanSchwartz = BrennanSchwartzProcess
CommodityConvenienceModel = BrennanSchwartzProcess
