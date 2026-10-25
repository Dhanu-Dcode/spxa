"""Merton (1974) Structural Credit Risk and Firm Value Process.

The structural credit risk framework of Robert C. Merton (1974) models a firm's total asset
value V(t) as a Geometric Brownian Motion:
    dV(t) = mu * V(t) dt + sigma_V * V(t) dW(t)
The firm has zero-coupon debt with face value D maturing at horizon T. Default occurs if
V(T) < D at maturity. Equity E_0 is modeled as a European call option on firm value with
strike D, while debt B_0 is a covered call / risk-free debt minus put option.

References:
    Merton, R. C. (1974). On the pricing of corporate debt: The risk structure of interest rates.
        The Journal of Finance, 29(2), 449-470.
    Black, F., & Cox, J. C. (1976). Valuing corporate securities: Some effects of bond indenture provisions.
        The Journal of Finance, 31(2), 351-367.
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


class MertonFirmValueProcess(Process):
    r"""Merton (1974) structural default and corporate credit model.

    .. math::
        dV_t = \mu V_t dt + \sigma_V V_t dW_t
    """

    def __init__(
        self,
        v0: float,
        debt: float,
        sigma_v: float,
        mu: float = 0.05,
        r: float = 0.02,
        t_debt: float = 1.0,
    ) -> None:
        if v0 <= 0:
            raise ValueError(f"Initial firm value v0 must be strictly positive: {v0}")
        if debt <= 0:
            raise ValueError(f"Debt face value debt must be strictly positive: {debt}")
        if sigma_v <= 0:
            raise ValueError(f"Asset volatility sigma_v must be strictly positive: {sigma_v}")
        if t_debt <= 0:
            raise ValueError(f"Debt maturity t_debt must be strictly positive: {t_debt}")

        self.v0 = float(v0)
        self.debt = float(debt)
        self.sigma_v = float(sigma_v)
        self.mu = float(mu)
        self.r = float(r)
        self.t_debt = float(t_debt)
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _triplet(self):
        raise ExactnessError(
            "MertonFirmValueProcess represents firm structural value with default thresholds, not a pure Levy process."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=True,
            is_martingale=(self.mu == 0.0),
            has_finite_variance=True,
            has_finite_mean=True,
            notes=["Merton (1974) structural firm value and default credit model"],
        )

    @property
    def leverage(self) -> float:
        """Quasi-market leverage ratio D / V_0."""
        return self.debt / self.v0

    def distance_to_default(self, t: Optional[float] = None) -> float:
        """Physical distance to default (DD) in standard deviations."""
        horizon = self.t_debt if t is None else float(t)
        if horizon <= 0:
            return float("inf") if self.v0 >= self.debt else float("-inf")
        num = math.log(self.v0 / self.debt) + (self.mu - 0.5 * self.sigma_v**2) * horizon
        denom = self.sigma_v * math.sqrt(horizon)
        return float(num / denom)

    def physical_default_probability(self, t: Optional[float] = None) -> float:
        """Real-world (P-measure) cumulative default probability P(V_T < D)."""
        dd = self.distance_to_default(t)
        return float(stats.norm.cdf(-dd))

    def _d1_d2(self, t: float) -> tuple[float, float]:
        """Compute Black-Scholes risk-neutral d1 and d2."""
        num = math.log(self.v0 / self.debt) + (self.r + 0.5 * self.sigma_v**2) * t
        denom = self.sigma_v * math.sqrt(t)
        d1 = num / denom
        d2 = d1 - self.sigma_v * math.sqrt(t)
        return float(d1), float(d2)

    def risk_neutral_default_probability(self, t: Optional[float] = None) -> float:
        """Risk-neutral (Q-measure) default probability Q(V_T < D) = N(-d2)."""
        horizon = self.t_debt if t is None else float(t)
        _, d2 = self._d1_d2(horizon)
        return float(stats.norm.cdf(-d2))

    def equity_value(self, t: Optional[float] = None) -> float:
        """Market value of equity E_0 = Call(V_0, D, T, r, sigma_V)."""
        horizon = self.t_debt if t is None else float(t)
        d1, d2 = self._d1_d2(horizon)
        e0 = self.v0 * stats.norm.cdf(d1) - self.debt * math.exp(
            -self.r * horizon
        ) * stats.norm.cdf(d2)
        return float(max(0.0, e0))

    def debt_value(self, t: Optional[float] = None) -> float:
        """Market value of debt B_0 = V_0 - E_0."""
        e0 = self.equity_value(t)
        return float(self.v0 - e0)

    def credit_spread(self, t: Optional[float] = None) -> float:
        """Yield-to-maturity credit spread s(T) = -ln(B_0 / D)/T - r."""
        horizon = self.t_debt if t is None else float(t)
        b0 = self.debt_value(horizon)
        if b0 <= 0:
            return float("inf")
        y = -math.log(b0 / self.debt) / horizon
        return float(max(0.0, y - self.r))

    def simulate(
        self,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        seed: Optional[int] = None,
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        n_paths: Optional[int] = None,
        *,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        """Simulate asset value paths up to debt maturity t_debt or T."""
        if seed is not None:
            np.random.seed(seed)

        final_t = T if T is not None else (t_max if t_max is not None else self.t_debt)
        steps = num_steps if num_steps is not None else (n_steps if n_steps is not None else 100)
        paths_count = (
            num_paths if num_paths is not None else (n_paths if n_paths is not None else 1)
        )

        dt = final_t / steps
        sqrt_dt = math.sqrt(dt)

        paths = np.zeros((paths_count, steps + 1))
        paths[:, 0] = self.v0

        drift = (self.mu - 0.5 * self.sigma_v**2) * dt
        vol = self.sigma_v * sqrt_dt

        for step in range(steps):
            z = np.random.normal(0.0, 1.0, paths_count)
            paths[:, step + 1] = paths[:, step] * np.exp(drift + vol * z)

        return paths


# Aliases
MertonDefaultModel = MertonFirmValueProcess
StructuralCreditModel = MertonFirmValueProcess
