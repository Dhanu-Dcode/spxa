r"""
Ho–Lee Short Rate Model (Ho & Lee, 1986).

The Ho–Lee model is the first arbitrage-free interest rate model calibrated
to match the market term structure of interest rates:

    dr_t = θ(t) dt + σ dW_t

where:
    θ(t) = ∂_t f^M(0, t) + σ² t  is calibrated to the market forward curve f^M(0, t)
    σ > 0 : Constant instantaneous short rate volatility
    W_t   : Standard 1-dimensional Brownian motion under risk-neutral measure Q

Theoretical Properties
----------------------
1. Exact Zero-Coupon Bond Price:
   The discount bond price P(t, T) admits an exact analytical formula:
       P(t, T) = A(t, T) exp(-B(t, T) r_t)
   where:
       B(t, T) = T - t
       ln A(t, T) = ln(P^M(0, T) / P^M(0, t)) - (T - t) f^M(0, t) - (1/2) σ² t (T - t)²

2. Normally Distributed Short Rate:
   Because the drift is deterministic, r_t is normally distributed with:
       E[r_t] = f^M(0, t) + (1/2) σ² t²
       Var(r_t) = σ² t

Exactness Level: MOMENT_PROPAGATION.

References
----------
Ho, T. S., & Lee, S. B. (1986). Term structure movements and pricing interest
rate contingent claims. *The Journal of Finance*, 41(5), 1011–1029.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Union

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class HoLeeProcess(Process):
    r"""
    Ho–Lee arbitrage-free short rate model.

    Parameters
    ----------
    sigma : float
        Volatility parameter σ > 0.
    r0 : float, optional
        Initial short rate r_0 (defaults to 0.03).
    theta_fn : callable, optional
        Deterministic time-dependent drift θ(t). Defaults to constant theta_0.
    theta_0 : float, optional
        Baseline constant drift if theta_fn is None (defaults to 0.01).
    """

    def __init__(
        self,
        sigma: float = 0.015,
        r0: float = 0.03,
        theta_fn: Optional[Callable[[float], float]] = None,
        theta_0: float = 0.01,
    ) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")

        self.sigma = float(sigma)
        self.r0 = float(r0)
        self.theta_0 = float(theta_0)
        self.theta_fn = theta_fn if theta_fn is not None else (lambda t: self.theta_0 + (self.sigma**2) * t)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"HoLeeProcess(σ={self.sigma}, r0={self.r0})",
            math_note=f"dr_t = θ(t) dt + σ dW_t; exact affine bond pricing B(t,T)=T-t",
            reference="Ho & Lee (1986), Journal of Finance 41(5), 1011–1029",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Exact Zero-Coupon Bond Pricing
    # ------------------------------------------------------------------

    def zero_coupon_bond(
        self,
        t: float,
        T: float,
        r_t: Optional[float] = None,
    ) -> float:
        r"""
        Exact zero-coupon bond price P(t, T).
        """
        if T < t:
            raise ValueError(f"Maturity T ({T}) cannot precede evaluation time t ({t})")
        if T == t:
            return 1.0

        current_r = self.r0 if r_t is None else float(r_t)
        tau = T - t
        b_val = tau

        # Analytical integral for theta(s) under default linear model
        # int_t^T theta(s) ds
        int_theta = self.theta_0 * tau + 0.5 * (self.sigma**2) * (T**2 - t**2)
        # Variance term: 0.5 * sigma^2 * int_t^T (T - s)^2 ds = (1/6) * sigma^2 * tau^3
        var_term = (1.0 / 6.0) * (self.sigma**2) * (tau**3)

        ln_a = -int_theta + var_term
        return math.exp(ln_a - b_val * current_r)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "HoLeeProcess has explicit time-dependent drift θ(t) and does not "
            "admit a stationary Lévy–Khintchine triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=True,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=0.5,
            notes=[
                "Ho-Lee arbitrage-free Gaussian term structure model.",
                f"Volatility σ = {self.sigma}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.r0}
        if order >= 2:
            res[2] = self.sigma**2
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "exact",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of the Ho-Lee short rate process.
        """
        if rng is None:
            rng = np.random.default_rng()

        if t_grid is None:
            t_final = float(T if T is not None else (t_max if t_max is not None else 1.0))
            steps = int(n_steps if n_steps is not None else (num_steps if num_steps is not None else 252))
            times = np.linspace(0.0, t_final, steps + 1)
        elif isinstance(t_grid, (int, float)):
            steps = int(n_steps if n_steps is not None else (num_steps if num_steps is not None else 252))
            times = np.linspace(0.0, float(t_grid), steps + 1)
        else:
            times = np.asarray(t_grid, dtype=np.float64)

        paths_count = int(n_paths if n_paths is not None else (num_paths if num_paths is not None else 500))
        n_pts = len(times)

        r_paths = np.empty((paths_count, n_pts), dtype=np.float64)
        r_paths[:, 0] = self.r0

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)
            t_mid = 0.5 * (times[i - 1] + times[i])
            theta_val = self.theta_fn(t_mid)
            z = rng.standard_normal(paths_count)

            r_paths[:, i] = r_paths[:, i - 1] + theta_val * dt + self.sigma * sqrt_dt * z

        return r_paths


# Aliases
HoLeeModel = HoLeeProcess
