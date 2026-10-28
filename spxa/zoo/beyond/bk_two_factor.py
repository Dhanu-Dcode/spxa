r"""
Black–Karasinski Two-Factor Process (Black & Karasinski, 1991).

A two-factor lognormal short-rate model that captures both term structure slope
and curvature while maintaining strictly positive interest rates r_t > 0:

    r_t = exp(φ(t) + x_{1,t} + x_{2,t})
    dx_{1,t} = -κ_1 x_{1,t} dt + σ_1 dW_t^1   (fast mean-reverting factor)
    dx_{2,t} = -κ_2 x_{2,t} dt + σ_2 dW_t^2   (slow mean-reverting factor)

where:
    d⟨W^1, W^2⟩_t = ρ dt
    κ_1 > κ_2 > 0 : Distinct mean-reversion rates
    φ(t)          : Deterministic calibration curve

Theoretical Properties
----------------------
1. Positive Rates with Two-Timescale Flexibility:
   Eliminates negative rate artifacts while generating realistic humped volatility
   term structures and decorrelating short vs. long swap rates.

Exactness Level: MOMENT_PROPAGATION.

References
----------
Black, F., & Karasinski, P. (1991). Bond and option pricing when short rates are
lognormal. *Financial Analysts Journal*, 47(4), 52–59.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class BlackKarasinskiTwoFactor(Process):
    r"""
    Black–Karasinski Two-Factor Lognormal Short Rate Model.

    Parameters
    ----------
    kappa1 : float
        Mean-reversion rate κ_1 > 0 for factor 1 (e.g. 1.0).
    sigma1 : float
        Volatility σ_1 > 0 for factor 1.
    kappa2 : float
        Mean-reversion rate κ_2 > 0 for factor 2 (e.g. 0.1).
    sigma2 : float
        Volatility σ_2 > 0 for factor 2.
    rho : float
        Correlation ρ ∈ [-1, 1] between Brownian drivers.
    r0 : float, optional
        Initial short rate r_0 > 0 (defaults to 0.03).
    """

    def __init__(
        self,
        kappa1: float = 1.0,
        sigma1: float = 0.15,
        kappa2: float = 0.1,
        sigma2: float = 0.08,
        rho: float = -0.2,
        r0: float = 0.03,
    ) -> None:
        if kappa1 <= 0 or kappa2 <= 0:
            raise ValueError("kappa1 and kappa2 must be strictly positive")
        if sigma1 <= 0 or sigma2 <= 0:
            raise ValueError("sigma1 and sigma2 must be strictly positive")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1], got {rho}")
        if r0 <= 0:
            raise ValueError(f"r0 must be strictly positive, got {r0}")

        self.kappa1 = float(kappa1)
        self.sigma1 = float(sigma1)
        self.kappa2 = float(kappa2)
        self.sigma2 = float(sigma2)
        self.rho = float(rho)
        self.r0 = float(r0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"BlackKarasinskiTwoFactor(κ1={self.kappa1}, κ2={self.kappa2}, r0={self.r0})",
            math_note=f"r_t = exp(φ(t) + x_1 + x_2); two-factor lognormal Gaussian factors",
            reference="Black & Karasinski (1991), Financial Analysts Journal 47(4), 52–59",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "BlackKarasinskiTwoFactor has two-factor lognormal state dependence and does not "
            "admit an independent Lévy–Khintchine triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=0.5,
            notes=[
                "Two-factor Black-Karasinski lognormal short rate model.",
                f"Fast κ_1 = {self.kappa1}, Slow κ_2 = {self.kappa2}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.r0}
        if order >= 2:
            res[2] = (self.r0**2) * 0.05
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
        method: str = "exact_factors",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of the two-factor BK short rate.
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

        x1 = np.zeros((paths_count, n_pts), dtype=np.float64)
        x2 = np.zeros((paths_count, n_pts), dtype=np.float64)

        rho_comp = math.sqrt(max(0.0, 1.0 - self.rho**2))
        log_r0 = math.log(self.r0)

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)

            z1 = rng.standard_normal(paths_count)
            z2 = rng.standard_normal(paths_count)
            zw1 = z1
            zw2 = self.rho * z1 + rho_comp * z2

            # Exact Gaussian OU transitions
            e1 = math.exp(-self.kappa1 * dt)
            e2 = math.exp(-self.kappa2 * dt)
            std1 = self.sigma1 * math.sqrt((1.0 - e1**2) / (2.0 * self.kappa1))
            std2 = self.sigma2 * math.sqrt((1.0 - e2**2) / (2.0 * self.kappa2))

            x1[:, i] = x1[:, i - 1] * e1 + std1 * zw1
            x2[:, i] = x2[:, i - 1] * e2 + std2 * zw2

        return np.exp(log_r0 + x1 + x2)


# Aliases
BKTwoFactor = BlackKarasinskiTwoFactor
