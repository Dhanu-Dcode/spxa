r"""
Double Mean-Reverting Stochastic Volatility Model (Gatheral, 2008).

Models two distinct timescales of volatility dynamics (a fast intraday/weekly
timescale and a slow macroeconomic trend timescale) via a two-factor additive
variance representation:

    dS_t = μ S_t dt + √(v_{1,t} + v_{2,t}) S_t dW_t^S
    dv_{1,t} = κ_1 (θ_1 - v_{1,t}) dt + σ_1 √v_{1,t} dW_t^1  (fast factor)
    dv_{2,t} = κ_2 (θ_2 - v_{2,t}) dt + σ_2 √v_{2,t} dW_t^2  (slow factor)

where:
    d⟨W^S, W^1⟩_t = ρ_1 dt
    d⟨W^S, W^2⟩_t = ρ_2 dt
    d⟨W^1, W^2⟩_t = 0
    κ_1 ≫ κ_2 > 0  (e.g., κ_1 ≈ 10.0, κ_2 ≈ 0.5)

Theoretical Properties
----------------------
1. Decoupled Volatility Smile Dynamics:
   - The fast factor κ_1 controls the steep slope and curvature of the short-end
     implied volatility smile.
   - The slow factor κ_2 controls the long-maturity smile persistence and level.

2. Affine Characteristic Function:
   The joint log-price characteristic function is affine and separates into
   two independent systems of Riccati ODEs.

Exactness Level: MOMENT_PROPAGATION.

References
----------
Gatheral, J. (2008). Consistent modeling of SPX and VIX options.
*Presentation at the 5th World Congress of the Bachelier Finance Society*.
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


class DoubleMeanRevertingSV(Process):
    r"""
    Double Mean-Reverting Two-Factor Stochastic Volatility Model.

    Parameters
    ----------
    kappa1 : float
        Fast mean-reversion speed κ_1 > 0 (e.g. 8.0).
    theta1 : float
        Fast long-term variance level θ_1 > 0 (e.g. 0.02).
    sigma1 : float
        Fast factor volatility-of-variance σ_1 > 0.
    rho1 : float
        Correlation ρ_1 ∈ [-1, 1] between spot and fast variance.
    kappa2 : float
        Slow mean-reversion speed κ_2 > 0 (e.g. 0.5).
    theta2 : float
        Slow long-term variance level θ_2 > 0 (e.g. 0.02).
    sigma2 : float
        Slow factor volatility-of-variance σ_2 > 0.
    rho2 : float
        Correlation ρ_2 ∈ [-1, 1] between spot and slow variance.
    s0 : float, optional
        Initial spot price S_0 > 0 (defaults to 100.0).
    v1_0 : float, optional
        Initial fast variance (defaults to θ_1).
    v2_0 : float, optional
        Initial slow variance (defaults to θ_2).
    mu : float, optional
        Asset drift μ (defaults to 0.0).
    """

    def __init__(
        self,
        kappa1: float = 8.0,
        theta1: float = 0.02,
        sigma1: float = 0.4,
        rho1: float = -0.7,
        kappa2: float = 0.8,
        theta2: float = 0.02,
        sigma2: float = 0.2,
        rho2: float = -0.5,
        s0: float = 100.0,
        v1_0: Optional[float] = None,
        v2_0: Optional[float] = None,
        mu: float = 0.0,
    ) -> None:
        if kappa1 <= 0 or kappa2 <= 0:
            raise ValueError("Both kappa1 and kappa2 must be strictly positive")
        if theta1 <= 0 or theta2 <= 0:
            raise ValueError("Both theta1 and theta2 must be strictly positive")
        if sigma1 <= 0 or sigma2 <= 0:
            raise ValueError("Both sigma1 and sigma2 must be strictly positive")
        if not (-1.0 <= rho1 <= 1.0) or not (-1.0 <= rho2 <= 1.0):
            raise ValueError("Both rho1 and rho2 must be in [-1, 1]")
        if (rho1**2 + rho2**2) > 1.0:
            raise ValueError("Total correlation variance constraint rho1^2 + rho2^2 <= 1 violated")
        if s0 <= 0:
            raise ValueError(f"s0 must be strictly positive, got {s0}")

        self.kappa1 = float(kappa1)
        self.theta1 = float(theta1)
        self.sigma1 = float(sigma1)
        self.rho1 = float(rho1)

        self.kappa2 = float(kappa2)
        self.theta2 = float(theta2)
        self.sigma2 = float(sigma2)
        self.rho2 = float(rho2)

        self.s0 = float(s0)
        self.mu = float(mu)
        self.v1_0 = float(theta1 if v1_0 is None else v1_0)
        self.v2_0 = float(theta2 if v2_0 is None else v2_0)

        if self.v1_0 <= 0 or self.v2_0 <= 0:
            raise ValueError("Initial variance components must be strictly positive")

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"DoubleMeanRevertingSV(κ1={self.kappa1}, κ2={self.kappa2}, θ1={self.theta1}, θ2={self.theta2})",
            math_note=(
                f"Two-scale variance v_t = v_1 + v_2; fast κ_1={self.kappa1:.1f}, slow κ_2={self.kappa2:.1f}; "
                f"ρ_1={self.rho1:.2f}, ρ_2={self.rho2:.2f}"
            ),
            reference="Gatheral (2008), 5th World Congress Bachelier Finance Society",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Properties & Moments
    # ------------------------------------------------------------------

    @property
    def total_initial_variance(self) -> float:
        r"""Total initial variance v_0 = v_{1,0} + v_{2,0}."""
        return self.v1_0 + self.v2_0

    @property
    def total_long_term_variance(self) -> float:
        r"""Total long-term variance level θ = θ_1 + θ_2."""
        return self.theta1 + self.theta2

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "DoubleMeanRevertingSV has state-dependent multi-scale variance and cannot be "
            "represented by an independent Lévy–Khintchine triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=(self.mu == 0.0),
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=None,
            notes=[
                "Gatheral double mean-reverting two-scale stochastic volatility model.",
                f"Fast rate κ_1 = {self.kappa1}, Slow rate κ_2 = {self.kappa2}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.s0 * math.exp(self.mu)}
        if order >= 2:
            res[2] = self.total_long_term_variance
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Simulation (Vectorized Full-Truncation Multi-Factor CIR)
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "full_truncation",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of asset price S_t driven by two variance factors.
        """
        if rng is None:
            rng = np.random.default_rng()

        if t_grid is None:
            t_final = float(T if T is not None else (t_max if t_max is not None else 1.0))
            steps = int(
                n_steps if n_steps is not None else (num_steps if num_steps is not None else 252)
            )
            times = np.linspace(0.0, t_final, steps + 1)
        elif isinstance(t_grid, (int, float)):
            steps = int(
                n_steps if n_steps is not None else (num_steps if num_steps is not None else 252)
            )
            times = np.linspace(0.0, float(t_grid), steps + 1)
        else:
            times = np.asarray(t_grid, dtype=np.float64)

        paths_count = int(
            n_paths if n_paths is not None else (num_paths if num_paths is not None else 1000)
        )
        n_pts = len(times)

        s_paths = np.empty((paths_count, n_pts), dtype=np.float64)
        s_paths[:, 0] = self.s0

        s_curr = np.full(paths_count, self.s0, dtype=np.float64)
        v1_curr = np.full(paths_count, self.v1_0, dtype=np.float64)
        v2_curr = np.full(paths_count, self.v2_0, dtype=np.float64)

        rho_s_indep = math.sqrt(max(0.0, 1.0 - (self.rho1**2 + self.rho2**2)))

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            if dt <= 0:
                s_paths[:, i] = s_curr
                continue

            sqrt_dt = math.sqrt(dt)
            z1 = rng.standard_normal(paths_count)
            z2 = rng.standard_normal(paths_count)
            z3 = rng.standard_normal(paths_count)

            zv1 = z1
            zv2 = z2
            zs = self.rho1 * z1 + self.rho2 * z2 + rho_s_indep * z3

            v1_pos = np.maximum(v1_curr, 0.0)
            v2_pos = np.maximum(v2_curr, 0.0)
            v_tot = v1_pos + v2_pos
            sqrt_v_tot = np.sqrt(v_tot)

            # Spot update
            s_curr = s_curr * np.exp((self.mu - 0.5 * v_tot) * dt + sqrt_v_tot * sqrt_dt * zs)

            # Fast variance factor update
            v1_curr = (
                v1_curr
                + self.kappa1 * (self.theta1 - v1_pos) * dt
                + self.sigma1 * np.sqrt(v1_pos) * sqrt_dt * zv1
            )

            # Slow variance factor update
            v2_curr = (
                v2_curr
                + self.kappa2 * (self.theta2 - v2_pos) * dt
                + self.sigma2 * np.sqrt(v2_pos) * sqrt_dt * zv2
            )

            s_paths[:, i] = s_curr

        return s_paths


# Aliases
TwoScaleSVProcess = DoubleMeanRevertingSV
