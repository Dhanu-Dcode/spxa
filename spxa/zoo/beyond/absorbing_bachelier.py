r"""
Absorbing Bachelier Process with Default Barrier (Bachelier, 1900; Lipton, 2018).

Arithmetic Brownian motion with an absorbing boundary at zero, serving as the
foundational structural model for equity default risk and negative commodity spreads
with physical insolvency:

    dX_t = μ dt + σ dW_t   for t < τ = inf{t > 0 : X_t ≤ 0}
    X_t = 0                for t ≥ τ

Theoretical Properties
----------------------
1. Exact First-Passage Default Time Density (Inverse Gaussian):
       f_τ(t) = (x_0 / (σ √(2π t³))) exp(-(x_0 + μ t)² / (2 σ² t))

2. Exact Survival Probability S(t) = P(τ > t):
       S(t) = Φ((x_0 + μ t)/(σ√t)) - exp(-2μ x_0 / σ²) Φ((-x_0 + μ t)/(σ√t))

3. Defective Absorbing Transition Density on (0, ∞):
       p(x, t | x_0) = (1 / (σ√t)) [ φ((x - x_0 - μ t)/(σ√t)) - exp(-2μ x_0 / σ²) φ((x + x_0 - μ t)/(σ√t)) ]

Exactness Level: MOMENT_PROPAGATION (exact analytical first passage and transition density).

References
----------
Bachelier, L. (1900). Théorie de la spéculation. *Annales Scientifiques de l'É.N.S.*, 17, 21–86.

Lipton, A. (2018). *Financial Engineering: Selected Works of Alexander Lipton*.
World Scientific.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy.stats import norm

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class AbsorbingBachelierProcess(Process):
    r"""
    Absorbing Bachelier Process with default barrier at 0.

    Parameters
    ----------
    mu : float
        Drift parameter μ.
    sigma : float
        Volatility parameter σ > 0.
    x0 : float
        Initial position X_0 > 0.
    """

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        x0: float = 2.0,
    ) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")
        if x0 <= 0:
            raise ValueError(f"x0 must be strictly positive (above barrier 0), got {x0}")

        self.mu = float(mu)
        self.sigma = float(sigma)
        self.x0 = float(x0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"AbsorbingBachelier(μ={self.mu}, σ={self.sigma}, x0={self.x0})",
            math_note="dX_t = μ dt + σ dW_t absorbed at X_t ≤ 0; exact survival S(t)",
            reference="Bachelier (1900); Lipton (2018)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Exact Survival & Default Analytics
    # ------------------------------------------------------------------

    def survival_probability(self, t: float, x0: Optional[float] = None) -> float:
        r"""
        Exact probability of surviving without hitting 0 up to time t: P(τ > t).
        """
        if t <= 0:
            return 1.0
        init_x = self.x0 if x0 is None else float(x0)
        s = self.sigma
        m = self.mu
        sqrt_t = math.sqrt(t)

        z1 = (init_x + m * t) / (s * sqrt_t)
        term1 = norm.cdf(z1)

        scale = -(2.0 * m * init_x) / (s**2)
        if scale > 500.0:
            term2 = 0.0
        else:
            z2 = (-init_x + m * t) / (s * sqrt_t)
            term2 = math.exp(scale) * norm.cdf(z2)

        surv = term1 - term2
        return min(max(surv, 0.0), 1.0)

    def default_probability(self, t: float, x0: Optional[float] = None) -> float:
        r"""Cumulative default probability P(τ ≤ t) = 1 - S(t)."""
        return 1.0 - self.survival_probability(t, x0)

    def transition_density(
        self,
        x: Union[float, np.ndarray],
        t: float,
        x0: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        r"""
        Defective transition density on (0, ∞) before absorption.
        """
        if t <= 0:
            raise ValueError("t must be strictly positive")
        init_x = self.x0 if x0 is None else float(x0)
        x_arr = np.asarray(x, dtype=np.float64)

        s = self.sigma
        m = self.mu
        sqrt_t = math.sqrt(t)

        z1 = (x_arr - init_x - m * t) / (s * sqrt_t)
        z2 = (x_arr + init_x - m * t) / (s * sqrt_t)

        scale = -(2.0 * m * init_x) / (s**2)
        term1 = (1.0 / (s * sqrt_t)) * norm.pdf(z1)

        if scale > 500.0:
            term2 = 0.0
        else:
            term2 = (math.exp(scale) / (s * sqrt_t)) * norm.pdf(z2)

        dens = np.where(x_arr <= 0.0, 0.0, np.maximum(0.0, term1 - term2))
        if np.isscalar(x):
            return float(dens)
        return dens

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "AbsorbingBachelierProcess has an absorbing boundary killing the path at 0 and does not "
            "admit a stationary Lévy–Khintchine triplet."
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
                "Bachelier arithmetic Brownian motion with absorbing default boundary at 0.",
                f"Initial distance to default x_0 = {self.x0}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.x0}
        if order >= 2:
            res[2] = self.sigma**2
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Exact Path Simulation with Pathwise Absorption
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "absorbing_exact",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of Absorbing Bachelier with exact absorption at 0.
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
            n_paths if n_paths is not None else (num_paths if num_paths is not None else 500)
        )
        n_pts = len(times)

        x_paths = np.empty((paths_count, n_pts), dtype=np.float64)
        x_paths[:, 0] = self.x0

        x_curr = np.full(paths_count, self.x0, dtype=np.float64)
        is_absorbed = np.zeros(paths_count, dtype=bool)

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)
            z = rng.standard_normal(paths_count)

            # Update active paths
            x_next = x_curr + self.mu * dt + self.sigma * sqrt_dt * z

            # Check boundary absorption
            newly_absorbed = (x_next <= 0.0) | is_absorbed
            x_curr = np.where(newly_absorbed, 0.0, x_next)
            is_absorbed = newly_absorbed

            x_paths[:, i] = x_curr

        return x_paths


# Aliases
DefaultableBachelier = AbsorbingBachelierProcess
