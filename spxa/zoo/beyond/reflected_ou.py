r"""
Reflected Ornstein–Uhlenbeck (ROU) Process on [0, ∞) (Ward & Glynn, 2003).

A mean-reverting diffusion constrained to the non-negative half-line [0, ∞)
via Skorokhod boundary reflection:

    dX_t = -θ (X_t - μ) dt + σ dW_t + dL_t,   X_t ≥ 0

where:
    θ > 0 : Mean-reversion speed
    μ     : Unconstrained reversion level
    σ > 0 : Volatility parameter
    L_t   : Boundary local time enforcing X_t ≥ 0 (∫_0^t X_s dL_s = 0)

Theoretical Properties
----------------------
1. Truncated Gaussian Stationary Law:
   The stationary distribution on [0, ∞) is an exact Truncated Normal distribution:
       X_∞ ~ TruncatedNormal(μ, σ² / (2θ), lower=0, upper=∞)
   with stationary density:
       f_∞(x) = (1 / Z) φ((x - μ) / (σ / √(2θ)))   for x ≥ 0
   where Z = (σ / √(2θ)) Φ(μ / (σ / √(2θ))).

Exactness Level: MOMENT_PROPAGATION.

References
----------
Ward, A. R., & Glynn, P. W. (2003). A diffusion approximation for a GI/GI/1 queue
with processor sharing. *Queueing Systems*, 43(1), 59–87.
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


class ReflectedOrnsteinUhlenbeck(Process):
    r"""
    Reflected Ornstein–Uhlenbeck Process on [0, ∞).

    Parameters
    ----------
    theta : float
        Mean-reversion rate θ > 0.
    mu : float
        Unconstrained mean reversion level μ.
    sigma : float
        Volatility parameter σ > 0.
    x0 : float, optional
        Initial position X_0 ≥ 0 (defaults to max(0.0, μ)).
    """

    def __init__(
        self,
        theta: float = 1.0,
        mu: float = 1.0,
        sigma: float = 0.5,
        x0: Optional[float] = None,
    ) -> None:
        if theta <= 0:
            raise ValueError(f"theta must be strictly positive, got {theta}")
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")

        self.theta = float(theta)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.x0 = float(max(0.0, mu) if x0 is None else x0)

        if self.x0 < 0:
            raise ValueError(f"x0 must be non-negative on [0, ∞), got {self.x0}")

        # Stationary scale tau = sigma / sqrt(2 theta)
        self.tau = self.sigma / math.sqrt(2.0 * self.theta)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"ReflectedOU(θ={self.theta}, μ={self.mu}, σ={self.sigma}, x0={self.x0})",
            math_note=f"dX_t = -θ(X_t - μ)dt + σ dW_t + dL_t on [0, ∞); stationary TruncNormal",
            reference="Ward & Glynn (2003), Queueing Systems 43(1), 59–87",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Stationary Distribution Analytics
    # ------------------------------------------------------------------

    @property
    def stationary_mean(self) -> float:
        r"""
        Mean of the truncated normal stationary distribution on [0, ∞):
        E[X_∞] = μ + τ λ(α), where α = -μ / τ and λ(α) = φ(α) / (1 - Φ(α)).
        """
        alpha = -self.mu / self.tau
        lam = norm.pdf(alpha) / (1.0 - norm.cdf(alpha))
        return self.mu + self.tau * lam

    @property
    def stationary_variance(self) -> float:
        r"""
        Variance of the truncated normal stationary distribution on [0, ∞):
        Var(X_∞) = τ² [ 1 - δ(α) ] with δ(α) = λ(α) (λ(α) - α).
        """
        alpha = -self.mu / self.tau
        lam = norm.pdf(alpha) / (1.0 - norm.cdf(alpha))
        delta = lam * (lam - alpha)
        return (self.tau**2) * max(0.0, 1.0 - delta)

    def stationary_density(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Probability density function of the stationary truncated normal distribution.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        alpha = -self.mu / self.tau
        denom = (1.0 - norm.cdf(alpha)) * self.tau
        pdf = np.where(x_arr < 0.0, 0.0, norm.pdf((x_arr - self.mu) / self.tau) / denom)
        if np.isscalar(x):
            return float(pdf)
        return pdf

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "ReflectedOrnsteinUhlenbeck process has boundary reflection and state-dependent "
            "drift, lacking an independent Lévy–Khintchine triplet."
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
                "Reflected Ornstein-Uhlenbeck process on [0, ∞).",
                f"Stationary Truncated Normal with mean {self.stationary_mean:.4f}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.stationary_mean}
        if order >= 2:
            res[2] = self.stationary_variance
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Simulation via Skorokhod Reflection
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "skorokhod_ou",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of Reflected Ornstein-Uhlenbeck on [0, ∞).
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

        x_paths = np.empty((paths_count, n_pts), dtype=np.float64)
        x_paths[:, 0] = self.x0

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)
            z = rng.standard_normal(paths_count)

            # OU drift + diffusion step with reflection at 0
            drift = -self.theta * (x_paths[:, i - 1] - self.mu) * dt
            diff = self.sigma * sqrt_dt * z
            x_paths[:, i] = np.maximum(0.0, x_paths[:, i - 1] + drift + diff)

        return x_paths


# Aliases
ReflectedOU = ReflectedOrnsteinUhlenbeck
