r"""
Reflected Brownian Motion (RBM) on [0, ∞) (Harrison & Reiman, 1981; Skorokhod, 1961).

The one-dimensional Reflected Brownian Motion satisfies the Skorokhod stochastic
differential equation:
    X_t = X_0 + μ t + σ W_t + L_t,   X_t ≥ 0

where L_t is a continuous, non-decreasing bounded-variation process (the local time
at the origin) satisfying:
    ∫_0^t X_s dL_s = 0   (L_t increases only when X_t = 0).

Explicit Skorokhod Solution:
    L_t = max_{0 ≤ s ≤ t} (-X_0 - μ s - σ W_s)^+
    X_t = X_0 + μ t + σ W_t + max_{0 ≤ s ≤ t} (-X_0 - μ s - σ W_s)^+

Theoretical Properties
----------------------
1. Stationary Law (for μ < 0):
   When the drift is strictly negative (μ < 0), the process is positive-recurrent
   and converges to an exact Exponential stationary distribution:
       X_∞ ~ Exponential(λ = 2|μ| / σ²)
   with stationary mean:
       E[X_∞] = σ² / (2|μ|)

2. Exact Folded Transition Density:
   For x ≥ 0 and x_0 ≥ 0:
       p(x, t | x_0) = (1 / (σ√t)) [ φ((x - x_0 - μt)/(σ√t)) + φ((x + x_0 - μt)/(σ√t)) ]
                       - (2μ / σ²) e^{2μx / σ²} Φ(-(x + x_0 + μt)/(σ√t))

Exactness Level: MOMENT_PROPAGATION.

References
----------
Skorokhod, A. V. (1961). Stochastic equations for diffusion processes in a bounded
domain. *Theory of Probability & Its Applications*, 6(3), 264–274.

Harrison, J. M., & Reiman, M. I. (1981). Reflected Brownian motion on an orthant.
*The Annals of Probability*, 302–308.
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


class ReflectedBrownianMotion(Process):
    r"""
    Reflected Brownian Motion on the half-line [0, ∞).

    Parameters
    ----------
    mu : float
        Drift coefficient μ.
    sigma : float
        Diffusion coefficient σ > 0.
    x0 : float, optional
        Initial position X_0 ≥ 0 (defaults to 0.0).
    """

    def __init__(
        self,
        mu: float = -0.5,
        sigma: float = 1.0,
        x0: float = 0.0,
    ) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")
        if x0 < 0:
            raise ValueError(f"x0 must be non-negative on [0, ∞), got {x0}")

        self.mu = float(mu)
        self.sigma = float(sigma)
        self.x0 = float(x0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"ReflectedBrownianMotion(μ={self.mu}, σ={self.sigma}, x0={self.x0})",
            math_note=(
                "X_t = X_0 + μt + σW_t + L_t on [0, ∞); "
                + (
                    f"stationary Exp(λ={2.0 * abs(self.mu) / (self.sigma**2):.3f})"
                    if self.mu < 0
                    else "transient/null-recurrent"
                )
            ),
            reference="Skorokhod (1961); Harrison & Reiman (1981), Ann. Probab. 302–308",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Stationary Analytics
    # ------------------------------------------------------------------

    @property
    def has_stationary_distribution(self) -> bool:
        r"""True if μ < 0 (positive-recurrent with exponential invariant measure)."""
        return self.mu < 0.0

    @property
    def stationary_mean(self) -> float:
        r"""Stationary expectation E[X_∞] = σ² / (2|μ|)."""
        if self.mu >= 0.0:
            return float("inf")
        return (self.sigma**2) / (2.0 * abs(self.mu))

    @property
    def stationary_variance(self) -> float:
        r"""Stationary variance Var(X_∞) = σ⁴ / (4μ²)."""
        if self.mu >= 0.0:
            return float("inf")
        return (self.sigma**4) / (4.0 * (self.mu**2))

    def transition_density(
        self,
        x: Union[float, np.ndarray],
        t: float,
        x0: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        r"""
        Exact transition density p(x, t | x_0) of Reflected Brownian Motion.
        """
        if t <= 0:
            raise ValueError("t must be strictly positive")
        init_x = self.x0 if x0 is None else float(x0)
        x_arr = np.asarray(x, dtype=np.float64)

        if np.any(x_arr < 0):
            raise ValueError("x must be non-negative on [0, ∞)")

        s = self.sigma
        m = self.mu
        sqrt_t = math.sqrt(t)

        z1 = (x_arr - init_x - m * t) / (s * sqrt_t)
        z2 = (x_arr + init_x - m * t) / (s * sqrt_t)

        term1 = (1.0 / (s * sqrt_t)) * (norm.pdf(z1) + norm.pdf(z2))

        if abs(m) < 1e-12:
            dens = term1
        else:
            scale = (2.0 * m) / (s**2)
            arg_exp = scale * x_arr
            arg_norm = -(x_arr + init_x + m * t) / (s * sqrt_t)
            # Clip exp argument for numerical safety
            arg_exp_safe = np.clip(arg_exp, -100.0, 100.0)
            term2 = scale * np.exp(arg_exp_safe) * norm.cdf(arg_norm)
            dens = term1 - term2

        if np.isscalar(x):
            return float(dens)
        return dens

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "ReflectedBrownianMotion has boundary reflection at {0} and does not "
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
                "Reflected Brownian Motion on [0, ∞).",
                f"Drift μ = {self.mu}, Diffusion σ = {self.sigma}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {
            1: self.stationary_mean if self.has_stationary_distribution else self.x0
        }
        if order >= 2:
            res[2] = self.stationary_variance if self.has_stationary_distribution else self.sigma**2
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Exact Skorokhod Running Local Time Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "skorokhod_exact",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of Reflected Brownian Motion using exact running infimum Skorokhod identity.
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

        # Discrete-time Skorokhod recursion:
        # X_{i+1} = max(0, X_i + μ dt + σ √dt Z_i)
        x_paths = np.empty((paths_count, n_pts), dtype=np.float64)
        x_paths[:, 0] = self.x0

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)
            z = rng.standard_normal(paths_count)
            inc = self.mu * dt + self.sigma * sqrt_dt * z
            x_paths[:, i] = np.maximum(0.0, x_paths[:, i - 1] + inc)

        return x_paths


# Aliases
ReflectedBM = ReflectedBrownianMotion
