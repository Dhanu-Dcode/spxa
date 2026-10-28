r"""
Skew Brownian Motion (Itô & McKean, 1965; Walsh, 1978).

A diffusion process X_t that behaves like standard Brownian motion away from zero,
but whose excursions from the origin choose the positive real line with probability
p ∈ [0, 1] and the negative real line with probability 1 - p:

    dX_t = σ dW_t + (2p - 1) dL_t^0(X)

where L_t^0(X) is the symmetric local time of X at the origin.

Theoretical Properties
----------------------
1. Limiting Regimes:
   - When p = 1/2: Recovers standard symmetric Brownian motion.
   - When p = 1: Recovers Reflected Brownian motion on [0, ∞).
   - When p = 0: Recovers Reflected Brownian motion on (-∞, 0].

2. Exact Asymmetric Transition Density (Walsh, 1978):
   For x_0 > 0 and y > 0:
       p(y, t | x_0) = (1 / (σ√t)) [ φ((y - x_0)/(σ√t)) + (2p - 1) φ((y + x_0)/(σ√t)) ]
   For x_0 > 0 and y < 0:
       p(y, t | x_0) = (2(1 - p) / (σ√t)) φ((|y| + x_0)/(σ√t))
   For x_0 < 0 and y > 0:
       p(y, t | x_0) = (2p / (σ√t)) φ((y + |x_0|)/(σ√t))
   For x_0 < 0 and y < 0:
       p(y, t | x_0) = (1 / (σ√t)) [ φ((y - x_0)/(σ√t)) + (1 - 2p) φ((|y| + |x_0|)/(σ√t)) ]

Exactness Level: MOMENT_PROPAGATION.

References
----------
Itô, K., & McKean, H. P. (1965). *Diffusion Processes and Their Sample Paths*.
Springer-Verlag.

Walsh, J. B. (1978). A diffusion with a discontinuous local time.
*Astérisque*, 52(53), 37–45.
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


class SkewBrownianMotion(Process):
    r"""
    Skew Brownian Motion with asymmetry probability parameter p ∈ [0, 1].

    Parameters
    ----------
    p : float
        Probability of positive excursions p ∈ [0, 1] (p=0.5 is symmetric BM).
    sigma : float
        Diffusion coefficient σ > 0.
    x0 : float, optional
        Initial position X_0 (defaults to 0.0).
    """

    def __init__(
        self,
        p: float = 0.7,
        sigma: float = 1.0,
        x0: float = 0.0,
    ) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")

        self.p = float(p)
        self.sigma = float(sigma)
        self.x0 = float(x0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"SkewBrownianMotion(p={self.p}, σ={self.sigma}, x0={self.x0})",
            math_note=f"dX_t = σ dW_t + (2p - 1) dL_t^0(X); skew probability p={self.p}",
            reference="Itô & McKean (1965); Walsh (1978), Astérisque 52, 37–45",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Exact Transition Density
    # ------------------------------------------------------------------

    def transition_density(
        self,
        y: Union[float, np.ndarray],
        t: float,
        x0: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        r"""
        Exact transition density p(y, t | x_0) of Skew Brownian Motion.
        """
        if t <= 0:
            raise ValueError("t must be strictly positive")
        init_x = self.x0 if x0 is None else float(x0)
        y_arr = np.asarray(y, dtype=np.float64)

        s = self.sigma
        sqrt_t = math.sqrt(t)
        p = self.p

        if init_x >= 0:
            # x0 >= 0
            pos_dens = (1.0 / (s * sqrt_t)) * (
                norm.pdf((y_arr - init_x) / (s * sqrt_t))
                + (2.0 * p - 1.0) * norm.pdf((y_arr + init_x) / (s * sqrt_t))
            )
            neg_dens = (2.0 * (1.0 - p) / (s * sqrt_t)) * norm.pdf(
                (np.abs(y_arr) + init_x) / (s * sqrt_t)
            )
            dens = np.where(y_arr >= 0, pos_dens, neg_dens)
        else:
            # x0 < 0
            abs_x0 = abs(init_x)
            pos_dens = (2.0 * p / (s * sqrt_t)) * norm.pdf((y_arr + abs_x0) / (s * sqrt_t))
            neg_dens = (1.0 / (s * sqrt_t)) * (
                norm.pdf((y_arr - init_x) / (s * sqrt_t))
                + (1.0 - 2.0 * p) * norm.pdf((np.abs(y_arr) + abs_x0) / (s * sqrt_t))
            )
            dens = np.where(y_arr >= 0, pos_dens, neg_dens)

        if np.isscalar(y):
            return float(dens)
        return dens

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "SkewBrownianMotion has asymmetric boundary push at 0 and does not "
            "admit an independent Lévy–Khintchine triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=(self.p == 0.5),
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=0.5,
            notes=[
                "Skew Brownian Motion with excursion asymmetry.",
                f"Excursion probability to positive half-line p = {self.p}.",
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
    # Exact Excursion Sign Sampling
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "skew_reflection",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of Skew Brownian Motion via excursion sign sampling.
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

        x_curr = np.full(paths_count, self.x0, dtype=np.float64)

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)
            z = rng.standard_normal(paths_count)

            prop = x_curr + self.sigma * sqrt_dt * z
            # Check sign flips across zero
            crossed = (x_curr * prop < 0.0) | (np.abs(x_curr) <= 1e-12)

            # Assign excursion sign according to p
            u = rng.uniform(0.0, 1.0, paths_count)
            assigned_sign = np.where(u < self.p, 1.0, -1.0)

            x_curr = np.where(crossed, assigned_sign * np.abs(prop), prop)
            x_paths[:, i] = x_curr

        return x_paths


# Aliases
SkewBM = SkewBrownianMotion
