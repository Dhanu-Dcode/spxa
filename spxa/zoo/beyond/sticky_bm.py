r"""
Feller's Sticky Brownian Motion (Feller, 1952; Chitashvili, 1989).

A continuous diffusion on [0, ∞) that behaves like standard Brownian motion
in the interior (0, ∞), but is delayed at the boundary {0}, spending positive
Lebesgue measure time there (so-called "sticky" boundary behavior).

The infinitesimal generator is:
    Af(x) = (1/2) σ² f''(x)   for x > 0
with boundary condition at x = 0:
    (1/2) σ² f''(0+) = ρ f'(0+)

where ρ > 0 is the stickiness parameter:
- As ρ → ∞: Recovers instantly reflected Brownian motion (zero time at 0).
- As ρ → 0: Recovers absorbing Brownian motion (trapped forever at 0).
- For 0 < ρ < ∞: The process visits 0, stays for an accumulation of time,
  and then leaves via excursions.

Theoretical Properties
----------------------
1. Discrete Probability Mass at 0:
   For any t > 0, the probability of being at the boundary {0} is strictly positive:
       P(X_t = 0 | X_0 = x_0) = exp(2ρ x_0 / σ² + 2ρ² t / σ²) erfc(x_0 / (σ√(2t)) + ρ√(2t) / σ)

Exactness Level: MOMENT_PROPAGATION.

References
----------
Feller, W. (1952). The parabolic differential equations and the associated
semi-groups of transformations. *Annals of Mathematics*, 468–519.

Chitashvili, R. (1989). On the smooth fit boundary conditions for sticky diffusions.
*Stochastics and Stochastic Reports*, 28(2), 103–127.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy.special import erfc

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class StickyBrownianMotion(Process):
    r"""
    Feller's Sticky Brownian Motion on [0, ∞).

    Parameters
    ----------
    sigma : float
        Diffusion coefficient σ > 0.
    rho : float
        Stickiness parameter ρ > 0 (boundary holding rate).
    x0 : float, optional
        Initial value X_0 ≥ 0 (defaults to 0.0).
    """

    def __init__(
        self,
        sigma: float = 1.0,
        rho: float = 1.0,
        x0: float = 0.0,
    ) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")
        if rho <= 0:
            raise ValueError(f"rho must be strictly positive, got {rho}")
        if x0 < 0:
            raise ValueError(f"x0 must be non-negative on [0, ∞), got {x0}")

        self.sigma = float(sigma)
        self.rho = float(rho)
        self.x0 = float(x0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"StickyBrownianMotion(σ={self.sigma}, ρ={self.rho}, x0={self.x0})",
            math_note=f"Af(x)=½σ²f''(x) on (0,∞) with sticky boundary ½σ²f''(0)=ρf'(0)",
            reference="Feller (1952), Ann. Math. 468–519; Chitashvili (1989)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Exact Boundary Mass
    # ------------------------------------------------------------------

    def boundary_probability(self, t: float, x0: Optional[float] = None) -> float:
        r"""
        Exact discrete probability mass P(X_t = 0 | X_0 = x_0).
        """
        if t <= 0:
            return 1.0 if (self.x0 if x0 is None else x0) == 0.0 else 0.0

        init_x = self.x0 if x0 is None else float(x0)
        s = self.sigma
        r = self.rho

        arg_exp = (2.0 * r * init_x) / (s**2) + (2.0 * (r**2) * t) / (s**2)
        arg_erfc = init_x / (s * math.sqrt(2.0 * t)) + (r * math.sqrt(2.0 * t)) / s

        if arg_exp > 500.0:
            # Use asymptotic expansion of exp(z^2) erfc(z) ~ 1 / (z sqrt(pi))
            return 1.0 / (arg_erfc * math.sqrt(math.pi))

        prob = math.exp(arg_exp) * erfc(arg_erfc)
        return min(max(prob, 0.0), 1.0)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "StickyBrownianMotion has non-linear boundary delay at {0} and does not "
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
                "Feller's Sticky Brownian Motion with positive boundary holding time.",
                f"Stickiness parameter ρ = {self.rho}.",
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
    # Vectorized Simulation via Skorokhod Local Time Thinning
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "sticky_skorokhod",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of Sticky Brownian Motion using Skorokhod reflection
        with stochastic boundary delay.
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
            if dt <= 0:
                x_paths[:, i] = x_curr
                continue

            sqrt_dt = math.sqrt(dt)
            z = rng.standard_normal(paths_count)

            # Check sticky trapping at zero: if at 0, stick with probability exp(-ρ dt)
            at_zero = x_curr <= 1e-12
            u_stick = rng.uniform(0.0, 1.0, paths_count)
            remains_stuck = at_zero & (u_stick < math.exp(-self.rho * dt))

            # Normal reflection update for non-stuck paths
            proposal = x_curr + self.sigma * sqrt_dt * z
            # Reflected at 0
            reflected = np.abs(proposal)

            x_curr = np.where(remains_stuck, 0.0, reflected)
            x_paths[:, i] = x_curr

        return x_paths


# Aliases
StickyBM = StickyBrownianMotion
