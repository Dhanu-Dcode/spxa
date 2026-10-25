r"""
Bilinear Diffusion process.

The Bilinear Diffusion model features an affine drift and affine diffusion coefficient:
    dX_t = (a + b X_t) dt + (c + d X_t) dW_t,   X_0 ∈ ℝ

where:
  a ∈ ℝ : Constant drift component
  b < 0 : Mean-reversion slope (speed κ = -b > 0, long-term mean θ = -a/b)
  c ≥ 0 : Additive noise level
  d ≥ 0 : Multiplicative noise sensitivity (c + d > 0)
  W_t   : Standard 1-dimensional Brownian motion

Theoretical Properties
----------------------
1. Pearson Diffusion Family:
   The bilinear diffusion belongs to the family of Pearson diffusions (Forman & Sørensen, 2008),
   characterized by polynomial drift and quadratic diffusion coefficient. It nests:
   - Ornstein–Uhlenbeck when d = 0
   - Geometric Brownian Motion when a = 0, c = 0
   - Continuous-time GARCH when a = κθ, b = -κ, c = 0, d = σ

2. Exact Conditional Moments:
   The conditional expectation is linear:
       E[X_t | X_0] = X_0 e^{bt} + (a / b) (e^{bt} - 1)
   The conditional second moment satisfies the linear ODE:
       d/dt E[X_t²] = (2b + d²) E[X_t²] + 2(a + cd) E[X_t] + c²
   yielding an exact closed-form analytical expression for Var(X_t | X_0).

3. Stationary Moments:
   When b < 0 and 2b + d² < 0, a stationary distribution exists with:
       E[X_∞] = -a / b
       Var(X_∞) = (d² (E[X_∞])² + 2cd E[X_∞] + c²) / (-2b - d²)

4. Exact Boundary Behavior:
   For d > 0, the state space is bounded below by x_min = -c/d, where the diffusion
   coefficient vanishes, forming an entrance or reflecting boundary depending on a - bc/d.

Exactness Level: MOMENT_PROPAGATION (exact analytical first and second conditional moments).

References
----------
Bibby, B.M. & Sørensen, M. (1995). Martingale estimating functions for discretely
observed diffusion processes. *Bernoulli*, 1(1-2), 17–39.

Forman, J.L. & Sørensen, M. (2008). The Pearson diffusions: A class of solvable
diffusions. *Scandinavian Journal of Statistics*, 35(3), 438–465.

Wong, E. (1964). The construction of a class of stationary Markoff processes.
*Proc. Sympos. Appl. Math.*, 16, 264–276.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class BilinearDiffusion(Process):
    r"""
    Bilinear Diffusion process with affine drift and affine volatility.

    dX_t = (a + b X_t) dt + (c + d X_t) dW_t.

    Parameters
    ----------
    a : float
        Drift intercept a.
    b : float
        Drift slope b < 0 (mean reversion).
    c : float
        Additive diffusion parameter c >= 0.
    d : float
        Multiplicative diffusion parameter d >= 0 (c + d > 0).
    x0 : float, optional
        Initial value X_0 (defaults to -a/b).
    """

    def __init__(
        self,
        a: float = 0.5,
        b: float = -1.0,
        c: float = 0.2,
        d: float = 0.3,
        x0: Optional[float] = None,
    ) -> None:
        if b >= 0:
            raise ValueError(f"Slope 'b' must be strictly negative for mean-reversion, got {b}")
        if c < 0:
            raise ValueError(f"Additive diffusion 'c' must be non-negative, got {c}")
        if d < 0:
            raise ValueError(f"Multiplicative diffusion 'd' must be non-negative, got {d}")
        if c == 0 and d == 0:
            raise ValueError("At least one of c or d must be strictly positive")

        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)

        # Long-term mean theta = -a / b
        self.theta = -self.a / self.b

        if x0 is None:
            self.x0 = self.theta
        else:
            self.x0 = float(x0)

        # Minimum state boundary if d > 0: x >= -c/d
        if self.d > 0:
            self.lower_bound = -self.c / self.d
            if self.x0 < self.lower_bound:
                raise ValueError(
                    f"Initial value x0={self.x0} cannot be less than lower boundary {self.lower_bound}"
                )
        else:
            self.lower_bound = -float("inf")

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"BilinearDiffusion(a={self.a}, b={self.b}, c={self.c}, d={self.d}, x0={self.x0})",
            math_note=(
                f"dX_t = ({self.a} + {self.b}X_t)dt + ({self.c} + {self.d}X_t)dW_t; "
                f"θ={self.theta:.4f}, lower_bound={self.lower_bound:.4f}"
            ),
            reference="Bibby & Sørensen (1995), Bernoulli 1(1-2), 17–39",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Analytical Moments
    # ------------------------------------------------------------------

    def conditional_mean(self, t: float, x0: Optional[float] = None) -> float:
        r"""
        Expected value E[X_t | X_0] = X_0 e^{bt} + (a / b)(e^{bt} - 1).
        """
        if t <= 0:
            return self.x0 if x0 is None else float(x0)
        cur_x0 = self.x0 if x0 is None else float(x0)
        exp_bt = math.exp(self.b * t)
        return float(cur_x0 * exp_bt + (self.a / self.b) * (exp_bt - 1.0))

    def conditional_second_moment(self, t: float, x0: Optional[float] = None) -> float:
        r"""
        Exact conditional second moment E[X_t² | X_0] from linear moment ODE.
        """
        if t <= 0:
            cur_x0 = self.x0 if x0 is None else float(x0)
            return cur_x0**2

        cur_x0 = self.x0 if x0 is None else float(x0)
        a, b, c, d = self.a, self.b, self.c, self.d
        lam = 2.0 * b + d**2

        # ODE: d/dt M2(t) = lam * M2(t) + alpha * e^{bt} + beta
        # where alpha = 2(a + c*d)*(x0 - theta)
        #       beta  = 2(a + c*d)*theta + c^2
        alpha = 2.0 * (a + c * d) * (cur_x0 - self.theta)
        beta = 2.0 * (a + c * d) * self.theta + c**2

        exp_lam_t = math.exp(lam * t)
        term_init = (cur_x0**2) * exp_lam_t

        # Integral of e^{lam(t-s)} * beta ds = (beta / (-lam)) * (1 - e^{lam*t}) = (beta / lam) * (e^{lam*t} - 1)
        if abs(lam) > 1e-12:
            term_beta = (beta / lam) * (exp_lam_t - 1.0)
        else:
            term_beta = beta * t

        # Integral of e^{lam(t-s)} * alpha * e^{bs} ds = alpha * e^{lam*t} * ∫_0^t e^{(b - lam)s} ds
        denom = b - lam
        if abs(denom) > 1e-12:
            term_alpha = alpha * (math.exp(b * t) - exp_lam_t) / denom
        else:
            term_alpha = alpha * t * exp_lam_t

        return float(term_init + term_beta + term_alpha)

    def conditional_variance(self, t: float, x0: Optional[float] = None) -> float:
        r"""
        Conditional variance Var(X_t | X_0) = E[X_t²] - (E[X_t])².
        """
        if t <= 0:
            return 0.0
        m1 = self.conditional_mean(t, x0)
        m2 = self.conditional_second_moment(t, x0)
        return float(max(m2 - m1**2, 0.0))

    @property
    def stationary_mean(self) -> float:
        r"""Stationary expectation E[X_∞] = -a / b."""
        return self.theta

    @property
    def stationary_variance(self) -> float:
        r"""
        Stationary variance Var(X_∞) = (d² θ² + 2cd θ + c²) / (-2b - d²).
        Finite if and only if 2b + d² < 0.
        """
        denom = -2.0 * self.b - self.d**2
        if denom <= 0:
            return float("inf")
        numer = (self.d**2) * (self.theta**2) + 2.0 * self.c * self.d * self.theta + self.c**2
        return float(numer / denom)

    # ------------------------------------------------------------------
    # Process Overrides & Vectorized Simulation
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "BilinearDiffusion has state-dependent drift and diffusion (c + d X_t); "
            "it does not possess stationary independent increments."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=bool(2.0 * self.b + self.d**2 < 0),
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=None,
            notes=[
                "Bilinear diffusion with affine drift and affine volatility.",
                "Member of the Pearson diffusion family.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {
            1: self.conditional_mean(1.0),
            2: self.conditional_variance(1.0),
        }
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    def simulate(
        self,
        t_grid: np.ndarray,
        n_paths: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""
        High-order Milstein simulation of Bilinear Diffusion sample paths.
        """
        t_grid = np.asarray(t_grid, dtype=np.float64)
        if t_grid.ndim != 1 or len(t_grid) < 2:
            raise ValueError("t_grid must be a 1-D array with at least 2 points")
        if rng is None:
            rng = np.random.default_rng()

        n_steps = len(t_grid)
        paths = np.zeros((n_paths, n_steps), dtype=np.float64)
        paths[:, 0] = self.x0

        for i in range(1, n_steps):
            dt = t_grid[i] - t_grid[i - 1]
            if dt <= 0:
                raise ValueError("t_grid must be strictly increasing")

            sqrt_dt = math.sqrt(dt)
            z = rng.standard_normal(size=n_paths)

            x_curr = paths[:, i - 1]
            drift = self.a + self.b * x_curr
            sigma = self.c + self.d * x_curr

            # Milstein correction: 0.5 * sigma * sigma' * (z^2 - 1) * dt
            # sigma' = d
            milstein = 0.5 * sigma * self.d * (z**2 - 1.0) * dt

            x_next = x_curr + drift * dt + sigma * sqrt_dt * z + milstein
            if self.d > 0:
                x_next = np.maximum(x_next, self.lower_bound + 1e-12)

            paths[:, i] = x_next

        return paths

    def __repr__(self) -> str:
        return f"BilinearDiffusion(a={self.a}, b={self.b}, c={self.c}, d={self.d}, x0={self.x0})"


# Aliases
BilinearDiffusionProcess = BilinearDiffusion
AffineDiffusion = BilinearDiffusion
