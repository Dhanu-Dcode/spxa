r"""
Nelson (1990) continuous-time GARCH(1,1) diffusion limit process.

The continuous-time GARCH diffusion model arises as the weak diffusion limit
of discrete-time GARCH(1,1) processes (Nelson, 1990):
    dv_t = κ(θ - v_t) dt + σ v_t dW_t,   v_0 > 0

where:
  κ > 0 : Speed of mean reversion
  θ > 0 : Long-term variance level
  σ > 0 : Volatility-of-variance parameter
  W_t   : Standard 1-dimensional Brownian motion

Theoretical Properties
----------------------
1. Linear Diffusion Coefficient:
   Unlike CIR (σ √v) and 3/2 (σ v^{3/2}), the GARCH diffusion features a linear
   diffusion coefficient g(v) = σ v. At the origin v = 0, the drift is κθ > 0
   while the diffusion vanishes, making v = 0 an entrance boundary. Thus v_t > 0
   holds almost surely for all t > 0.

2. Exact Stationary Distribution (Inverse-Gamma):
   The stationary distribution solves the stationary Fokker–Planck equation
   and is an Inverse-Gamma distribution:
       v_∞ ~ Inv-Gamma(α = 1 + 2κ / σ², β = 2κθ / σ²)
   with stationary mean:
       E[v_∞] = θ
   and stationary variance (finite if 2κ > σ², i.e. α > 2):
       Var(v_∞) = σ² θ² / (2κ - σ²)

3. Exact Conditional Moments via Moment ODEs:
   Conditional expectation satisfies:
       E[v_t | v_0] = θ + (v_0 - θ) e^{-κt}
   The conditional second moment E[v_t² | v_0] satisfies a closed-form linear ODE:
       d/dt E[v_t²] = 2κθ E[v_t] - (2κ - σ²) E[v_t²]

Exactness Level: MOMENT_PROPAGATION (exact analytical conditional and stationary moments).

References
----------
Nelson, D.B. (1990). ARCH models as diffusion approximations.
*Journal of Econometrics*, 45(1-2), 7–38.

Andersen, T.G. & Bollerslev, T. (1998). Answering the skeptics: Yes, standard
volatility models do provide accurate forecasts. *International Economic Review*, 39(4), 885–905.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy.stats import invgamma

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class GARCHDiffusion(Process):
    r"""
    Continuous-time GARCH(1,1) diffusion limit (Nelson, 1990).

    dv_t = κ(θ - v_t) dt + σ v_t dW_t,   v_0 > 0.

    Parameters
    ----------
    kappa : float
        Mean-reversion rate κ > 0.
    theta : float
        Long-term variance parameter θ > 0.
    sigma : float
        Volatility parameter σ > 0.
    v0 : float, optional
        Initial variance v_0 > 0 (defaults to theta).
    """

    def __init__(
        self,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma: float = 0.3,
        v0: Optional[float] = None,
    ) -> None:
        if kappa <= 0:
            raise ValueError(f"kappa must be strictly positive, got {kappa}")
        if theta <= 0:
            raise ValueError(f"theta must be strictly positive, got {theta}")
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")

        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)

        if v0 is None:
            self.v0 = self.theta
        else:
            if v0 <= 0:
                raise ValueError(f"v0 must be strictly positive, got {v0}")
            self.v0 = float(v0)

        # Stationary shape and scale
        self.alpha_stat = 1.0 + 2.0 * self.kappa / (self.sigma**2)
        self.beta_stat = 2.0 * self.kappa * self.theta / (self.sigma**2)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"GARCHDiffusion(kappa={self.kappa}, theta={self.theta}, sigma={self.sigma}, v0={self.v0})",
            math_note=(
                f"dv_t = {self.kappa}({self.theta} - v_t)dt + {self.sigma} v_t dW_t; "
                f"v_0={self.v0}; stationary Inv-Gamma(α={self.alpha_stat:.4f}, β={self.beta_stat:.4f})"
            ),
            reference="Nelson (1990), Journal of Econometrics 45(1-2), 7–38",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Stationary Moments & Density
    # ------------------------------------------------------------------

    @property
    def stationary_shape(self) -> float:
        r"""Shape parameter α = 1 + 2κ / σ² of the stationary Inverse-Gamma law."""
        return self.alpha_stat

    @property
    def stationary_scale(self) -> float:
        r"""Scale parameter β = 2κθ / σ² of the stationary Inverse-Gamma law."""
        return self.beta_stat

    @property
    def stationary_mean(self) -> float:
        r"""Stationary expectation E[v_∞] = θ."""
        return self.theta

    @property
    def stationary_variance(self) -> float:
        r"""
        Stationary variance Var(v_∞) = σ² θ² / (2κ - σ²).
        Finite if and only if 2κ > σ².
        """
        if 2.0 * self.kappa <= self.sigma**2:
            return float("inf")
        return (self.sigma**2 * self.theta**2) / (2.0 * self.kappa - self.sigma**2)

    def stationary_density(self, v: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Probability density function of the stationary Inverse-Gamma distribution."""
        v_arr = np.asarray(v, dtype=np.float64)
        pdf = invgamma.pdf(v_arr, a=self.alpha_stat, scale=self.beta_stat)
        if np.isscalar(v):
            return float(pdf)
        return pdf

    # ------------------------------------------------------------------
    # Conditional Moments
    # ------------------------------------------------------------------

    def conditional_mean(self, t: float, v0: Optional[float] = None) -> float:
        r"""Expected variance E[v_t | v_0] = θ + (v_0 - θ) e^{-κt}."""
        if t <= 0:
            return self.v0 if v0 is None else float(v0)
        cur_v0 = self.v0 if v0 is None else float(v0)
        return self.theta + (cur_v0 - self.theta) * math.exp(-self.kappa * t)

    def conditional_second_moment(self, t: float, v0: Optional[float] = None) -> float:
        r"""
        Exact conditional second moment E[v_t² | v_0] from linear moment ODE.
        """
        if t <= 0:
            cur_v0 = self.v0 if v0 is None else float(v0)
            return cur_v0**2

        cur_v0 = self.v0 if v0 is None else float(v0)
        k, th, sig = self.kappa, self.theta, self.sigma
        gamma = 2.0 * k - sig**2

        exp_gamma = math.exp(-gamma * t)
        exp_k = math.exp(-k * t)

        if abs(gamma - k) > 1e-10:
            term1 = exp_gamma * (cur_v0**2)
            term2 = (2.0 * k * th**2 / gamma) * (1.0 - exp_gamma)
            term3 = (2.0 * k * th * (cur_v0 - th) / (gamma - k)) * (exp_k - exp_gamma)
            return float(term1 + term2 + term3)
        else:
            term1 = exp_k * (cur_v0**2)
            term2 = (2.0 * th**2) * (1.0 - exp_k)
            term3 = 2.0 * k * th * (cur_v0 - th) * t * exp_k
            return float(term1 + term2 + term3)

    def conditional_variance(self, t: float, v0: Optional[float] = None) -> float:
        r"""Conditional variance Var(v_t | v_0) = E[v_t²] - (E[v_t])²."""
        if t <= 0:
            return 0.0
        m1 = self.conditional_mean(t, v0)
        m2 = self.conditional_second_moment(t, v0)
        return float(max(m2 - m1**2, 0.0))

    # ------------------------------------------------------------------
    # Process Overrides & Vectorized Simulation
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "GARCHDiffusion is a mean-reverting non-linear diffusion with state-dependent "
            "diffusion σ v_t; it does not admit a stationary Lévy–Khintchine triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=bool(2.0 * self.kappa > self.sigma**2),
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=self.alpha_stat,
            is_subordinator=False,
            hurst_index=None,
            notes=[
                "Nelson (1990) GARCH(1,1) continuous-time diffusion limit.",
                "Stationary distribution is Inverse-Gamma.",
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
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Vectorized simulation of GARCH diffusion paths using log-Euler scheme.
        d ln v_t = (κθ / v_t - κ - 0.5 σ²) dt + σ dW_t
        ensuring guaranteed positivity v_t > 0 along all sample paths.
        """
        paths_count = (
            n_paths if n_paths is not None else (num_paths if num_paths is not None else 1)
        )
        steps = n_steps if n_steps is not None else (num_steps if num_steps is not None else 100)
        final_t = T if T is not None else (t_max if t_max is not None else 1.0)

        if t_grid is None:
            grid = np.linspace(0.0, final_t, steps + 1)
        elif isinstance(t_grid, (int, float)):
            grid = np.linspace(0.0, float(t_grid), steps + 1)
        else:
            grid = np.asarray(t_grid, dtype=np.float64)

        if grid.ndim != 1 or len(grid) < 2:
            raise ValueError("t_grid must be a 1-D array with at least 2 points")
        if rng is None:
            rng = np.random.default_rng()

        num_pts = len(grid)
        paths = np.zeros((paths_count, num_pts), dtype=np.float64)
        paths[:, 0] = self.v0

        log_v = np.full(paths_count, math.log(self.v0), dtype=np.float64)

        for i in range(1, num_pts):
            dt = grid[i] - grid[i - 1]
            if dt <= 0:
                raise ValueError("t_grid must be strictly increasing")

            sqrt_dt = math.sqrt(dt)
            z = rng.standard_normal(size=paths_count)

            v_curr = np.exp(log_v)
            drift = (self.kappa * self.theta / v_curr) - self.kappa - 0.5 * (self.sigma**2)
            log_v += drift * dt + self.sigma * sqrt_dt * z
            paths[:, i] = np.exp(log_v)

        return paths

    def __repr__(self) -> str:
        return (
            f"GARCHDiffusion(kappa={self.kappa}, theta={self.theta}, "
            f"sigma={self.sigma}, v0={self.v0})"
        )


# Aliases
ContinuousGARCH = GARCHDiffusion
GARCHDiffusionProcess = GARCHDiffusion
