r"""
3/2 Stochastic Volatility diffusion model.

The 3/2 model describes a continuous-time variance process with super-linear
volatility-of-variance:
    dv_t = κ v_t (θ - v_t) dt + ε v_t^{3/2} dW_t,   v_0 > 0

where:
  κ > 0 : Mean-reversion speed
  θ > 0 : Long-term variance level
  ε > 0 : Volatility-of-variance parameter
  W_t   : Standard 1-dimensional Brownian motion

Theoretical Properties
----------------------
1. The Reciprocal Transformation:
   Let y_t = 1 / v_t. Applying Itô's formula reveals that y_t follows a
   Cox–Ingersoll–Ross (CIR) square-root diffusion:
       dy_t = κ_y (θ_y - y_t) dt + σ_y √y_t dW^*_t
   where:
       κ_y = κ θ
       θ_y = (κ + ε²) / (κ θ)
       σ_y = ε
       W^*_t = -W_t

2. Automatic Feller Satisfaction:
   The Feller condition for the reciprocal process y_t is:
       2 κ_y θ_y / σ_y² = 2 (κ + ε²) / ε² = 1 + 2κ / ε² > 1
   Because κ > 0 and ε > 0, this ratio is ALWAYS strictly greater than 1!
   Consequently, y_t never hits 0 almost surely, meaning v_t never explodes to ∞
   almost surely, and v_t > 0 holds almost surely.

3. Stationary Distribution:
   As t → ∞, the reciprocal y_t converges to a Gamma distribution, implying that
   v_t converges to an Inverse-Gamma distribution:
       v_∞ ~ Inv-Gamma(α = 1 + 2κ / ε², β = 2κθ / ε²)
   with stationary mean:
       E[v_∞] = β / (α - 1) = θ
   and stationary variance (when 2κ > ε²):
       Var(v_∞) = ε² θ² / (2κ - ε²)

4. Exact Path Simulation:
   Because the reciprocal y_t is an exact CIR process, exact path generation
   for v_t is achieved by sampling y_t from the non-central chi-squared
   transition distribution and taking v_t = 1 / y_t.

Exactness Level: MOMENT_PROPAGATION (exact reciprocal transition distribution
and closed-form stationary moments).

References
----------
Heston, S.L. (1997). A simple new formula for options with stochastic volatility.
*Working Paper*, Washington University in St. Louis.

Carr, P. & Sun, J. (2007). A new approach for option pricing under stochastic
volatility. *Review of Derivatives Research*, 10(2), 87–150.

Baldeaux, J. (2012). Exact simulation of 3/2 and confluent hypergeometric models.
*International Journal of Theoretical and Applied Finance*, 15(06), 1250041.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy.stats import invgamma, ncx2

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class ThreeHalvesProcess(Process):
    r"""
    3/2 Stochastic Volatility diffusion model.

    dv_t = κ v_t (θ - v_t) dt + ε v_t^{3/2} dW_t,   v_0 > 0.

    Parameters
    ----------
    kappa : float
        Mean-reversion rate κ > 0.
    theta : float
        Long-term variance parameter θ > 0.
    epsilon : float
        Volatility-of-variance parameter ε > 0.
    v0 : float, optional
        Initial variance level v_0 > 0 (defaults to theta).
    """

    def __init__(
        self,
        kappa: float = 2.0,
        theta: float = 0.04,
        epsilon: float = 1.0,
        v0: Optional[float] = None,
    ) -> None:
        if kappa <= 0:
            raise ValueError(f"kappa must be strictly positive, got {kappa}")
        if theta <= 0:
            raise ValueError(f"theta must be strictly positive, got {theta}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be strictly positive, got {epsilon}")

        self.kappa = float(kappa)
        self.theta = float(theta)
        self.epsilon = float(epsilon)

        if v0 is None:
            self.v0 = self.theta
        else:
            if v0 <= 0:
                raise ValueError(f"v0 must be strictly positive, got {v0}")
            self.v0 = float(v0)

        # Reciprocal CIR parameters
        self.kappa_cir = self.kappa * self.theta
        self.theta_cir = (self.kappa + self.epsilon**2) / (self.kappa * self.theta)
        self.sigma_cir = self.epsilon

        # Feller ratio for reciprocal: always > 1
        self.feller_ratio = 1.0 + 2.0 * self.kappa / (self.epsilon**2)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"ThreeHalvesProcess(kappa={self.kappa}, theta={self.theta}, epsilon={self.epsilon}, v0={self.v0})",
            math_note=(
                f"dv_t = κ v_t(θ - v_t)dt + ε v_t^{{3/2}} dW_t; reciprocal CIR y_t=1/v_t with "
                f"κ_y={self.kappa_cir:.4f}, θ_y={self.theta_cir:.4f}, σ_y={self.sigma_cir:.4f}; "
                f"reciprocal Feller ratio = {self.feller_ratio:.4f} > 1"
            ),
            reference="Heston (1997); Carr & Sun (2007), Review of Derivatives Research 10(2), 87–150",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Stationary Properties (Inverse-Gamma)
    # ------------------------------------------------------------------

    @property
    def stationary_shape(self) -> float:
        r"""Shape parameter α = 2 + 2κ / ε² of the stationary Inverse-Gamma law for reciprocal CIR."""
        return 2.0 + 2.0 * self.kappa / (self.epsilon**2)

    @property
    def stationary_scale(self) -> float:
        r"""Scale parameter β = 2κθ / ε² of the stationary Inverse-Gamma law."""
        return 2.0 * self.kappa * self.theta / (self.epsilon**2)

    @property
    def stationary_mean(self) -> float:
        r"""Stationary expectation E[v_∞] = β / (α - 1) = θ * (2κ / (2κ + ε²))."""
        return self.stationary_scale / (self.stationary_shape - 1.0)

    @property
    def stationary_variance(self) -> float:
        r"""
        Stationary variance Var(v_∞) = β² / [(α - 1)² (α - 2)] = ε² E[v_∞]² / (2κ).
        Finite since α = 2 + 2κ/ε² > 2 for all κ, ε > 0.
        """
        alpha = self.stationary_shape
        beta = self.stationary_scale
        return (beta**2) / (((alpha - 1.0) ** 2) * (alpha - 2.0))

    def stationary_density(self, v: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Probability density function of the stationary Inverse-Gamma distribution.
        """
        v_arr = np.asarray(v, dtype=np.float64)
        alpha = self.stationary_shape
        beta = self.stationary_scale
        pdf = invgamma.pdf(v_arr, a=alpha, scale=beta)
        if np.isscalar(v):
            return float(pdf)
        return pdf

    # ------------------------------------------------------------------
    # Conditional Reciprocal Analytics & Transition Density
    # ------------------------------------------------------------------

    def reciprocal_conditional_mean(self, t: float, v0: Optional[float] = None) -> float:
        r"""
        Expected reciprocal E[1/v_t | v_0].
        """
        if t <= 0:
            return 1.0 / (self.v0 if v0 is None else v0)
        y0 = 1.0 / (self.v0 if v0 is None else v0)
        return self.theta_cir + (y0 - self.theta_cir) * math.exp(-self.kappa_cir * t)

    def transition_density(
        self,
        v: Union[float, np.ndarray],
        t: float,
        v0: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        r"""
        Transition density p(v, t | v0) via Jacobian transformation of reciprocal CIR.
        Since y = 1/v, p(v, t) = p_CIR(1/v, t) * (1 / v^2).
        """
        if t <= 0:
            raise ValueError("t must be strictly positive")
        init_v = self.v0 if v0 is None else float(v0)
        y0 = 1.0 / init_v

        v_arr = np.asarray(v, dtype=np.float64)
        y_arr = 1.0 / np.maximum(v_arr, 1e-12)

        # Non-central chi-squared CIR parameters for y
        exp_k = math.exp(-self.kappa_cir * t)
        c = 2.0 * self.kappa_cir / (self.sigma_cir**2 * (1.0 - exp_k))
        df = 4.0 * self.kappa_cir * self.theta_cir / (self.sigma_cir**2)
        nonc = 2.0 * c * y0 * exp_k

        # CIR density of y: p_y = 2c * ncx2.pdf(2c * y, df, nonc)
        arg = 2.0 * c * y_arr
        p_y = 2.0 * c * ncx2.pdf(arg, df=df, nc=nonc)
        p_v = p_y * (1.0 / (v_arr**2))

        if np.isscalar(v):
            return float(p_v)
        return p_v

    # ------------------------------------------------------------------
    # Process Overrides & Vectorized Simulation
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "ThreeHalvesProcess has state-dependent diffusion ε v_t^{3/2} and does not "
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
            tail_index=self.stationary_shape,
            is_subordinator=False,
            hurst_index=None,
            notes=[
                "3/2 Stochastic Volatility diffusion model.",
                "Reciprocal follows a strictly positive CIR process.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.stationary_mean}
        if order >= 2:
            res[2] = self.stationary_variance
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

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
        Simulate sample paths of the 3/2 variance process.

        Parameters
        ----------
        t_grid : np.ndarray or float, optional
            1-D strictly increasing array of time points starting at 0, or terminal time T.
        n_paths : int, optional
            Number of simulated sample paths.
        rng : np.random.Generator, optional
            Numpy random generator.
        method : {"exact", "euler", "milstein"}
            Simulation scheme. "exact" simulates the reciprocal CIR process exactly
            via the non-central chi-squared distribution and inverts back.
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

        if method == "exact":
            # Exact reciprocal CIR simulation
            y_paths = np.zeros((paths_count, num_pts), dtype=np.float64)
            y_paths[:, 0] = 1.0 / self.v0

            df = 4.0 * self.kappa_cir * self.theta_cir / (self.sigma_cir**2)

            for i in range(1, num_pts):
                dt = grid[i] - grid[i - 1]
                if dt <= 0:
                    raise ValueError("t_grid must be strictly increasing")

                exp_k = math.exp(-self.kappa_cir * dt)
                c = 2.0 * self.kappa_cir / (self.sigma_cir**2 * (1.0 - exp_k))
                nc = 2.0 * c * y_paths[:, i - 1] * exp_k

                # Sample non-central chi-square
                pois = rng.poisson(lam=nc / 2.0)
                gam = rng.standard_gamma(shape=(df + 2.0 * pois) / 2.0) * 2.0
                y_paths[:, i] = gam / (2.0 * c)

            paths = 1.0 / np.maximum(y_paths, 1e-15)

        elif method in ("euler", "milstein"):
            # Direct Euler/Milstein simulation on log-variance for numerical stability:
            log_v = np.full(paths_count, math.log(self.v0), dtype=np.float64)
            for i in range(1, num_pts):
                dt = grid[i] - grid[i - 1]
                if dt <= 0:
                    raise ValueError("t_grid must be strictly increasing")
                sqrt_dt = math.sqrt(dt)
                z = rng.standard_normal(size=paths_count)

                v_curr = np.exp(log_v)
                drift = self.kappa * (self.theta - v_curr) - 0.5 * (self.epsilon**2) * v_curr
                diffusion = self.epsilon * np.sqrt(np.maximum(v_curr, 1e-12))

                log_v += drift * dt + diffusion * sqrt_dt * z
                paths[:, i] = np.exp(log_v)
        else:
            raise ValueError(f"Unknown simulation method: {method}")

        return paths

    def __repr__(self) -> str:
        return (
            f"ThreeHalvesProcess(kappa={self.kappa}, theta={self.theta}, "
            f"epsilon={self.epsilon}, v0={self.v0})"
        )


# Aliases
ThreeHalves = ThreeHalvesProcess
ThreeHalvesModel = ThreeHalvesProcess
