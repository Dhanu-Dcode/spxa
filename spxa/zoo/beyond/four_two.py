r"""
4/2 Stochastic Volatility diffusion model (Grasselli, 2017).

The 4/2 model unifies the Heston (1/2) model and the 3/2 model by representing
instantaneous volatility as a linear combination of a CIR square-root process
and its reciprocal:

    dS_t = μ S_t dt + (a √v_t + b / √v_t) S_t dW_t^S
    dv_t = κ (θ - v_t) dt + σ_v √v_t dW_t^v

where:
    d⟨W^S, W^v⟩_t = ρ dt
    a ≥ 0, b ≥ 0 (with a + b > 0)
    κ > 0, θ > 0, σ_v > 0
    v_0 > 0, S_0 > 0

Theoretical Properties
----------------------
1. Limiting Regimes:
   - When b = 0, a = 1: Recovers the classic Heston (1993) model.
   - When a = 0, b = 1: Recovers the 3/2 stochastic volatility model.
   - When a > 0 and b > 0: Simultaneously resolves the steep short-maturity smile
     and the upward-sloping long-maturity implied volatility smile.

2. Instantaneous Variance:
   The total instantaneous variance is:
       V_t = (a √v_t + b / √v_t)² = a² v_t + 2 a b + b² / v_t

3. Expectation and Moments:
   Because v_t follows a CIR process, v_∞ ~ Gamma(2κθ / σ_v², 2κ / σ_v²),
   and 1/v_∞ follows an Inverse-Gamma distribution. Consequently:
       E[V_∞] = a² θ + 2 a b + b² / (θ (1 - σ_v² / (2κθ)))
   when 2κθ > σ_v² (the Feller condition).

Exactness Level: MOMENT_PROPAGATION.

References
----------
Grasselli, M. (2017). The 4/2 stochastic volatility model.
*Mathematical Finance*, 27(4), 987–1027.
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


class FourTwoProcess(Process):
    r"""
    4/2 Stochastic Volatility diffusion process (Grasselli, 2017).

    Parameters
    ----------
    kappa : float
        Mean-reversion speed κ > 0 for underlying variance v_t.
    theta : float
        Long-term mean θ > 0 for v_t.
    sigma_v : float
        Volatility-of-variance σ_v > 0.
    rho : float
        Correlation ρ ∈ [-1, 1] between asset and variance drivers.
    a : float
        Weight on the Heston square-root component a ≥ 0.
    b : float
        Weight on the 3/2 reciprocal component b ≥ 0.
    v0 : float, optional
        Initial variance level v_0 > 0 (defaults to theta).
    s0 : float, optional
        Initial spot price S_0 > 0 (defaults to 100.0).
    mu : float, optional
        Asset drift μ (defaults to 0.0).
    """

    def __init__(
        self,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma_v: float = 0.3,
        rho: float = -0.7,
        a: float = 0.8,
        b: float = 0.02,
        v0: Optional[float] = None,
        s0: float = 100.0,
        mu: float = 0.0,
    ) -> None:
        if kappa <= 0:
            raise ValueError(f"kappa must be strictly positive, got {kappa}")
        if theta <= 0:
            raise ValueError(f"theta must be strictly positive, got {theta}")
        if sigma_v <= 0:
            raise ValueError(f"sigma_v must be strictly positive, got {sigma_v}")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1], got {rho}")
        if a < 0 or b < 0 or (a == 0.0 and b == 0.0):
            raise ValueError(f"a and b must be non-negative with a + b > 0, got a={a}, b={b}")
        if s0 <= 0:
            raise ValueError(f"s0 must be strictly positive, got {s0}")

        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma_v = float(sigma_v)
        self.rho = float(rho)
        self.a = float(a)
        self.b = float(b)
        self.s0 = float(s0)
        self.mu = float(mu)
        self.v0 = float(theta if v0 is None else v0)

        if self.v0 <= 0:
            raise ValueError(f"v0 must be strictly positive, got {self.v0}")

        self.feller_ratio = (2.0 * self.kappa * self.theta) / (self.sigma_v**2)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"FourTwoProcess(κ={self.kappa}, θ={self.theta}, σ_v={self.sigma_v}, a={self.a}, b={self.b})",
            math_note=(
                f"4/2 model with σ_t = (a√v_t + b/√v_t); Feller ratio={self.feller_ratio:.3f}; "
                f"a={self.a}, b={self.b}, ρ={self.rho}"
            ),
            reference="Grasselli (2017), Mathematical Finance 27(4), 987–1027",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Properties & Analytical Moments
    # ------------------------------------------------------------------

    @property
    def is_feller_satisfied(self) -> bool:
        r"""True if 2κθ > σ_v²."""
        return self.feller_ratio > 1.0

    def total_instantaneous_variance(self, v: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Compute total instantaneous variance V_t = (a √v_t + b / √v_t)²."""
        v_clipped = np.maximum(v, 1e-12)
        sqrt_v = np.sqrt(v_clipped)
        vol = self.a * sqrt_v + self.b / sqrt_v
        return vol**2

    @property
    def stationary_mean_variance(self) -> float:
        r"""
        Asymptotic stationary mean of total variance E[V_∞] = a² θ + 2ab + b² E[1/v_∞].
        Finite if Feller ratio > 1.
        """
        if self.feller_ratio <= 1.0:
            return float("inf")
        # E[1/v_∞] for CIR = (2κ) / (2κθ - σ_v²)
        e_inv_v = (2.0 * self.kappa) / (2.0 * self.kappa * self.theta - self.sigma_v**2)
        return (self.a**2) * self.theta + 2.0 * self.a * self.b + (self.b**2) * e_inv_v

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "FourTwoProcess has stochastic variance (a√v_t + b/√v_t)² and does not admit a "
            "stationary Lévy–Khintchine triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=(self.mu == 0.0),
            has_finite_variance=self.is_feller_satisfied,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=None,
            notes=[
                "4/2 Stochastic Volatility diffusion model (Grasselli 2017).",
                f"Feller ratio 2κθ/σ_v² = {self.feller_ratio:.4f}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.s0 * math.exp(self.mu)}
        if order >= 2:
            res[2] = (
                self.stationary_mean_variance if math.isfinite(self.stationary_mean_variance) else 1.0
            )
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Simulation (Full Truncation Predictor-Corrector)
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
        Simulate asset paths S_t under the 4/2 model using full-truncation Euler scheme.

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, len(t_grid)) of simulated asset prices.
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

        paths_count = int(n_paths if n_paths is not None else (num_paths if num_paths is not None else 1000))
        n_pts = len(times)

        s_paths = np.empty((paths_count, n_pts), dtype=np.float64)
        s_paths[:, 0] = self.s0

        v_curr = np.full(paths_count, self.v0, dtype=np.float64)
        s_curr = np.full(paths_count, self.s0, dtype=np.float64)

        rho_comp = math.sqrt(max(0.0, 1.0 - self.rho**2))

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            if dt <= 0:
                s_paths[:, i] = s_curr
                continue

            sqrt_dt = math.sqrt(dt)
            z1 = rng.standard_normal(paths_count)
            z2 = rng.standard_normal(paths_count)

            zv = z1
            zs = self.rho * z1 + rho_comp * z2

            v_pos = np.maximum(v_curr, 0.0)
            sqrt_v = np.sqrt(v_pos)

            # Volatility of asset: a*sqrt(v) + b/sqrt(v) with floor for numerical stability
            v_safe = np.maximum(v_pos, 0.01 * self.theta)
            vol_s = self.a * np.sqrt(v_safe) + self.b / np.sqrt(v_safe)

            # SDE update for asset log-price with exponent clipping against floating-point underflow
            d_log_s = np.clip((self.mu - 0.5 * (vol_s**2)) * dt + vol_s * sqrt_dt * zs, -50.0, 50.0)
            s_curr = s_curr * np.exp(d_log_s)

            # SDE update for variance CIR (full truncation)
            v_curr = v_curr + self.kappa * (self.theta - v_pos) * dt + self.sigma_v * sqrt_v * sqrt_dt * zv

            s_paths[:, i] = s_curr

        return s_paths


# Aliases
FourTwoModel = FourTwoProcess
Grasselli42 = FourTwoProcess
