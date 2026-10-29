r"""
Bilinear Jump-Diffusion Process (Chacko & Viceira, 2003).

A continuous-time stochastic volatility and jump model where instantaneous
variance enters bilinearly in both the asset drift and the Poisson jump arrival intensity:

    dS_t / S_t = (μ_0 + μ_1 v_t) dt + √v_t dW_t^S + (e^{Y_t} - 1) dN_t
    dv_t = κ (θ - v_t) dt + σ_v √v_t dW_t^v + J_v dN_t

where:
    d⟨W^S, W^v⟩_t = ρ dt
    λ(v_t) = λ_0 + λ_1 v_t : State-dependent jump intensity (market panics cluster with high volatility)
    Y_t ~ Normal(μ_J, σ_J²)  : Asset jump size
    J_v ~ Exponential(η_v)  : Variance jump size (volatility spikes upon crash)

Theoretical Properties
----------------------
1. Volatility-Dependent Jump Clustering:
   Because λ(v) increases linearly with v_t (λ_1 > 0), turbulent volatility regimes
   endogenously trigger severe asset price crashes.

2. Affine Solvability:
   Admits exact affine characteristic functions and closed-form optimal consumption
   and portfolio allocation rules.

Exactness Level: MOMENT_PROPAGATION.

References
----------
Chacko, G., & Viceira, L. M. (2003). Spectral GMM estimation of continuous-time
processes. *Journal of Econometrics*, 116(1–2), 259–292.
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


class BilinearJumpDiffusion(Process):
    r"""
    Bilinear Jump-Diffusion process with volatility-dependent jump intensity.

    Parameters
    ----------
    mu0 : float
        Baseline asset drift μ_0 (defaults to 0.03).
    mu1 : float
        Risk-premium slope μ_1 (defaults to 1.5).
    kappa : float
        Variance mean-reversion rate κ > 0 (defaults to 2.0).
    theta : float
        Long-term variance level θ > 0 (defaults to 0.04).
    sigma_v : float
        Vol-of-vol parameter σ_v > 0 (defaults to 0.3).
    rho : float
        Correlation ρ ∈ [-1, 1] between asset and variance noise.
    lam0 : float
        Baseline jump arrival rate λ_0 ≥ 0 (defaults to 0.1).
    lam1 : float
        Volatility-dependent jump sensitivity λ_1 ≥ 0 (defaults to 5.0).
    jump_mu : float
        Mean asset jump size (defaults to -0.05).
    jump_sigma : float
        Asset jump volatility (defaults to 0.08).
    var_jump_mean : float
        Mean variance spike J_v > 0 (defaults to 0.02).
    s0 : float, optional
        Initial spot price S_0 > 0 (defaults to 100.0).
    v0 : float, optional
        Initial variance v_0 > 0 (defaults to θ).
    """

    def __init__(
        self,
        mu0: float = 0.03,
        mu1: float = 1.5,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma_v: float = 0.3,
        rho: float = -0.7,
        lam0: float = 0.1,
        lam1: float = 5.0,
        jump_mu: float = -0.05,
        jump_sigma: float = 0.08,
        var_jump_mean: float = 0.02,
        s0: float = 100.0,
        v0: Optional[float] = None,
    ) -> None:
        if kappa <= 0 or theta <= 0 or sigma_v <= 0:
            raise ValueError("kappa, theta, and sigma_v must be strictly positive")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1], got {rho}")
        if lam0 < 0 or lam1 < 0:
            raise ValueError("Jump intensity coefficients lam0 and lam1 must be non-negative")
        if var_jump_mean <= 0 or jump_sigma <= 0:
            raise ValueError("Jump dispersion parameters must be strictly positive")
        if s0 <= 0:
            raise ValueError(f"s0 must be strictly positive, got {s0}")

        self.mu0 = float(mu0)
        self.mu1 = float(mu1)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma_v = float(sigma_v)
        self.rho = float(rho)
        self.lam0 = float(lam0)
        self.lam1 = float(lam1)
        self.jump_mu = float(jump_mu)
        self.jump_sigma = float(jump_sigma)
        self.var_jump_mean = float(var_jump_mean)
        self.s0 = float(s0)
        self.v0 = float(theta if v0 is None else v0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"BilinearJumpDiffusion(κ={self.kappa}, θ={self.theta}, λ0={self.lam0}, λ1={self.lam1})",
            math_note=f"dS/S = (μ_0 + μ_1 v)dt + √v dW^S + (e^Y - 1)dN with λ(v) = {self.lam0} + {self.lam1} v",
            reference="Chacko & Viceira (2003), Journal of Econometrics 116(1–2), 259–292",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "BilinearJumpDiffusion features state-dependent jump arrivals and stochastic volatility, "
            "lacking an independent Lévy–Khintchine triplet."
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
                "Bilinear jump-diffusion with volatility-dependent jump arrivals.",
                f"State-dependent jump intensity λ(v) = {self.lam0} + {self.lam1} v.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.s0 * math.exp(self.mu0 + self.mu1 * self.theta)}
        if order >= 2:
            res[2] = self.theta
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "full_truncation_jumps",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of asset price S_t.
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

        s_paths = np.empty((paths_count, n_pts), dtype=np.float64)
        s_paths[:, 0] = self.s0

        s_curr = np.full(paths_count, self.s0, dtype=np.float64)
        v_curr = np.full(paths_count, self.v0, dtype=np.float64)

        rho_comp = math.sqrt(max(0.0, 1.0 - self.rho**2))

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)

            z1 = rng.standard_normal(paths_count)
            z2 = rng.standard_normal(paths_count)
            zv = z1
            zs = self.rho * z1 + rho_comp * z2

            v_pos = np.maximum(v_curr, 0.0)
            sqrt_v = np.sqrt(v_pos)

            # Volatility-dependent jump probability: λ(v) dt
            lam_t = self.lam0 + self.lam1 * v_pos
            n_jumps = rng.poisson(lam_t * dt)

            # Jump sizes in asset price and variance
            y_jump = np.where(
                n_jumps > 0, rng.normal(self.jump_mu, self.jump_sigma, paths_count) * n_jumps, 0.0
            )
            var_jump = np.where(
                n_jumps > 0, rng.exponential(self.var_jump_mean, paths_count) * n_jumps, 0.0
            )

            # Spot log-price update
            d_log_s = (
                (self.mu0 + self.mu1 * v_pos - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * zs + y_jump
            )
            s_curr = s_curr * np.exp(d_log_s)

            # Variance update with variance jump spike
            v_curr = (
                v_curr
                + self.kappa * (self.theta - v_pos) * dt
                + self.sigma_v * sqrt_v * sqrt_dt * zv
                + var_jump
            )

            s_paths[:, i] = s_curr

        return s_paths
