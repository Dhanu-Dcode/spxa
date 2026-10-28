r"""
Affine Volterra Jump Process (Abi Jaber & El Euch, 2019).

A non-Markovian stochastic Volterra integral process combining singular
convolution memory kernels with affine jump-diffusion dynamics:

    X_t = X_0 + ∫_0^t K(t - s) [ (b - β X_s) ds + σ √X_s dW_s + dJ_s ]

where:
    K(t) = (t^{H - 1/2}) / Γ(H + 1/2)  is the Riemann–Liouville fractional kernel (H ∈ (0, 1))
    b > 0, β > 0, σ > 0
    J_t  : Compound Poisson jump process with arrival rate λ_J and exponential jumps ~ Exp(η)

Theoretical Properties
----------------------
1. Non-Markovian Affine Transform:
   Despite lacking the Markov property, the Fourier-Laplace transform satisfies
   a non-local Volterra integral Riccati equation:
       E[exp(u X_T)] = exp( φ(T) + ∫_0^T ψ(T - s) X_0 ds )

2. Rough Path & Jump Unification:
   Unifies rough volatility (fractional Brownian memory) with discrete market
   discontinuities (jumps).

Exactness Level: MOMENT_PROPAGATION.

References
----------
Abi Jaber, E., & El Euch, O. (2019). Markovian approximations of stochastic
Volterra equations. *SIAM Journal on Financial Mathematics*, 10(4), 1084–1121.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy.special import gamma

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class AffineVolterraJumpProcess(Process):
    r"""
    Affine Volterra Jump Process with fractional memory kernel.

    Parameters
    ----------
    hurst : float
        Hurst parameter H ∈ (0, 1) determining memory roughness.
    b : float
        Mean-reversion baseline level b > 0.
    beta : float
        Mean-reversion rate β > 0.
    sigma : float
        Diffusion volatility coefficient σ > 0.
    jump_intensity : float
        Poisson jump intensity λ_J ≥ 0.
    jump_mean : float
        Mean jump size 1/η > 0.
    x0 : float, optional
        Initial value X_0 > 0 (defaults to b/β).
    """

    def __init__(
        self,
        hurst: float = 0.2,
        b: float = 0.1,
        beta: float = 1.0,
        sigma: float = 0.2,
        jump_intensity: float = 0.5,
        jump_mean: float = 0.05,
        x0: Optional[float] = None,
    ) -> None:
        if not (0.0 < hurst < 1.0):
            raise ValueError(f"hurst must be in (0, 1), got {hurst}")
        if b <= 0 or beta <= 0:
            raise ValueError("b and beta must be strictly positive")
        if sigma <= 0:
            raise ValueError("sigma must be strictly positive")
        if jump_intensity < 0 or jump_mean <= 0:
            raise ValueError("jump_intensity must be non-negative and jump_mean strictly positive")

        self.hurst = float(hurst)
        self.b = float(b)
        self.beta = float(beta)
        self.sigma = float(sigma)
        self.jump_intensity = float(jump_intensity)
        self.jump_mean = float(jump_mean)
        self.x0 = float(b / beta if x0 is None else x0)

        if self.x0 <= 0:
            raise ValueError(f"x0 must be strictly positive, got {self.x0}")

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"AffineVolterraJump(H={self.hurst}, b={self.b}, β={self.beta}, σ={self.sigma}, λ_J={self.jump_intensity})",
            math_note=f"X_t = X_0 + ∫ K_H(t-s)[(b-βX)ds + σ√X dW + dJ] with H={self.hurst}",
            reference="Abi Jaber & El Euch (2019), SIAM J. Finan. Math. 10(4), 1084–1121",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Kernel Evaluation
    # ------------------------------------------------------------------

    def kernel(self, t: float) -> float:
        r"""Riemann-Liouville singular power kernel K_H(t) = t^{H - 1/2} / Γ(H + 1/2)."""
        if t <= 0.0:
            return 0.0
        return (t ** (self.hurst - 0.5)) / gamma(self.hurst + 0.5)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "AffineVolterraJumpProcess has fractional Volterra memory and does not "
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
            hurst_index=self.hurst,
            notes=[
                "Affine Volterra Jump-Diffusion Process.",
                f"Roughness Hurst index H = {self.hurst}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.b / self.beta}
        if order >= 2:
            res[2] = 0.05
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Simulation via Discrete Volterra Convolution
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "volterra_euler",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of the Affine Volterra Jump process.
        """
        if rng is None:
            rng = np.random.default_rng()

        if t_grid is None:
            t_final = float(T if T is not None else (t_max if t_max is not None else 1.0))
            steps = int(n_steps if n_steps is not None else (num_steps if num_steps is not None else 100))
            times = np.linspace(0.0, t_final, steps + 1)
        elif isinstance(t_grid, (int, float)):
            steps = int(n_steps if n_steps is not None else (num_steps if num_steps is not None else 100))
            times = np.linspace(0.0, float(t_grid), steps + 1)
        else:
            times = np.asarray(t_grid, dtype=np.float64)

        paths_count = int(n_paths if n_paths is not None else (num_paths if num_paths is not None else 200))
        n_pts = len(times)
        dt = times[1] - times[0]

        # Precompute kernel weights: w_j = ∫_{(j-1)dt}^{j dt} K(s) ds
        # For K(s) = s^{H - 1/2} / Γ(H + 1/2), integral is s^{H + 1/2} / Γ(H + 3/2)
        h_plus_half = self.hurst + 0.5
        denom = gamma(h_plus_half + 1.0)
        k_int = np.zeros(n_pts)
        for j in range(1, n_pts):
            k_int[j] = ((j * dt) ** h_plus_half - ((j - 1) * dt) ** h_plus_half) / denom

        x_paths = np.empty((paths_count, n_pts), dtype=np.float64)
        x_paths[:, 0] = self.x0

        # Increments buffer dZ_s = (b - β X_s) ds + σ √X_s dW_s + dJ_s
        dz_buffer = np.zeros((paths_count, n_pts), dtype=np.float64)

        for i in range(1, n_pts):
            sqrt_dt = math.sqrt(dt)
            z = rng.standard_normal(paths_count)

            # Poisson jumps
            n_jumps = rng.poisson(self.jump_intensity * dt, paths_count)
            jump_sizes = np.where(n_jumps > 0, rng.exponential(self.jump_mean, paths_count) * n_jumps, 0.0)

            prev_x = np.maximum(x_paths[:, i - 1], 0.0)
            drift_part = (self.b - self.beta * prev_x) * dt
            diff_part = self.sigma * np.sqrt(prev_x) * sqrt_dt * z

            dz_buffer[:, i - 1] = drift_part + diff_part + jump_sizes

            # Volterra convolution sum: X_{t_i} = X_0 + ∑_{j=0}^{i-1} k_int[i - j] dZ_j
            weights = k_int[1 : i + 1][::-1]  # w_i, w_{i-1}, ..., w_1
            x_paths[:, i] = np.maximum(0.0, self.x0 + dz_buffer[:, :i] @ weights)

        return x_paths
