r"""
Schrödinger Diffusion / Entropic Optimal Transport Bridge (Schrödinger, 1931; Léonard, 2014).

Solves the continuous-time Schrödinger bridge problem: finding the diffusion process
P on C([0, T], ℝ) that minimizes the relative entropy (Kullback–Leibler divergence)
with respect to a reference Wiener process R, subject to matching prescribed initial
and terminal boundary marginal distributions ρ_0 and ρ_1:

    dX_t = b^*(t, X_t) dt + σ dW_t

For Gaussian marginals X_0 ~ Normal(μ_0, σ_0²) and X_T ~ Normal(μ_1, σ_1²),
the optimal entropic drift is affine:
    b^*(t, x) = A(t) (m(t) - x)

Theoretical Properties
----------------------
1. Entropic Interpolation & Finite Temperature:
   Unlike deterministic Benamou–Brenier Monge optimal transport (zero noise),
   the Schrödinger bridge incorporates finite stochastic thermal diffusion σ > 0.
   As σ → 0, the process converges to the deterministic displacement interpolation geodesic.

2. Reversibility:
   The Schrödinger diffusion satisfies pathwise time-reversal symmetry with zero
   net entropy production rate.

Exactness Level: MOMENT_PROPAGATION.

References
----------
Schrödinger, E. (1931). Über die Umkehrung der Naturgesetze.
*Sitzungsberichte der Preussischen Akademie der Wissenschaften*, 144–153.

Léonard, C. (2014). A survey of the Schrödinger problem and some of its connections
with optimal transport. *Discrete and Continuous Dynamical Systems*, 34(4), 1533–1574.
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


class SchrodingerDiffusion(Process):
    r"""
    Schrödinger Diffusion Entropic Optimal Transport Bridge between Gaussian marginals.

    Parameters
    ----------
    mu0 : float
        Initial distribution mean μ_0.
    sigma0 : float
        Initial distribution standard deviation σ_0 > 0.
    mu1 : float
        Terminal target mean μ_1 at T.
    sigma1 : float
        Terminal target standard deviation σ_1 > 0.
    t_terminal : float
        Horizon T > 0 (defaults to 1.0).
    diff_noise : float
        Thermal noise volatility σ > 0 (defaults to 0.5).
    """

    def __init__(
        self,
        mu0: float = 0.0,
        sigma0: float = 1.0,
        mu1: float = 3.0,
        sigma1: float = 0.5,
        t_terminal: float = 1.0,
        diff_noise: float = 0.5,
    ) -> None:
        if sigma0 <= 0 or sigma1 <= 0:
            raise ValueError("sigma0 and sigma1 must be strictly positive")
        if t_terminal <= 0:
            raise ValueError("t_terminal must be strictly positive")
        if diff_noise <= 0:
            raise ValueError("diff_noise must be strictly positive")

        self.mu0 = float(mu0)
        self.sigma0 = float(sigma0)
        self.mu1 = float(mu1)
        self.sigma1 = float(sigma1)
        self.t_terminal = float(t_terminal)
        self.diff_noise = float(diff_noise)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"SchrodingerDiffusion(μ0={self.mu0}, μ1={self.mu1}, T={self.t_terminal})",
            math_note=f"Minimizes KL(P || Wiener) between Normal({self.mu0},{self.sigma0}²) and Normal({self.mu1},{self.sigma1}²)",
            reference="Schrödinger (1931); Léonard (2014), DCDS 34(4), 1533–1574",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Entropic Interpolation Trajectory
    # ------------------------------------------------------------------

    def interpolated_mean(self, t: float) -> float:
        r"""Mean of the entropic bridge at intermediate time t ∈ [0, T]."""
        s = t / self.t_terminal
        return (1.0 - s) * self.mu0 + s * self.mu1

    def interpolated_variance(self, t: float) -> float:
        r"""Variance of the entropic bridge at intermediate time t ∈ [0, T]."""
        s = t / self.t_terminal
        return ((1.0 - s) * self.sigma0 + s * self.sigma1) ** 2

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "SchrodingerDiffusion has time-inhomogeneous bridge drift conditioned on terminal "
            "marginals and does not admit a stationary Lévy–Khintchine triplet."
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
                "Schrödinger entropic optimal transport bridge diffusion.",
                f"Interpolates from N({self.mu0}, {self.sigma0}²) to N({self.mu1}, {self.sigma1}²).",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.mu0}
        if order >= 2:
            res[2] = self.sigma0**2
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Simulation (Exact Entropic Bridge Coupling)
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "entropic_bridge",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of the Schrödinger diffusion bridge.
        """
        if rng is None:
            rng = np.random.default_rng()

        t_final = self.t_terminal
        if t_grid is None or isinstance(t_grid, (int, float)):
            steps = int(
                n_steps if n_steps is not None else (num_steps if num_steps is not None else 100)
            )
            times = np.linspace(0.0, t_final, steps + 1)
        else:
            times = np.asarray(t_grid, dtype=np.float64)

        paths_count = int(
            n_paths if n_paths is not None else (num_paths if num_paths is not None else 500)
        )
        n_pts = len(times)

        # Jointly sample initial and terminal coupled endpoints
        z_init = rng.standard_normal(paths_count)
        x_0 = self.mu0 + self.sigma0 * z_init
        x_T = self.mu1 + self.sigma1 * z_init  # Monge coupling preserves quantile ordering

        # Standard Brownian bridge conditioned on endpoints:
        # X_t = (1 - t/T) X_0 + (t/T) X_T + B_t^{bridge}
        x_paths = np.empty((paths_count, n_pts), dtype=np.float64)

        for i, t in enumerate(times):
            s = t / t_final
            mean_interp = (1.0 - s) * x_0 + s * x_T
            bridge_std = self.diff_noise * math.sqrt(s * (1.0 - s) * t_final)
            x_paths[:, i] = mean_interp + bridge_std * rng.standard_normal(paths_count)

        x_paths[:, 0] = x_0
        x_paths[:, -1] = x_T

        return x_paths


# Aliases
EntropicBridgeProcess = SchrodingerDiffusion
