r"""
Continuous Time Random Walk (CTRW) with Anomalous Subdiffusion (Metzler & Klafter, 2000).

A non-Markovian renewal jump process where particle waiting times between successive
spatial displacements follow heavy-tailed power-law distributions:
    ψ(t) ~ t^{-(1 + α)}   as t → ∞,   0 < α < 1

The trajectory is given by:
    X_t = ∑_{k=1}^{N_t} ξ_k

where:
    ξ_k ~ Normal(0, σ²) are independent spatial jumps
    N_t is an inverse stable subordinator counting renewal arrivals

Theoretical Properties
----------------------
1. Fractional Fokker–Planck Diffusion:
   The ensemble density obeys the fractional diffusion equation:
       ∂_t^α P(x, t) = D_α ∂_x² P(x, t)
   where ∂_t^α is the Caputo fractional derivative of order α.

2. Subdiffusive Mean Square Displacement (MSD):
   The particle exhibits genuine subdiffusion with sub-linear power law:
       ⟨X_t²⟩ = (2 D_α / Γ(1 + α)) t^α

Exactness Level: MOMENT_PROPAGATION.

References
----------
Metzler, R., & Klafter, J. (2000). The random walk's guide to anomalous
diffusion: a fractional dynamics approach. *Physics Reports*, 339(1), 1–77.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.special import gamma

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class ContinuousTimeRandomWalk(Process):
    r"""
    Continuous Time Random Walk with heavy-tailed subdiffusive waiting times.

    Parameters
    ----------
    alpha : float
        Subdiffusive exponent α ∈ (0, 1) (e.g. 0.7).
    jump_sigma : float
        Standard deviation of Gaussian spatial steps σ > 0.
    scale : float
        Time scale parameter τ_0 > 0.
    x0 : float, optional
        Initial particle position X_0 (defaults to 0.0).
    """

    def __init__(
        self,
        alpha: float = 0.7,
        jump_sigma: float = 1.0,
        scale: float = 1.0,
        x0: float = 0.0,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1) for subdiffusion, got {alpha}")
        if jump_sigma <= 0:
            raise ValueError(f"jump_sigma must be strictly positive, got {jump_sigma}")
        if scale <= 0:
            raise ValueError(f"scale must be strictly positive, got {scale}")

        self.alpha = float(alpha)
        self.jump_sigma = float(jump_sigma)
        self.scale = float(scale)
        self.x0 = float(x0)

        # Anomalous diffusion coefficient D_alpha = sigma^2 / (2 tau_0^alpha)
        self.d_alpha = (self.jump_sigma**2) / (2.0 * (self.scale**self.alpha))

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"CTRW(α={self.alpha}, σ={self.jump_sigma})",
            math_note=f"Subdiffusion ⟨X²⟩ ~ t^α with α={self.alpha}; fractional Caputo diffusion",
            reference="Metzler & Klafter (2000), Physics Reports 339(1), 1–77",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Anomalous Mean Square Displacement
    # ------------------------------------------------------------------

    def mean_square_displacement(self, t: float) -> float:
        r"""
        Exact subdiffusive mean square displacement:
        ⟨X_t²⟩ = (2 D_α / Γ(1 + α)) t^α.
        """
        if t <= 0.0:
            return 0.0
        return (2.0 * self.d_alpha / gamma(1.0 + self.alpha)) * (t**self.alpha)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "ContinuousTimeRandomWalk has non-exponential heavy-tailed waiting times and does not "
            "admit a stationary Lévy–Khintchine triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=True,
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=self.alpha / 2.0,
            tail_index=None,
            is_subordinator=False,
            hurst_index=self.alpha / 2.0,
            notes=[
                f"Continuous Time Random Walk with subdiffusive index α = {self.alpha}.",
                f"Mean-square displacement scales as t^{self.alpha}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.x0}
        if order >= 2:
            res[2] = self.mean_square_displacement(1.0)
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Vectorized Simulation via Mittag-Leffler / Pareto Waiting Times
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "renewal_sampling",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of CTRW with heavy-tailed waiting times.
        """
        if rng is None:
            rng = np.random.default_rng()

        if t_grid is None:
            t_final = float(T if T is not None else (t_max if t_max is not None else 1.0))
            steps = int(
                n_steps if n_steps is not None else (num_steps if num_steps is not None else 100)
            )
            times = np.linspace(0.0, t_final, steps + 1)
        elif isinstance(t_grid, (int, float)):
            steps = int(
                n_steps if n_steps is not None else (num_steps if num_steps is not None else 100)
            )
            times = np.linspace(0.0, float(t_grid), steps + 1)
        else:
            times = np.asarray(t_grid, dtype=np.float64)

        paths_count = int(
            n_paths if n_paths is not None else (num_paths if num_paths is not None else 200)
        )
        n_pts = len(times)
        x_paths = np.empty((paths_count, n_pts), dtype=np.float64)

        for p in range(paths_count):
            t_curr = 0.0
            pos_curr = self.x0
            history_t = [0.0]
            history_x = [self.x0]

            # Generate renewal steps until t_final
            while t_curr < times[-1]:
                # Sample heavy-tailed waiting time: U^(-1/alpha) - 1
                u = rng.uniform(1e-5, 1.0)
                dt = self.scale * (u ** (-1.0 / self.alpha) - 1.0)
                t_curr += dt

                # Sample Gaussian spatial jump
                dx = rng.normal(0.0, self.jump_sigma)
                pos_curr += dx

                history_t.append(t_curr)
                history_x.append(pos_curr)

            # Interpolate piecewise-constant trajectory onto grid
            hist_t_arr = np.array(history_t)
            hist_x_arr = np.array(history_x)
            indices = np.searchsorted(hist_t_arr, times, side="right") - 1
            indices = np.clip(indices, 0, len(hist_x_arr) - 1)
            x_paths[p, :] = hist_x_arr[indices]

        return x_paths


# Aliases
CTRWProcess = ContinuousTimeRandomWalk
