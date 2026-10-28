r"""
Run-and-Tumble Particle Process (Active Brownian Motion) (Cates & Tailleur, 2015).

Models the non-equilibrium motility of active biological swimmers (such as E. coli)
and self-propelled synthetic microswimmers:

    dx_t = v_0 cos(θ_t) dt + √{2 D_t} dW_t^x
    dy_t = v_0 sin(θ_t) dt + √{2 D_t} dW_t^y

where:
    v_0 > 0 : Constant self-propulsion swim speed
    θ_t     : Orientation angle undergoing random Poisson tumbling at rate λ > 0
    D_t ≥ 0 : Translational passive thermal diffusion

Theoretical Properties
----------------------
1. Ballistic-to-Diffusive Crossover:
   - Short times (t ≪ 1/λ): Pure ballistic persistence:
       ⟨r_t²⟩ ≈ v_0² t²
   - Long times (t ≫ 1/λ): Random walk diffusion with effective active diffusivity:
       ⟨r_t²⟩ ≈ 4 D_eff t,   where D_eff = D_t + (v_0² / (2 λ))

Exactness Level: MOMENT_PROPAGATION.

References
----------
Cates, M. E., & Tailleur, J. (2015). Motility-induced phase separation.
*Annual Review of Condensed Matter Physics*, 6(1), 219–244.
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


class RunAndTumbleProcess(Process):
    r"""
    Run-and-Tumble Particle Process in 2D active matter.

    Parameters
    ----------
    speed : float
        Ballistic swim velocity v_0 > 0.
    tumble_rate : float
        Poisson tumbling rate λ > 0.
    diffusion : float
        Passive thermal diffusion D_t ≥ 0.
    x0 : float, optional
        Initial x position (defaults to 0.0).
    y0 : float, optional
        Initial y position (defaults to 0.0).
    """

    def __init__(
        self,
        speed: float = 10.0,
        tumble_rate: float = 1.0,
        diffusion: float = 0.1,
        x0: float = 0.0,
        y0: float = 0.0,
    ) -> None:
        if speed <= 0:
            raise ValueError(f"speed must be strictly positive, got {speed}")
        if tumble_rate <= 0:
            raise ValueError(f"tumble_rate must be strictly positive, got {tumble_rate}")
        if diffusion < 0:
            raise ValueError(f"diffusion must be non-negative, got {diffusion}")

        self.speed = float(speed)
        self.tumble_rate = float(tumble_rate)
        self.diffusion = float(diffusion)
        self.x0 = float(x0)
        self.y0 = float(y0)

        # Effective active diffusion coefficient D_eff = D_t + v0^2 / (2 lambda)
        self.d_eff = self.diffusion + (self.speed**2) / (2.0 * self.tumble_rate)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"RunAndTumble(v0={self.speed}, λ={self.tumble_rate}, D_eff={self.d_eff:.2f})",
            math_note=f"Active swimmer with speed {self.speed} and tumble rate {self.tumble_rate}; D_eff={self.d_eff:.3f}",
            reference="Cates & Tailleur (2015), Annu. Rev. Condens. Matter Phys. 6(1), 219–244",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Analytical Mean Square Displacement
    # ------------------------------------------------------------------

    def mean_square_displacement(self, t: float) -> float:
        r"""
        Exact crossover MSD formula:
        ⟨r_t²⟩ = 4 D_t t + (2 v_0² / λ²) [ λ t - 1 + exp(-λ t) ].
        """
        if t <= 0.0:
            return 0.0
        lam = self.tumble_rate
        v0 = self.speed
        thermal_term = 4.0 * self.diffusion * t
        active_term = (2.0 * (v0**2) / (lam**2)) * (lam * t - 1.0 + math.exp(-lam * t))
        return thermal_term + active_term

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "RunAndTumbleProcess features persistent non-Markovian velocity orientation and does not "
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
            hurst_index=1.0,  # Ballistic at short times
            notes=[
                "Active matter Run-and-Tumble swimmer process.",
                f"Effective diffusion D_eff = {self.d_eff:.4f}.",
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
    # Simulation (2D Active Paths)
    # ------------------------------------------------------------------

    def simulate_2d(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: int = 200,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Simulate joint 2D sample paths (X_t, Y_t).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (x_paths, y_paths) of shape (n_paths, len(t_grid)).
        """
        if rng is None:
            rng = np.random.default_rng()

        times = np.linspace(0.0, 1.0 if t_grid is None else float(t_grid), 252) if isinstance(t_grid, (int, float, type(None))) else np.asarray(t_grid)
        n_pts = len(times)

        x_paths = np.empty((n_paths, n_pts), dtype=np.float64)
        y_paths = np.empty((n_paths, n_pts), dtype=np.float64)
        x_paths[:, 0] = self.x0
        y_paths[:, 0] = self.y0

        theta_curr = rng.uniform(0.0, 2.0 * math.pi, n_paths)
        diff_scale = math.sqrt(2.0 * self.diffusion) if self.diffusion > 0 else 0.0

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)

            # Poisson tumble events
            tumbles = rng.uniform(0.0, 1.0, n_paths) < (1.0 - math.exp(-self.tumble_rate * dt))
            new_thetas = rng.uniform(0.0, 2.0 * math.pi, n_paths)
            theta_curr = np.where(tumbles, new_thetas, theta_curr)

            # Ballistic update
            vx = self.speed * np.cos(theta_curr)
            vy = self.speed * np.sin(theta_curr)

            # Thermal noise
            zx = rng.standard_normal(n_paths)
            zy = rng.standard_normal(n_paths)

            x_paths[:, i] = x_paths[:, i - 1] + vx * dt + diff_scale * sqrt_dt * zx
            y_paths[:, i] = y_paths[:, i - 1] + vy * dt + diff_scale * sqrt_dt * zy

        return x_paths, y_paths

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "active_swimmer",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""Simulate radial displacement R_t = √((X_t - x_0)² + (Y_t - y_0)²)."""
        paths_count = int(n_paths if n_paths is not None else (num_paths if num_paths is not None else 200))
        grid = t_grid if t_grid is not None else float(T if T is not None else (t_max if t_max is not None else 1.0))
        xs, ys = self.simulate_2d(t_grid=grid, n_paths=paths_count, rng=rng)
        return np.sqrt((xs - self.x0) ** 2 + (ys - self.y0) ** 2)


# Aliases
ActiveBrownianMotion = RunAndTumbleProcess
