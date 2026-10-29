r"""
Geometric Hawkes Process (Stabile & Torrisi, 2010).

A jump-diffusion asset price model where jumps cluster endogenously according
to a self-exciting Hawkes process:

    S_t = S_0 exp( (μ - ½ σ²) t + σ W_t + ∑_{k=1}^{N_t} Y_k )

where:
    W_t      : Standard Brownian motion
    N_t      : Hawkes point process with intensity λ(t) = μ_h + ∫_0^t α e^{-β(t - s)} dN_s
    Y_k      : Independent jump return sizes ~ Normal(μ_J, σ_J²)

Theoretical Properties
----------------------
1. Endogenous Jump Cascades:
   Unlike classical Merton or Kou jump-diffusions where jumps arrive independently,
   an initial market shock in the Geometric Hawkes model increases the probability
   of follow-up shocks (aftershocks and market contagion).

Exactness Level: MOMENT_PROPAGATION.

References
----------
Stabile, G., & Torrisi, G. L. (2010). Risk processes with Hawkes arrival process.
*Queueing Systems*, 66(3), 293–314.
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


class GeometricHawkesProcess(Process):
    r"""
    Geometric Hawkes jump-diffusion asset price process.

    Parameters
    ----------
    mu : float
        Drift coefficient μ.
    sigma : float
        Continuous Brownian diffusion volatility σ > 0.
    hawkes_mu : float
        Baseline Hawkes jump intensity μ_h > 0.
    hawkes_alpha : float
        Self-excitation jump weight α ≥ 0.
    hawkes_beta : float
        Memory decay rate β > 0 (subcritical condition α < β).
    jump_mu : float
        Mean jump size μ_J in log-returns (defaults to -0.02).
    jump_sigma : float
        Volatility of jump sizes σ_J > 0 (defaults to 0.05).
    s0 : float, optional
        Initial asset price S_0 > 0 (defaults to 100.0).
    """

    def __init__(
        self,
        mu: float = 0.05,
        sigma: float = 0.15,
        hawkes_mu: float = 0.5,
        hawkes_alpha: float = 0.8,
        hawkes_beta: float = 1.2,
        jump_mu: float = -0.02,
        jump_sigma: float = 0.05,
        s0: float = 100.0,
    ) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")
        if hawkes_mu <= 0:
            raise ValueError(f"hawkes_mu must be strictly positive, got {hawkes_mu}")
        if hawkes_alpha < 0:
            raise ValueError(f"hawkes_alpha must be non-negative, got {hawkes_alpha}")
        if hawkes_beta <= 0:
            raise ValueError(f"hawkes_beta must be strictly positive, got {hawkes_beta}")
        if jump_sigma <= 0:
            raise ValueError(f"jump_sigma must be strictly positive, got {jump_sigma}")
        if s0 <= 0:
            raise ValueError(f"s0 must be strictly positive, got {s0}")

        self.mu = float(mu)
        self.sigma = float(sigma)
        self.hawkes_mu = float(hawkes_mu)
        self.hawkes_alpha = float(hawkes_alpha)
        self.hawkes_beta = float(hawkes_beta)
        self.jump_mu = float(jump_mu)
        self.jump_sigma = float(jump_sigma)
        self.s0 = float(s0)

        self.branching_ratio = self.hawkes_alpha / self.hawkes_beta

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"GeometricHawkes(μ={self.mu}, σ={self.sigma}, α={self.hawkes_alpha}, β={self.hawkes_beta})",
            math_note=f"S_t = S_0 exp((μ - ½σ²)t + σW_t + ∑ Y_k) with branching ratio {self.branching_ratio:.3f}",
            reference="Stabile & Torrisi (2010), Queueing Systems 66(3), 293–314",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "GeometricHawkesProcess features self-exciting clustered jumps and does not "
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
                "Geometric Hawkes process with clustered self-exciting jumps.",
                f"Hawkes branching ratio η = α/β = {self.branching_ratio:.4f}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.s0 * math.exp(self.mu)}
        if order >= 2:
            res[2] = self.sigma**2 + 0.05
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_hawkes_jumps(
        self,
        t_max: float,
        rng: np.random.Generator,
    ) -> list[tuple[float, float]]:
        r"""Simulate jump arrival times and sizes (t_k, Y_k)."""
        jumps: list[tuple[float, float]] = []
        t_curr = 0.0
        r_state = 0.0
        max_rate = self.hawkes_mu * 20.0 + 10.0

        while t_curr < t_max:
            dt = rng.exponential(1.0 / max_rate)
            t_prop = t_curr + dt
            if t_prop > t_max:
                break

            r_prop = r_state * math.exp(-self.hawkes_beta * dt)
            lam_prop = self.hawkes_mu + self.hawkes_alpha * r_prop

            u = rng.uniform(0.0, max_rate)
            if u <= lam_prop:
                t_curr = t_prop
                y_k = rng.normal(self.jump_mu, self.jump_sigma)
                jumps.append((t_curr, y_k))
                r_state = r_prop + 1.0
            else:
                t_curr = t_prop
                r_state = r_prop

        return jumps

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "clustered_jumps",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""Simulate sample paths of asset price S_t."""
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

        dt = times[1] - times[0]
        sqrt_dt = math.sqrt(dt)

        for p in range(paths_count):
            # Simulate continuous Brownian path
            z = rng.standard_normal(n_pts - 1)
            bm_inc = (self.mu - 0.5 * (self.sigma**2)) * dt + self.sigma * sqrt_dt * z
            log_s_cont = np.zeros(n_pts)
            log_s_cont[1:] = np.cumsum(bm_inc)

            # Simulate Hawkes jump events
            jumps = self.simulate_hawkes_jumps(t_max=times[-1], rng=rng)
            log_jump_cum = np.zeros(n_pts)
            for t_k, y_k in jumps:
                idx = np.searchsorted(times, t_k, side="left")
                if idx < n_pts:
                    log_jump_cum[idx:] += y_k

            s_paths[p, :] = self.s0 * np.exp(log_s_cont + log_jump_cum)

        return s_paths
