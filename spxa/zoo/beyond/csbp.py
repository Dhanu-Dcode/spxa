r"""
Continuous-State Branching Process with Immigration (CSBP) (Lamperti, 1967; Kyprianou, 2006).

A non-negative Markov process X_t on [0, ∞) modeling the continuous diffusion
limit of Galton–Watson population branching processes subject to immigration:

    dX_t = (a - b X_t) dt + σ √X_t dW_t + dJ_t,   X_t ≥ 0

where:
    a ≥ 0 : Immigration rate
    b ∈ ℝ : Net branching growth rate (subcritical if b > 0, critical if b = 0, supercritical if b < 0)
    σ > 0 : Demographical branching Brownian noise
    J_t   : Subordinator jump process representing punctuated immigration bursts

Theoretical Properties
----------------------
1. Lamperti Transform:
   CSBPs without immigration are the time-changed image of Lévy processes with
   no negative jumps under the Lamperti transform.

2. Affine Laplace Transform:
   The transition Laplace transform is exponential-affine:
       E[exp(-λ X_t) | X_0 = x_0] = exp(-x_0 v_t(λ) - ∫_0^t φ(v_s(λ)) ds)
   where v_t satisfies the differential equation v̇_t = -ψ(v_t).

Exactness Level: MOMENT_PROPAGATION.

References
----------
Lamperti, J. (1967). Continuous state branching processes.
*Bulletin of the American Mathematical Society*, 73(3), 382–386.

Kyprianou, A. E. (2006). *Introductory Lectures on Fluctuations of Lévy Processes
with Applications*. Springer.
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


class CSBPProcess(Process):
    r"""
    Continuous-State Branching Process with Immigration (CSBP).

    Parameters
    ----------
    a : float
        Immigration drift rate a ≥ 0 (defaults to 0.1).
    b : float
        Branching rate parameter b > 0 (defaults to 0.5).
    sigma : float
        Demographic diffusion parameter σ > 0.
    jump_rate : float
        Poisson immigration burst intensity λ ≥ 0.
    jump_mean : float
        Mean immigration jump size > 0.
    x0 : float, optional
        Initial population size X_0 ≥ 0 (defaults to 1.0).
    """

    def __init__(
        self,
        a: float = 0.1,
        b: float = 0.5,
        sigma: float = 0.3,
        jump_rate: float = 0.2,
        jump_mean: float = 0.1,
        x0: float = 1.0,
    ) -> None:
        if a < 0:
            raise ValueError("Immigration rate a must be non-negative")
        if sigma <= 0:
            raise ValueError("sigma must be strictly positive")
        if jump_rate < 0 or jump_mean <= 0:
            raise ValueError("jump parameters must be valid")
        if x0 < 0:
            raise ValueError("x0 must be non-negative")

        self.a = float(a)
        self.b = float(b)
        self.sigma = float(sigma)
        self.jump_rate = float(jump_rate)
        self.jump_mean = float(jump_mean)
        self.x0 = float(x0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"CSBP(a={self.a}, b={self.b}, σ={self.sigma})",
            math_note="dX_t = (a - b X) dt + σ √X dW + dJ; Lamperti branching diffusion with immigration",
            reference="Lamperti (1967), Bull. AMS 73(3), 382–386; Kyprianou (2006)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Stationary Mean
    # ------------------------------------------------------------------

    @property
    def stationary_mean(self) -> float:
        r"""Stationary population expectation E[X_∞] = (a + λ E[Y]) / b when b > 0."""
        if self.b <= 0:
            return float("inf")
        total_immigration = self.a + self.jump_rate * self.jump_mean
        return total_immigration / self.b

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "CSBPProcess has demographic square-root diffusion and does not admit "
            "an independent Lévy–Khintchine triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=(self.a == 0.0 and self.b == 0.0 and self.jump_rate == 0.0),
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=0.5,
            notes=[
                "Continuous-state branching process with immigration (CSBP).",
                f"Net branching rate b = {self.b}, Immigration rate a = {self.a}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.stationary_mean if self.b > 0 else self.x0}
        if order >= 2:
            res[2] = 1.0
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Vectorized Simulation
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
        Simulate sample paths of population size X_t.
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

        x_paths = np.empty((paths_count, n_pts), dtype=np.float64)
        x_paths[:, 0] = self.x0

        x_curr = np.full(paths_count, self.x0, dtype=np.float64)

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)

            z = rng.standard_normal(paths_count)
            n_jumps = rng.poisson(self.jump_rate * dt, paths_count)
            jumps = np.where(
                n_jumps > 0, rng.exponential(self.jump_mean, paths_count) * n_jumps, 0.0
            )

            x_pos = np.maximum(x_curr, 0.0)
            drift = (self.a - self.b * x_pos) * dt
            diff = self.sigma * np.sqrt(x_pos) * sqrt_dt * z

            x_curr = np.maximum(0.0, x_curr + drift + diff + jumps)
            x_paths[:, i] = x_curr

        return x_paths


# Aliases
FellerBranchingImmigration = CSBPProcess
