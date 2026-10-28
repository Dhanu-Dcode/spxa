r"""
Dyson Brownian Motion (Dyson, 1962).

The interacting particle system describing the diffusion of the N eigenvalues
λ_1 < λ_2 < ... < λ_N of an N × N Hermitian random matrix evolving under the
Gaussian Unitary Ensemble (GUE):

    dλ_i = (1 / √N) dW_t^i + (1 / N) ∑_{j ≠ i} [ 1 / (λ_i - λ_j) ] dt

where W_t^i are N independent standard Brownian motions and the drift represents
logarithmic 2D Coulomb electrostatic repulsion.

Theoretical Properties
----------------------
1. No-Collision Property:
   Because the logarithmic repulsion potential diverges as λ_i → λ_j, the eigenvalues
   never collide almost surely for all t > 0:
       P(∃ t > 0, i ≠ j : λ_i(t) = λ_j(t)) = 0

2. Wigner Semicircle Law Asymptotics:
   As N → ∞ and t → ∞ (with harmonic confinement), the empirical spectral density
   converges weakly to the Wigner Semicircle Distribution:
       ρ(x) = (1 / (2π)) √(4 - x²)   for x ∈ [-2, 2]

Exactness Level: MOMENT_PROPAGATION.

References
----------
Dyson, F. J. (1962). A Brownian-motion model for the eigenvalues of a random matrix.
*Journal of Mathematical Physics*, 3(6), 1191–1198.
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


class DysonBrownianMotion(Process):
    r"""
    Dyson Brownian Motion for N interacting GUE eigenvalues.

    Parameters
    ----------
    num_particles : int
        Number of eigenvalues N ≥ 2 (defaults to 10).
    confinement : float
        Harmonic confinement potential strength γ ≥ 0 (drifts towards 0; defaults to 1.0).
    """

    def __init__(
        self,
        num_particles: int = 10,
        confinement: float = 1.0,
    ) -> None:
        if num_particles < 2:
            raise ValueError("num_particles must be at least 2")
        if confinement < 0:
            raise ValueError("confinement must be non-negative")

        self.num_particles = int(num_particles)
        self.confinement = float(confinement)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"DysonBM(N={self.num_particles}, γ={self.confinement})",
            math_note="dλ_i = (1/√N) dW_i + [1/N ∑ 1/(λ_i - λ_j) - γ λ_i] dt with Coulomb repulsion",
            reference="Dyson (1962), J. Math. Phys. 3(6), 1191–1198",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Asymptotic Semicircle Density
    # ------------------------------------------------------------------

    def wigner_semicircle_density(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Wigner semicircle density ρ(x) = (1 / 2π) √(4 - x²) on [-2, 2]."""
        x_arr = np.asarray(x, dtype=np.float64)
        rad = 4.0 - x_arr**2
        pdf = np.where(rad >= 0.0, np.sqrt(np.maximum(0.0, rad)) / (2.0 * math.pi), 0.0)
        if np.isscalar(x):
            return float(pdf)
        return pdf

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "DysonBrownianMotion is an interacting multi-particle system with Coulomb singular "
            "drift and does not admit an independent Lévy–Khintchine triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=(self.confinement == 0.0),
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=0.5,
            notes=[
                f"Dyson Brownian Motion for N = {self.num_particles} interacting GUE eigenvalues.",
                "Logarithmic Coulomb repulsion prevents eigenvalue collisions.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: 0.0}
        if order >= 2:
            res[2] = 1.0  # Semicircle variance = R^2 / 4 = 1.0
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Vectorized Simulation
    # ------------------------------------------------------------------

    def simulate_particles(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""
        Simulate all N eigenvalue trajectories across time.

        Returns
        -------
        np.ndarray
            Array of shape (len(t_grid), N) of ordered eigenvalues λ_1 < ... < λ_N.
        """
        if rng is None:
            rng = np.random.default_rng()

        times = (
            np.linspace(0.0, 1.0 if t_grid is None else float(t_grid), 200)
            if isinstance(t_grid, (int, float, type(None)))
            else np.asarray(t_grid)
        )
        n_pts = len(times)
        n_p = self.num_particles

        # Initial eigenvalues evenly spaced around 0
        lambda_trajectories = np.empty((n_pts, n_p), dtype=np.float64)
        lambda_trajectories[0, :] = np.linspace(-1.0, 1.0, n_p)

        curr_lambdas = lambda_trajectories[0, :].copy()
        inv_sqrt_n = 1.0 / math.sqrt(n_p)
        inv_n = 1.0 / n_p

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)

            # Compute pairwise Coulomb repulsion forces: ∑_{j ≠ i} 1 / (λ_i - λ_j)
            diff = curr_lambdas[:, None] - curr_lambdas[None, :]
            np.fill_diagonal(diff, np.inf)  # avoid division by zero
            # Regularize against tiny denominators during discrete Euler steps
            safe_diff = np.where(np.abs(diff) < 1e-4, np.sign(diff) * 1e-4, diff)
            repulsion = np.sum(1.0 / safe_diff, axis=1)

            # Drift = (1/N) repulsion - γ λ_i
            drift = inv_n * repulsion - self.confinement * curr_lambdas

            # Gaussian noise
            dw = rng.standard_normal(n_p) * (inv_sqrt_n * sqrt_dt)

            curr_lambdas = curr_lambdas + drift * dt + dw
            # Sort to preserve eigenvalue ordering
            curr_lambdas.sort()
            lambda_trajectories[i, :] = curr_lambdas

        return lambda_trajectories

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "top_eigenvalue",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""Simulate sample paths of the maximal eigenvalue λ_max(t)."""
        paths_count = int(
            n_paths if n_paths is not None else (num_paths if num_paths is not None else 50)
        )
        grid = (
            t_grid
            if t_grid is not None
            else float(T if T is not None else (t_max if t_max is not None else 1.0))
        )
        times = np.linspace(0.0, grid, 100) if isinstance(grid, (int, float)) else np.asarray(grid)

        max_paths = np.empty((paths_count, len(times)), dtype=np.float64)
        for p in range(paths_count):
            trajs = self.simulate_particles(t_grid=times, rng=rng)
            max_paths[p, :] = trajs[:, -1]

        return max_paths
