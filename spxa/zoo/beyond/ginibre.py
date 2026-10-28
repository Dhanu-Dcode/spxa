r"""
Ginibre Ensemble Complex Eigenvalue Process (Ginibre, 1965).

Describes the 2D diffusion of the N complex eigenvalues z_1, ..., z_N ∈ ℂ
of an N × N non-Hermitian random matrix:

    dz_i = (1 / √(2N)) (dW_t^{x,i} + i dW_t^{y,i}) + (1 / N) ∑_{j ≠ i} [ 1 / (z_i^* - z_j^*) ] dt - γ z_i dt

where:
    z_i^* = x_i - i y_i is the complex conjugate
    γ ≥ 0               is the harmonic trapping potential

Theoretical Properties
----------------------
1. 2D Coulomb Plasma:
   The eigenvalues form a 2D one-component Coulomb gas experiencing 2D electrostatic
   logarithmic repulsion V(z, w) = -ln |z - w|.

2. Girko's Circular Law Asymptotics:
   As N → ∞, the empirical density in the complex plane converges to the
   uniform distribution on the unit disk:
       ρ(z) = (1 / π) 1_{|z| ≤ 1}

Exactness Level: MOMENT_PROPAGATION.

References
----------
Ginibre, J. (1965). Statistical ensembles of complex, quaternion, and real matrices.
*Journal of Mathematical Physics*, 6(3), 440–449.
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


class GinibreEigenvalueProcess(Process):
    r"""
    Ginibre Ensemble 2D Complex Eigenvalue Process.

    Parameters
    ----------
    num_particles : int
        Number of complex eigenvalues N ≥ 2 (defaults to 10).
    confinement : float
        Harmonic confinement strength γ ≥ 0 (defaults to 1.0).
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
            name=f"GinibreEigenvalues(N={self.num_particles}, γ={self.confinement})",
            math_note=f"2D complex Coulomb gas dz_i = (1/√(2N)) dW_C + (1/N) ∑ 1/(z_i^* - z_j^*) dt - γ z_i dt",
            reference="Ginibre (1965), J. Math. Phys. 6(3), 440–449",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "GinibreEigenvalueProcess is a 2D interacting Coulomb gas and does not "
            "admit an independent 1D Lévy–Khintchine triplet."
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
                f"Ginibre 2D complex eigenvalue diffusion with N = {self.num_particles} particles.",
                "Converges to Girko's Circular Law on the unit disk |z| ≤ 1.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: 0.0}
        if order >= 2:
            res[2] = 0.5  # Mean radius squared on unit disk = 1/2
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Complex Particle Simulation
    # ------------------------------------------------------------------

    def simulate_complex_particles(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""
        Simulate all N complex eigenvalues z_k(t) ∈ ℂ.

        Returns
        -------
        np.ndarray
            Array of shape (len(t_grid), N) of complex numbers.
        """
        if rng is None:
            rng = np.random.default_rng()

        times = np.linspace(0.0, 1.0 if t_grid is None else float(t_grid), 100) if isinstance(t_grid, (int, float, type(None))) else np.asarray(t_grid)
        n_pts = len(times)
        n_p = self.num_particles

        # Initial particles on small circle
        theta = np.linspace(0.0, 2.0 * math.pi, n_p, endpoint=False)
        z_curr = 0.5 * np.exp(1j * theta)

        z_traj = np.empty((n_pts, n_p), dtype=np.complex128)
        z_traj[0] = z_curr.copy()

        scale_w = 1.0 / math.sqrt(2.0 * n_p)
        inv_n = 1.0 / n_p

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)

            # Pairwise 2D Coulomb force: 1 / (z_i^* - z_j^*)
            z_conj = np.conj(z_curr)
            diff = z_conj[:, None] - z_conj[None, :]
            np.fill_diagonal(diff, np.inf)
            safe_diff = np.where(np.abs(diff) < 1e-4, np.exp(1j * np.angle(diff)) * 1e-4, diff)
            repulsion = np.sum(1.0 / safe_diff, axis=1)

            drift = inv_n * repulsion - self.confinement * z_curr

            # 2D Gaussian complex noise
            dw = (rng.standard_normal(n_p) + 1j * rng.standard_normal(n_p)) * (scale_w * sqrt_dt)

            z_curr = z_curr + drift * dt + dw
            z_traj[i] = z_curr

        return z_traj

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "max_radius",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""Simulate sample paths of the spectral radius R_max(t) = max_i |z_i(t)|."""
        paths_count = int(n_paths if n_paths is not None else (num_paths if num_paths is not None else 30))
        grid = t_grid if t_grid is not None else float(T if T is not None else (t_max if t_max is not None else 1.0))
        times = np.linspace(0.0, grid, 50) if isinstance(grid, (int, float)) else np.asarray(grid)

        r_paths = np.empty((paths_count, len(times)), dtype=np.float64)
        for p in range(paths_count):
            trajs = self.simulate_complex_particles(t_grid=times, rng=rng)
            r_paths[p, :] = np.max(np.abs(trajs), axis=1)

        return r_paths
