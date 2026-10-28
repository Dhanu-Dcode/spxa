r"""
Brownian Motion on Riemannian Manifolds (Laplace–Beltrami Diffusion) (Hsu, 2002).

A stochastic diffusion process evolving intrinsically on a curved Riemannian
manifold (M, g) whose infinitesimal Dynkin generator is the Laplace–Beltrami
operator Δ_M:

    Af(x) = (1/2) Δ_M f(x) = (1/2) (1 / √|g|) ∂_i ( √|g| g^{ij} ∂_j f(x) )

In extrinsic coordinates on the unit 2-sphere S² = {x ∈ ℝ³ : ||x|| = 1} (curvature K = +1):
    dX_t = P_{X_t}(dW_t) - X_t dt = (I - X_t X_t^⊤) ∘ dW_t

where:
    P_x = I - x x^⊤  is the orthogonal projector onto the tangent space T_x S²
    ∘ denotes the Stratonovich stochastic integral preserving the sphere constraint ||X_t|| = 1.

Theoretical Properties
----------------------
1. Invariant Curvature & Area:
   The process preserves Riemannian geometry with positive Gaussian curvature K = +1
   on S² (and negative curvature K = -1 on the hyperbolic plane ℍ²).

2. Uniform Ergodic Measure:
   As t → ∞, the distribution of X_t on S² converges to the uniform Haar measure.

Exactness Level: MOMENT_PROPAGATION.

References
----------
Hsu, E. P. (2002). *Stochastic Analysis on Manifolds*.
American Mathematical Society, Graduate Studies in Mathematics, Vol. 38.
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


class RiemannianBrownianMotion(Process):
    r"""
    Brownian Motion on the 2-Sphere S² (Laplace–Beltrami diffusion).

    Parameters
    ----------
    radius : float
        Sphere radius R > 0 (curvature K = 1 / R²; defaults to 1.0).
    diff_scale : float
        Diffusion scale parameter σ > 0.
    """

    def __init__(
        self,
        radius: float = 1.0,
        diff_scale: float = 1.0,
    ) -> None:
        if radius <= 0:
            raise ValueError(f"radius must be strictly positive, got {radius}")
        if diff_scale <= 0:
            raise ValueError(f"diff_scale must be strictly positive, got {diff_scale}")

        self.radius = float(radius)
        self.diff_scale = float(diff_scale)
        self.curvature = 1.0 / (self.radius**2)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"RiemannianBM(R={self.radius}, K={self.curvature:.2f})",
            math_note=f"Laplace-Beltrami diffusion ½Δ_M on S² with Gaussian curvature K={self.curvature:.3f}",
            reference="Hsu (2002), Stochastic Analysis on Manifolds, AMS GSM Vol. 38",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "RiemannianBrownianMotion evolves on a curved Riemannian manifold and does not "
            "admit a flat Euclidean Lévy–Khintchine triplet."
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
                f"Brownian motion on the 2-sphere S² with radius R = {self.radius}.",
                f"Gaussian curvature K = {self.curvature}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.radius}
        if order >= 2:
            res[2] = 0.0  # Constant radius
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Exact Extrinsic Projection Stratonovich Simulation
    # ------------------------------------------------------------------

    def simulate_sphere_3d(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: int = 100,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""
        Simulate 3D coordinates X_t ∈ ℝ³ strictly satisfying ||X_t|| = R.

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, len(t_grid), 3) of points on the sphere.
        """
        if rng is None:
            rng = np.random.default_rng()

        times = np.linspace(0.0, 1.0 if t_grid is None else float(t_grid), 100) if isinstance(t_grid, (int, float, type(None))) else np.asarray(t_grid)
        n_pts = len(times)

        # Initial point at North Pole (0, 0, R)
        paths_3d = np.empty((n_paths, n_pts, 3), dtype=np.float64)
        paths_3d[:, 0, :] = np.array([0.0, 0.0, self.radius])

        curr_pos = paths_3d[:, 0, :].copy()

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)

            # Unconstrained 3D Gaussian noise
            dw = rng.standard_normal((n_paths, 3)) * (self.diff_scale * sqrt_dt)

            # Project noise onto tangent space: P_x(dW) = dW - (x · dW / ||x||²) x
            x_norm_sq = np.sum(curr_pos**2, axis=1, keepdims=True)
            radial_proj = np.sum(curr_pos * dw, axis=1, keepdims=True) / x_norm_sq
            tangent_dw = dw - radial_proj * curr_pos

            # Retraction update onto sphere (exact geodesic exponential map)
            v_norm = np.linalg.norm(tangent_dw, axis=1, keepdims=True)
            theta = v_norm / self.radius

            # Geodesic formula on sphere: x cos(θ) + (v / ||v||) R sin(θ)
            safe_v = np.where(v_norm < 1e-12, 1.0, v_norm)
            unit_v = tangent_dw / safe_v
            next_pos = curr_pos * np.cos(theta) + unit_v * self.radius * np.sin(theta)

            # Strict normalization to radius R
            curr_pos = next_pos * (self.radius / np.linalg.norm(next_pos, axis=1, keepdims=True))
            paths_3d[:, i, :] = curr_pos

        return paths_3d

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "geodesic_retraction",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""Simulate geodesic distance from North Pole d_R(X_t, (0,0,R))."""
        paths_count = int(n_paths if n_paths is not None else (num_paths if num_paths is not None else 100))
        grid = t_grid if t_grid is not None else float(T if T is not None else (t_max if t_max is not None else 1.0))
        p3d = self.simulate_sphere_3d(t_grid=grid, n_paths=paths_count, rng=rng)

        # Geodesic distance: R * arccos(z / R)
        z_coords = np.clip(p3d[:, :, 2] / self.radius, -1.0, 1.0)
        return self.radius * np.arccos(z_coords)


# Aliases
ManifoldBM = RiemannianBrownianMotion
