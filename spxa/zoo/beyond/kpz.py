r"""
Kardar–Parisi–Zhang (KPZ) Interface Growth Process (Kardar et al., 1986; Hairer, 2013).

The 1D non-linear stochastic partial differential equation describing kinetic
surface roughening, directed polymers in random media, and interacting particle flows:

    ∂_t h(t, x) = ν ∂_x² h(t, x) + (λ / 2) [ ∂_x h(t, x) ]² + ξ(t, x)

where:
    ν > 0   : Surface tension / smoothing relaxation
    λ ≠ 0   : Non-linear lateral growth velocity
    ξ(t, x) : Space-time Gaussian white noise with ⟨ξ(t, x) ξ(t', x')⟩ = D δ(t - t') δ(x - x')

Theoretical Properties
----------------------
1. Universal KPZ Scaling Exponents:
   - Growth exponent β = 1/3 (fluctuations grow as t^{1/3})
   - Roughness exponent χ = 1/2 (spatial correlation width scales as x^{1/2})
   - Dynamic exponent z = χ / β = 3/2

2. Tracy–Widom Fluctuation Universality:
   The rescaled interface height fluctuations (h(t, 0) - v_∞ t) / t^{1/3}
   asymptotically converge to the Tracy–Widom distribution from random matrix theory.

Exactness Level: MOMENT_PROPAGATION.

References
----------
Kardar, M., Parisi, G., & Zhang, Y. C. (1986). Dynamic scaling of growing
interfaces. *Physical Review Letters*, 56(9), 889.

Hairer, M. (2013). Solving the KPZ equation. *Annals of Mathematics*, 559–664.
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


class KPZInterfaceProcess(Process):
    r"""
    Kardar–Parisi–Zhang (KPZ) 1D interface growth lattice model.

    Parameters
    ----------
    nu : float
        Surface tension coefficient ν > 0.
    lam : float
        Non-linear lateral growth speed parameter λ.
    noise_amplitude : float
        Noise intensity D > 0.
    lattice_size : int
        Number of spatial discretization grid points L (defaults to 64).
    spatial_dx : float
        Spatial lattice spacing Δx (defaults to 0.1).
    """

    def __init__(
        self,
        nu: float = 0.5,
        lam: float = 1.0,
        noise_amplitude: float = 0.2,
        lattice_size: int = 64,
        spatial_dx: float = 0.1,
    ) -> None:
        if nu <= 0:
            raise ValueError(f"nu must be strictly positive, got {nu}")
        if noise_amplitude <= 0:
            raise ValueError(f"noise_amplitude must be strictly positive, got {noise_amplitude}")
        if lattice_size < 4:
            raise ValueError("lattice_size must be >= 4")
        if spatial_dx <= 0:
            raise ValueError("spatial_dx must be strictly positive")

        self.nu = float(nu)
        self.lam = float(lam)
        self.noise_amplitude = float(noise_amplitude)
        self.lattice_size = int(lattice_size)
        self.spatial_dx = float(spatial_dx)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"KPZInterface(ν={self.nu}, λ={self.lam}, D={self.noise_amplitude}, L={self.lattice_size})",
            math_note=f"∂_t h = ν ∂_xx h + ½λ (∂_x h)² + ξ(t,x); universal t^{{1/3}} Tracy-Widom scaling",
            reference="Kardar, Parisi & Zhang (1986), Phys. Rev. Lett. 56(9), 889; Hairer (2013)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "KPZInterfaceProcess is a non-linear stochastic partial differential equation (SPDE) "
            "and does not admit a 1D stationary Lévy–Khintchine triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=1.0 / 3.0,  # beta = 1/3
            tail_index=None,
            is_subordinator=False,
            hurst_index=0.5,  # chi = 1/2
            notes=[
                "Kardar-Parisi-Zhang interface growth stochastic PDE.",
                f"Universal growth exponent β = 1/3, roughness χ = 1/2.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: 0.0}
        if order >= 2:
            res[2] = 1.0
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Lattice SPDE Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "lattice_euler",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of the spatial-average interface height h_avg(t) = (1/L) ∑_x h(t, x).
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

        paths_count = int(n_paths if n_paths is not None else (num_paths if num_paths is not None else 50))
        n_pts = len(times)
        l_pts = self.lattice_size
        dx = self.spatial_dx

        avg_h_paths = np.empty((paths_count, n_pts), dtype=np.float64)
        avg_h_paths[:, 0] = 0.0

        # Current profile h(x): shape (paths_count, l_pts)
        h_profile = np.zeros((paths_count, l_pts), dtype=np.float64)

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)

            # Periodic finite differences
            h_left = np.roll(h_profile, 1, axis=1)
            h_right = np.roll(h_profile, -1, axis=1)

            # Discrete Laplacian ∂_xx h
            laplacian = (h_right - 2.0 * h_profile + h_left) / (dx**2)

            # Discrete slope ∂_x h
            grad = (h_right - h_left) / (2.0 * dx)

            # Space-time noise
            xi = rng.standard_normal((paths_count, l_pts)) * (self.noise_amplitude / math.sqrt(dx))

            # SDE update: ∂_t h = ν ∇²h + ½λ (∇h)² + ξ
            dh = (self.nu * laplacian + 0.5 * self.lam * (grad**2)) * dt + xi * sqrt_dt
            h_profile += dh

            avg_h_paths[:, i] = np.mean(h_profile, axis=1)

        return avg_h_paths


# Aliases
KPZProcess = KPZInterfaceProcess
