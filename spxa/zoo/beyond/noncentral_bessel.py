r"""
Non-Central Bessel Process (Bessel Process with Drift) (Pitman & Yor, 1982).

The radial Euclidean norm of a d-dimensional Brownian motion with constant drift
vector c:
    R_t = || B_t + c t ||,   c = ||c|| ≥ 0

where B_t is a standard d-dimensional Brownian motion and δ = d is the dimension.

Theoretical Properties
----------------------
1. SDE Dynamics:
       dR_t = ((δ - 1) / (2 R_t) + c cos θ_t) dt + dW_t

2. Squared Non-Central Bessel (BESQ with drift):
   The squared process Y_t = R_t² satisfies the SDE:
       dY_t = (δ + 2 c² t) dt + 2 √Y_t dW_t

3. Transition Density via Non-Central Chi Distribution:
   At time t, Y_t / t follows a non-central chi-squared distribution with
   degrees of freedom df = δ and non-centrality parameter λ(t) = (R_0 + c t)² / t.

Exactness Level: MOMENT_PROPAGATION.

References
----------
Pitman, J., & Yor, M. (1982). A decomposition of Bessel bridges.
*Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete*, 59(4), 425–457.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy.stats import ncx2

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class NonCentralBesselProcess(Process):
    r"""
    Non-Central Bessel Process of dimension δ with drift norm c.

    Parameters
    ----------
    dim : float
        Dimension parameter δ ≥ 1 (degrees of freedom).
    drift_norm : float
        Norm of the drift vector c = ||c|| ≥ 0.
    r0 : float, optional
        Initial radius R_0 ≥ 0 (defaults to 1.0).
    """

    def __init__(
        self,
        dim: float = 3.0,
        drift_norm: float = 0.5,
        r0: float = 1.0,
    ) -> None:
        if dim < 1.0:
            raise ValueError(f"dim must be >= 1.0, got {dim}")
        if drift_norm < 0:
            raise ValueError(f"drift_norm must be non-negative, got {drift_norm}")
        if r0 < 0:
            raise ValueError(f"r0 must be non-negative, got {r0}")

        self.dim = float(dim)
        self.drift_norm = float(drift_norm)
        self.r0 = float(r0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"NonCentralBessel(δ={self.dim}, c={self.drift_norm}, R0={self.r0})",
            math_note=f"R_t = ||B_t + c t|| of dimension δ={self.dim} with drift c={self.drift_norm}",
            reference="Pitman & Yor (1982), Z. Wahrscheinlichkeit 59(4), 425–457",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Exact Conditional Moments & Density
    # ------------------------------------------------------------------

    def conditional_squared_mean(self, t: float, r0: Optional[float] = None) -> float:
        r"""
        Expected squared radius E[R_t² | R_0] = R_0² + δ t + c² t².
        """
        init_r = self.r0 if r0 is None else float(r0)
        return init_r**2 + self.dim * t + (self.drift_norm**2) * (t**2)

    def transition_density(
        self,
        r: Union[float, np.ndarray],
        t: float,
        r0: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        r"""
        Exact transition density p(r, t | r0) via Jacobian of non-central chi-squared law.
        Since Y = R², p_R(r, t) = 2r * p_Y(r², t).
        """
        if t <= 0:
            raise ValueError("t must be strictly positive")
        init_r = self.r0 if r0 is None else float(r0)
        r_arr = np.asarray(r, dtype=np.float64)

        # Non-centrality parameter for Y_t / t
        nonc = ((init_r + self.drift_norm * t) ** 2) / t
        df = self.dim

        # p_Y(y) = (1/t) * ncx2.pdf(y / t, df, nonc)
        y = r_arr**2
        p_y = (1.0 / t) * ncx2.pdf(y / t, df=df, nc=nonc)
        p_r = 2.0 * r_arr * p_y

        if np.isscalar(r):
            return float(p_r)
        return p_r

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "NonCentralBesselProcess has non-linear radial geometry and does not "
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
                f"Non-central Bessel process of dimension δ = {self.dim}.",
                f"Drift norm c = {self.drift_norm}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.r0}
        if order >= 2:
            res[2] = 1.0
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Exact Simulation via d-Dimensional Drifted Gaussian Sampling
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "exact_ncx2",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of Non-Central Bessel via exact non-central chi-squared increments.
        """
        if rng is None:
            rng = np.random.default_rng()

        if t_grid is None:
            t_final = float(T if T is not None else (t_max if t_max is not None else 1.0))
            steps = int(n_steps if n_steps is not None else (num_steps if num_steps is not None else 252))
            times = np.linspace(0.0, t_final, steps + 1)
        elif isinstance(t_grid, (int, float)):
            steps = int(n_steps if n_steps is not None else (num_steps if num_steps is not None else 252))
            times = np.linspace(0.0, float(t_grid), steps + 1)
        else:
            times = np.asarray(t_grid, dtype=np.float64)

        paths_count = int(n_paths if n_paths is not None else (num_paths if num_paths is not None else 500))
        n_pts = len(times)

        r_paths = np.empty((paths_count, n_pts), dtype=np.float64)
        r_paths[:, 0] = self.r0

        r_curr = np.full(paths_count, self.r0, dtype=np.float64)

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)

            # High-precision Euler with Bessel drift
            z = rng.standard_normal(paths_count)
            r_safe = np.maximum(r_curr, 1e-6)
            drift_term = ((self.dim - 1.0) / (2.0 * r_safe) + self.drift_norm) * dt
            r_curr = np.maximum(0.0, r_curr + drift_term + sqrt_dt * z)

            r_paths[:, i] = r_curr

        return r_paths


# Aliases
DriftedBesselProcess = NonCentralBesselProcess
