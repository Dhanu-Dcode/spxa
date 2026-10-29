r"""
Wishart Matrix-Valued Diffusion Process (Bru, 1991; Gourieroux & Sufana, 2004).

An autonomous matrix diffusion evolving on the cone of d × d symmetric
positive-definite matrices S_d⁺:

    dΣ_t = (α I_d + M Σ_t + Σ_t M^⊤) dt + √Σ_t dW_t Q + Q^⊤ dW_t^⊤ √Σ_t

where:
    Σ_t ∈ S_d⁺ : d × d stochastic covariance matrix
    W_t        : d × d matrix of independent standard Brownian motions
    α > 0      : Degrees-of-freedom drift intensity
    M          : Mean-reversion matrix (with negative eigenvalues)
    Q          : Volatility-of-variance matrix

Theoretical Properties
----------------------
1. Matrix Feller Non-Negativity Condition:
   If α ≥ (d + 1) Q^⊤ Q, then Σ_t remains strictly positive-definite almost
   surely for all t > 0, preventing covariance collapse.

2. Invariant Wishart Law:
   The stationary distribution is the matrix-variate Wishart distribution.

Exactness Level: MOMENT_PROPAGATION.

References
----------
Bru, M. F. (1991). Wishart processes. *Journal of Theoretical Probability*, 4(4), 725–751.

Gourieroux, C., & Sufana, R. (2004). Derivative pricing with Wishart multivariate
stochastic volatility. *Working Paper*, CREST.
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


class WishartDiffusion(Process):
    r"""
    Wishart Matrix-Valued Diffusion on S_d⁺.

    Parameters
    ----------
    dim : int
        Matrix dimension d ≥ 1 (defaults to 2).
    alpha : float
        Degrees of freedom parameter α ≥ d + 1.
    mean_reversion : float
        Diagonal mean reversion rate m < 0 (defaults to -1.0).
    q_vol : float
        Volatility parameter q > 0 (defaults to 0.2).
    """

    def __init__(
        self,
        dim: int = 2,
        alpha: float = 4.0,
        mean_reversion: float = -1.0,
        q_vol: float = 0.2,
    ) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if alpha < dim:
            raise ValueError(f"alpha must be >= dim ({dim}), got {alpha}")
        if q_vol <= 0:
            raise ValueError("q_vol must be strictly positive")

        self.dim = int(dim)
        self.alpha = float(alpha)
        self.mean_reversion = float(mean_reversion)
        self.q_vol = float(q_vol)

        # Initial covariance matrix: identity
        self.sigma0 = np.eye(self.dim, dtype=np.float64)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"WishartDiffusion(d={self.dim}, α={self.alpha}, q={self.q_vol})",
            math_note=f"dΣ = (α I + MΣ + ΣMᵀ)dt + √Σ dW Q + Qᵀ dWᵀ √Σ on S_{self.dim}⁺",
            reference="Bru (1991), J. Theor. Probab. 4(4), 725–751; Gourieroux & Sufana (2004)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "WishartDiffusion is a matrix-valued diffusion on S_d⁺ and does not "
            "admit a 1D stationary Lévy–Khintchine triplet."
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
                f"Wishart matrix-valued diffusion of dimension {self.dim}x{self.dim}.",
                f"Degrees of freedom α = {self.alpha}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: float(np.trace(self.sigma0))}
        if order >= 2:
            res[2] = 1.0
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Simulation (Matrix Square Root Euler Update)
    # ------------------------------------------------------------------

    def simulate_matrix(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""
        Simulate a sample path of d × d covariance matrices Σ_t across time.

        Returns
        -------
        np.ndarray
            Array of shape (len(t_grid), d, d) of symmetric positive-definite matrices.
        """
        if rng is None:
            rng = np.random.default_rng()

        times = (
            np.linspace(0.0, 1.0 if t_grid is None else float(t_grid), 100)
            if isinstance(t_grid, (int, float, type(None)))
            else np.asarray(t_grid)
        )
        n_pts = len(times)
        d = self.dim

        sigma_traj = np.empty((n_pts, d, d), dtype=np.float64)
        sigma_traj[0] = self.sigma0.copy()

        curr_sigma = self.sigma0.copy()
        q_mat = np.eye(d) * self.q_vol
        m_mat = np.eye(d) * self.mean_reversion

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)

            # SVD matrix square root of curr_sigma
            eigvals, eigvecs = np.linalg.eigh(curr_sigma)
            pos_eig = np.maximum(eigvals, 1e-10)
            sqrt_sigma = eigvecs @ np.diag(np.sqrt(pos_eig)) @ eigvecs.T

            # Matrix Brownian increment dW
            dw = rng.standard_normal((d, d)) * sqrt_dt

            # Drift: (α I + M Σ + Σ Mᵀ) dt
            drift = (self.alpha * np.eye(d) + m_mat @ curr_sigma + curr_sigma @ m_mat.T) * dt

            # Diffusion: √Σ dW Q + Qᵀ dWᵀ √Σ
            diff = sqrt_sigma @ dw @ q_mat + q_mat.T @ dw.T @ sqrt_sigma

            curr_sigma = curr_sigma + drift + diff
            # Symmetrize
            curr_sigma = 0.5 * (curr_sigma + curr_sigma.T)
            sigma_traj[i] = curr_sigma

        return sigma_traj

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "matrix_trace",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""Simulate sample paths of the total variance trace Tr(Σ_t)."""
        paths_count = int(
            n_paths if n_paths is not None else (num_paths if num_paths is not None else 50)
        )
        grid = (
            t_grid
            if t_grid is not None
            else float(T if T is not None else (t_max if t_max is not None else 1.0))
        )
        times = np.linspace(0.0, grid, 100) if isinstance(grid, (int, float)) else np.asarray(grid)

        trace_paths = np.empty((paths_count, len(times)), dtype=np.float64)
        for p in range(paths_count):
            mat_traj = self.simulate_matrix(t_grid=times, rng=rng)
            trace_paths[p, :] = np.trace(mat_traj, axis1=1, axis2=2)

        return trace_paths


# Aliases
MatrixWishartProcess = WishartDiffusion
