r"""
Fractional Ornstein–Uhlenbeck Process (Cheridito et al., 2003).

The Fractional Ornstein–Uhlenbeck (fOU) process satisfies the SDE:
    dX_t = -θ (X_t - μ) dt + σ dB_t^H,   X_0 = x_0

driven by a fractional Brownian motion B_t^H with Hurst parameter H ∈ (0, 1).

Theoretical Properties
----------------------
1. Explicit Integral Representation:
       X_t = μ + (x_0 - μ) e^{-θ t} + σ ∫_0^t e^{-θ(t - s)} dB_s^H

2. Long-Range Dependence:
   - For H > 1/2: Increments are positively correlated, and the autocovariance
     decays as a power law: Cov(X_t, X_{t+τ}) ~ τ^{2H - 2} as τ → ∞, exhibiting
     genuine long memory.
   - For H = 1/2: Recovers the classic Markovian Ornstein–Uhlenbeck diffusion.
   - For H < 1/2: Sub-diffusive and antipersistent (rough paths).

3. Stationary Variance:
   The stationary asymptotic variance is given by:
       Var(X_∞) = H Γ(2H) σ² / θ^{2H}

Exactness Level: MOMENT_PROPAGATION.

References
----------
Cheridito, P., Kawaguchi, H., & Maejima, M. (2003). Fractional Ornstein–Uhlenbeck
processes. *Electronic Journal of Probability*, 8, 1–34.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy.special import gamma

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class FractionalOrnsteinUhlenbeck(Process):
    r"""
    Fractional Ornstein–Uhlenbeck (fOU) process.

    Parameters
    ----------
    theta : float
        Mean-reversion rate θ > 0.
    mu : float
        Long-term mean μ.
    sigma : float
        Volatility parameter σ > 0.
    hurst : float
        Hurst parameter H ∈ (0, 1).
    x0 : float, optional
        Initial value X_0 (defaults to μ).
    """

    def __init__(
        self,
        theta: float = 1.0,
        mu: float = 0.0,
        sigma: float = 0.5,
        hurst: float = 0.7,
        x0: Optional[float] = None,
    ) -> None:
        if theta <= 0:
            raise ValueError(f"theta must be strictly positive, got {theta}")
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")
        if not (0.0 < hurst < 1.0):
            raise ValueError(f"hurst must be in (0, 1), got {hurst}")

        self.theta = float(theta)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.hurst = float(hurst)
        self.x0 = float(mu if x0 is None else x0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"FractionalOU(θ={self.theta}, μ={self.mu}, σ={self.sigma}, H={self.hurst})",
            math_note=f"dX_t = -θ(X_t - μ)dt + σ dB_t^H with Hurst index H={self.hurst}",
            reference="Cheridito, Kawaguchi & Maejima (2003), Electron. J. Probab. 8, 1–34",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Analytical Moments
    # ------------------------------------------------------------------

    @property
    def stationary_variance(self) -> float:
        r"""
        Stationary variance Var(X_∞) = H Γ(2H) σ² / θ^{2H}.
        """
        h = self.hurst
        return (h * gamma(2.0 * h) * (self.sigma**2)) / (self.theta ** (2.0 * h))

    def conditional_mean(self, t: float, x0: Optional[float] = None) -> float:
        r"""Expectation E[X_t | X_0]."""
        init_x = self.x0 if x0 is None else float(x0)
        return self.mu + (init_x - self.mu) * math.exp(-self.theta * t)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "FractionalOrnsteinUhlenbeck process has memory-dependent increments and "
            "does not admit an independent Lévy–Khintchine triplet."
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
            hurst_index=self.hurst,
            notes=[
                f"Fractional Ornstein-Uhlenbeck process with Hurst index H = {self.hurst}.",
                "Exhibits long-range dependence for H > 0.5."
                if self.hurst > 0.5
                else "Sub-diffusive / anti-persistent for H < 0.5.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.mu}
        if order >= 2:
            res[2] = self.stationary_variance
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Exact Fractional Path Simulation via Cholesky Decomposition
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "cholesky_euler",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of Fractional OU using exact fBM covariance and Euler integration.
        """
        if rng is None:
            rng = np.random.default_rng()

        if t_grid is None:
            t_final = float(T if T is not None else (t_max if t_max is not None else 1.0))
            steps = int(n_steps if n_steps is not None else (num_steps if num_steps is not None else 200))
            times = np.linspace(0.0, t_final, steps + 1)
        elif isinstance(t_grid, (int, float)):
            steps = int(n_steps if n_steps is not None else (num_steps if num_steps is not None else 200))
            times = np.linspace(0.0, float(t_grid), steps + 1)
        else:
            times = np.asarray(t_grid, dtype=np.float64)

        paths_count = int(n_paths if n_paths is not None else (num_paths if num_paths is not None else 500))
        n_pts = len(times)

        # Build fractional Brownian motion covariance matrix:
        # R_H(t, s) = 0.5 * (t^{2H} + s^{2H} - |t - s|^{2H})
        h2 = 2.0 * self.hurst
        t_col = times[:, None]
        t_row = times[None, :]

        cov = 0.5 * (np.power(t_col, h2) + np.power(t_row, h2) - np.power(np.abs(t_col - t_row), h2))
        cov[0, :] = 0.0
        cov[:, 0] = 0.0

        # Regularize covariance matrix for numerical stability
        cov += np.eye(n_pts) * 1e-11
        chol = np.linalg.cholesky(cov)

        # Sample fBM paths: shape (paths_count, n_pts)
        white_noise = rng.standard_normal((paths_count, n_pts))
        fbm_paths = white_noise @ chol.T
        fbm_inc = np.diff(fbm_paths, axis=1)

        # Integrate fOU: X_{i+1} = X_i - θ (X_i - μ) dt + σ ΔB_i^H
        x_paths = np.empty((paths_count, n_pts), dtype=np.float64)
        x_paths[:, 0] = self.x0

        for i in range(n_pts - 1):
            dt = times[i + 1] - times[i]
            x_paths[:, i + 1] = (
                x_paths[:, i]
                - self.theta * (x_paths[:, i] - self.mu) * dt
                + self.sigma * fbm_inc[:, i]
            )

        return x_paths


# Aliases
FractionalOU = FractionalOrnsteinUhlenbeck
FractionalOUProcess = FractionalOrnsteinUhlenbeck
