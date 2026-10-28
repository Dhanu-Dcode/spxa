r"""
Multifractal Random Walk (MRW) process (Bacry, Delour & Muzy, 2001).

The Multifractal Random Walk is a continuous-time stochastic process with
non-Gaussian heavy-tailed increments, long-range volatility correlations,
and exact multifractal scaling:

    X_t = ∫_0^t e^{ω(s)} dW_s

where ω(t) is a Gaussian log-correlated cascade field satisfying:
    Cov(ω(t), ω(s)) = λ² ln(T / (|t - s| + ε))   for |t - s| ≤ T
    Cov(ω(t), ω(s)) = 0                           for |t - s| > T

Theoretical Properties
----------------------
1. Concave Multifractal Spectrum:
   The structure function exhibits non-linear power-law multiscaling:
       E[|X_t|^q] = C_q t^{ζ(q)}
   with strictly parabolic spectrum:
       ζ(q) = q (1/2 + λ²) - (λ² q² / 2)
   In contrast to Brownian motion (where ζ(q) = q/2 is linear), MRW exhibits
   genuine multifractal intermittency.

2. Heavy Tails and Intermittency:
   The parameter λ² is the intermittency coefficient. As λ² increases, returns
   exhibit increasingly heavy tails and severe volatility clustering.

Exactness Level: MOMENT_PROPAGATION.

References
----------
Bacry, E., Delour, J., & Muzy, J. F. (2001). Multifractal random walk.
*Physical Review E*, 64(2), 026103.
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


class MultifractalRandomWalk(Process):
    r"""
    Multifractal Random Walk (MRW) continuous cascade process.

    Parameters
    ----------
    lambda_sq : float
        Intermittency parameter λ² > 0 (typically 0.02 - 0.08 in equities).
    integral_scale : float
        Macro-scale correlation horizon T > 0 (e.g. 1.0 year).
    sigma : float
        Baseline volatility scale σ > 0.
    x0 : float, optional
        Initial value X_0 (defaults to 0.0).
    """

    def __init__(
        self,
        lambda_sq: float = 0.04,
        integral_scale: float = 1.0,
        sigma: float = 0.2,
        x0: float = 0.0,
    ) -> None:
        if lambda_sq <= 0:
            raise ValueError(f"lambda_sq must be strictly positive, got {lambda_sq}")
        if integral_scale <= 0:
            raise ValueError(f"integral_scale must be strictly positive, got {integral_scale}")
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")

        self.lambda_sq = float(lambda_sq)
        self.integral_scale = float(integral_scale)
        self.sigma = float(sigma)
        self.x0 = float(x0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"MultifractalRandomWalk(λ²={self.lambda_sq}, T={self.integral_scale}, σ={self.sigma})",
            math_note="X_t = ∫ e^{ω(s)} dW_s with parabolic spectrum ζ(q)=q(1/2+λ²) - λ²q²/2",
            reference="Bacry, Delour & Muzy (2001), Phys. Rev. E 64(2), 026103",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Multifractal Scaling Spectrum
    # ------------------------------------------------------------------

    def multifractal_scaling_exponent(
        self, q: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        r"""
        Exact parabolic multifractal scaling exponent ζ(q) = q (1/2 + λ²) - (λ² q² / 2).
        """
        q_arr = np.asarray(q, dtype=np.float64)
        zeta = q_arr * (0.5 + self.lambda_sq) - (self.lambda_sq * (q_arr**2) / 2.0)
        if np.isscalar(q):
            return float(zeta)
        return zeta

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "MultifractalRandomWalk features continuous log-normal cascade memory and does not "
            "admit an independent Lévy–Khintchine triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=True,
            has_independent_increments=False,
            is_martingale=True,
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=math.sqrt(2.0 / self.lambda_sq) if self.lambda_sq > 0 else None,
            is_subordinator=False,
            hurst_index=0.5,
            notes=[
                "Multifractal Random Walk with log-normal cascade volatility field.",
                f"Intermittency coefficient λ² = {self.lambda_sq}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.x0}
        if order >= 2:
            res[2] = self.sigma**2
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Vectorized Simulation via Log-Correlated Gaussian Cascade
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "cascade_fft",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of the Multifractal Random Walk.
        """
        if rng is None:
            rng = np.random.default_rng()

        if t_grid is None:
            t_final = float(T if T is not None else (t_max if t_max is not None else 1.0))
            steps = int(
                n_steps if n_steps is not None else (num_steps if num_steps is not None else 256)
            )
            times = np.linspace(0.0, t_final, steps + 1)
        elif isinstance(t_grid, (int, float)):
            steps = int(
                n_steps if n_steps is not None else (num_steps if num_steps is not None else 256)
            )
            times = np.linspace(0.0, float(t_grid), steps + 1)
        else:
            times = np.asarray(t_grid, dtype=np.float64)

        paths_count = int(
            n_paths if n_paths is not None else (num_paths if num_paths is not None else 500)
        )
        n_pts = len(times)
        n_steps_actual = n_pts - 1

        dt = times[1] - times[0]

        # Logarithmic covariance function C(tau) = lambda_sq * ln(T / (tau + dt))
        lags = np.arange(n_steps_actual) * dt
        cov_row = np.zeros(n_steps_actual)
        valid = lags < self.integral_scale
        cov_row[valid] = self.lambda_sq * np.log(self.integral_scale / (lags[valid] + dt))
        cov_row[0] += self.lambda_sq  # Regularize diagonal

        # Toeplitz embedding for fast circulant embedding / Cholesky
        from scipy.linalg import toeplitz

        cov_mat = toeplitz(cov_row)
        cov_mat += np.eye(n_steps_actual) * 1e-9
        chol = np.linalg.cholesky(cov_mat)

        # Generate Gaussian log-volatility field ω: mean -λ² ln(T/dt) / 2 to normalize E[e^{2ω}] = 1
        omega_mean = -0.5 * cov_row[0]
        z_omega = rng.standard_normal((paths_count, n_steps_actual))
        omega = omega_mean + z_omega @ chol.T

        # Standard Brownian increments dW
        dw = rng.standard_normal((paths_count, n_steps_actual)) * math.sqrt(dt)

        # Subordinated increments: dX = σ e^{ω} dW
        dx = self.sigma * np.exp(omega) * dw

        x_paths = np.empty((paths_count, n_pts), dtype=np.float64)
        x_paths[:, 0] = self.x0
        x_paths[:, 1:] = self.x0 + np.cumsum(dx, axis=1)

        return x_paths


# Aliases
MRWProcess = MultifractalRandomWalk
