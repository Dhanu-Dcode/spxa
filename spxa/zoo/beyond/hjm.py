r"""
Heath–Jarrow–Morton (HJM) Forward Rate Field (Heath, Jarrow & Morton, 1992).

Models the entire continuum of instantaneous forward rates f(t, T) across all
maturities T ≥ t under the risk-neutral measure Q:

    df(t, T) = α(t, T) dt + σ(t, T) dW_t

The Fundamental HJM No-Arbitrage Drift Condition enforces:
    α(t, T) = σ(t, T) ∫_t^T σ(t, u) du

For exponential volatility σ(t, T) = σ exp(-λ (T - t)):
    α(t, T) = (σ² / λ) exp(-λ(T - t)) [ 1 - exp(-λ(T - t)) ]

Theoretical Properties
----------------------
1. No-Arbitrage Consistency:
   The drift of the entire yield curve is uniquely and endogenously determined
   by the volatility structure σ(t, T), automatically precluding bond arbitrage.

2. Short Rate Dynamics:
   The instantaneous spot rate r_t = f(t, t) satisfies:
       r_t = f(0, t) + ∫_0^t α(s, t) ds + ∫_0^t σ(s, t) dW_s

Exactness Level: MOMENT_PROPAGATION.

References
----------
Heath, D., Jarrow, R., & Morton, A. (1992). Bond pricing and the term structure
of interest rates: A new methodology for contingent claims valuation.
*Econometrica*, 60(1), 77–105.
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


class HJMField(Process):
    r"""
    Heath–Jarrow–Morton (HJM) Forward Rate Field with exponential volatility.

    Parameters
    ----------
    sigma : float
        Volatility scale σ > 0.
    lam : float
        Mean-reversion decay rate λ > 0.
    f0 : float, optional
        Flat initial forward curve f(0, T) (defaults to 0.03).
    """

    def __init__(
        self,
        sigma: float = 0.01,
        lam: float = 0.1,
        f0: float = 0.03,
    ) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")
        if lam <= 0:
            raise ValueError(f"lam must be strictly positive, got {lam}")

        self.sigma = float(sigma)
        self.lam = float(lam)
        self.f0 = float(f0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"HJMField(σ={self.sigma}, λ={self.lam}, f0={self.f0})",
            math_note="df(t,T) = α(t,T)dt + σ e^{-λ(T-t)} dW_t with no-arbitrage drift α",
            reference="Heath, Jarrow & Morton (1992), Econometrica 60(1), 77–105",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # No-Arbitrage Drift & Volatility
    # ------------------------------------------------------------------

    def volatility(self, t: float, T: float) -> float:
        r"""HJM forward rate volatility σ(t, T) = σ exp(-λ(T - t))."""
        if T < t:
            return 0.0
        return self.sigma * math.exp(-self.lam * (T - t))

    def no_arbitrage_drift(self, t: float, T: float) -> float:
        r"""
        Exact no-arbitrage drift:
        α(t, T) = (σ² / λ) exp(-λ(T - t)) [ 1 - exp(-λ(T - t)) ].
        """
        if T < t:
            return 0.0
        e_tau = math.exp(-self.lam * (T - t))
        return ((self.sigma**2) / self.lam) * e_tau * (1.0 - e_tau)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "HJMField is an infinite-dimensional forward rate field and does not "
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
                "Heath-Jarrow-Morton arbitrage-free forward rate curve field.",
                f"Volatility scale σ = {self.sigma}, Decay rate λ = {self.lam}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.f0}
        if order >= 2:
            res[2] = self.sigma**2
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Forward Curve Simulation on a Tenor Grid
    # ------------------------------------------------------------------

    def simulate_curve(
        self,
        t_eval: float,
        tenors: np.ndarray,
        n_paths: int = 500,
        n_steps: int = 50,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""
        Simulate the forward rate curve f(t_eval, T) for a vector of maturities T ∈ tenors.

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, len(tenors)) representing the simulated curve at t_eval.
        """
        if rng is None:
            rng = np.random.default_rng()

        times = np.linspace(0.0, t_eval, n_steps + 1)
        m_tenors = len(tenors)
        f_curve = np.full((n_paths, m_tenors), self.f0, dtype=np.float64)

        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)
            t_mid = 0.5 * (times[i - 1] + times[i])
            z = rng.standard_normal(n_paths)[:, None]

            # Vectorized over all tenors T
            taus = np.maximum(0.0, tenors - t_mid)
            e_taus = np.exp(-self.lam * taus)
            drifts = ((self.sigma**2) / self.lam) * e_taus * (1.0 - e_taus)
            vols = self.sigma * e_taus

            f_curve += drifts * dt + vols * sqrt_dt * z

        return f_curve

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "short_rate",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of the instantaneous short rate r_t = f(t, t).
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

        r_paths = np.empty((paths_count, n_pts), dtype=np.float64)
        r_paths[:, 0] = self.f0

        # Exact Cheyette representation for r_t = f(0, t) + x_t + y_t
        x_curr = np.zeros(paths_count, dtype=np.float64)
        y_curr = 0.0

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)
            z = rng.standard_normal(paths_count)

            # dx = (y - λ x) dt + σ dW
            # dy = (σ² - 2λ y) dt
            x_curr = x_curr + (y_curr - self.lam * x_curr) * dt + self.sigma * sqrt_dt * z
            y_curr = y_curr + (self.sigma**2 - 2.0 * self.lam * y_curr) * dt

            r_paths[:, i] = self.f0 + x_curr

        return r_paths


# Aliases
HJMProcess = HJMField
