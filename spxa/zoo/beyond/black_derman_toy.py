r"""
Black–Derman–Toy (BDT) Short Rate Process (Black, Derman & Toy, 1990).

A lognormal short-rate model ensuring strictly positive interest rates r_t > 0:

    d(ln r_t) = [ θ(t) + (σ'(t)/σ(t)) ln r_t ] dt + σ(t) dW_t

In the standard constant-volatility case σ(t) = σ:
    d(ln r_t) = θ(t) dt + σ dW_t

where θ(t) is calibrated to match the initial term structure of interest rates.

Theoretical Properties
----------------------
1. Strict Positivity:
   Because ln r_t is normally distributed, r_t = exp(ln r_t) is lognormally distributed
   and remains strictly positive almost surely for all t > 0.

2. Exact Lognormal Moments:
   At time t:
       ln r_t ~ Normal(ln r_0 + ∫_0^t θ(s) ds, σ² t)
       E[r_t] = exp(ln r_0 + ∫_0^t θ(s) ds + (1/2) σ² t)

Exactness Level: MOMENT_PROPAGATION.

References
----------
Black, F., Derman, E., & Toy, W. (1990). A one-factor model of interest rates
and its application to Treasury bond options. *Financial Analysts Journal*, 46(1), 33–39.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Union

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class BlackDermanToyProcess(Process):
    r"""
    Black–Derman–Toy (BDT) lognormal short rate model.

    Parameters
    ----------
    sigma : float
        Volatility parameter σ > 0.
    r0 : float, optional
        Initial short rate r_0 > 0 (defaults to 0.03).
    theta_fn : callable, optional
        Time-dependent drift θ(t). Defaults to constant theta_0.
    theta_0 : float, optional
        Baseline drift (defaults to 0.0).
    """

    def __init__(
        self,
        sigma: float = 0.15,
        r0: float = 0.03,
        theta_fn: Optional[Callable[[float], float]] = None,
        theta_0: float = 0.0,
    ) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")
        if r0 <= 0:
            raise ValueError(f"r0 must be strictly positive for lognormal rates, got {r0}")

        self.sigma = float(sigma)
        self.r0 = float(r0)
        self.theta_0 = float(theta_0)
        self.theta_fn = theta_fn if theta_fn is not None else (lambda t: self.theta_0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"BlackDermanToy(σ={self.sigma}, r0={self.r0})",
            math_note=f"d(ln r_t) = θ(t) dt + σ dW_t; lognormal strictly positive rates",
            reference="Black, Derman & Toy (1990), Financial Analysts Journal 46(1), 33–39",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "BlackDermanToyProcess has log-rate dynamics and does not admit an "
            "independent Lévy–Khintchine triplet."
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
                "Black-Derman-Toy lognormal short rate model.",
                f"Instantaneous log-volatility σ = {self.sigma}.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.r0}
        if order >= 2:
            res[2] = (self.r0**2) * (math.exp(self.sigma**2) - 1.0)
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_grid: Optional[Union[np.ndarray, float]] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "exact_log",
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of the BDT short rate process.
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

        log_r = np.empty((paths_count, n_pts), dtype=np.float64)
        log_r[:, 0] = math.log(self.r0)

        for i in range(1, n_pts):
            dt = times[i] - times[i - 1]
            sqrt_dt = math.sqrt(dt)
            t_mid = 0.5 * (times[i - 1] + times[i])
            theta_val = self.theta_fn(t_mid)
            z = rng.standard_normal(paths_count)

            log_r[:, i] = log_r[:, i - 1] + theta_val * dt + self.sigma * sqrt_dt * z

        return np.exp(log_r)


# Aliases
BDTProcess = BlackDermanToyProcess
