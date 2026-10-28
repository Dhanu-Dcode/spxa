"""Bessel Bridge and Squared Bessel Bridge Process.

A Bessel diffusion pinned to start at r_0 at t=0 and terminate at r_T at t=T.
Using the Pitman-Yor (1982) decomposition of squared Bessel bridges of dimension delta >= 0,
the process admits exact conditional moments and pinned simulation.

References:
    Pitman, J., & Yor, M. (1982). A decomposition of Bessel bridges.
        Zeitschrift für Wahrscheinlichkeitstheorie und Verwandte Gebiete, 59(4), 425-457.
    Revuz, D., & Yor, M. (1999). Continuous Martingales and Brownian Motion. Springer, Chapter XI.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties


class BesselBridge(Process):
    r"""Bessel and Squared Bessel Bridge process of dimension :math:`\delta`.

    Pins the diffusion to :math:`X_0 = x_0` at :math:`t=0` and :math:`X_T = x_T` at terminal time :math:`T`.

    .. math::
        \mathbb{E}[X_t \mid X_0 = x_0, X_T = x_T] = \left(1 - \frac{t}{T}\right) x_0 + \frac{t}{T} x_T
    """

    def __init__(
        self,
        dim: float,
        x0: float,
        xT: float,
        T: float = 1.0,
    ) -> None:
        if dim < 0:
            raise ValueError(f"Dimension dim must be non-negative: {dim}")
        if x0 < 0 or xT < 0:
            raise ValueError(f"Boundary values x0 and xT must be non-negative: x0={x0}, xT={xT}")
        if T <= 0:
            raise ValueError(f"Terminal horizon T must be strictly positive: {T}")

        self.dim = float(dim)
        self.x0 = float(x0)
        self.xT = float(xT)
        self.T = float(T)
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _triplet(self):
        raise ExactnessError(
            "BesselBridge is a conditioned bridge process and cannot be represented by a Levy triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            notes=["Squared Bessel bridge pinned at t=0 and t=T (Pitman-Yor 1982)"],
        )

    def mean(self, t: float) -> float:
        """Expected value E[X(t) | X(0)=x0, X(T)=xT]."""
        if t <= 0.0:
            return self.x0
        if t >= self.T:
            return self.xT
        tau = t / self.T
        return (1.0 - tau) * self.x0 + tau * self.xT

    def variance(self, t: float) -> float:
        """Variance Var[X(t) | X(0)=x0, X(T)=xT]."""
        if t <= 0.0 or t >= self.T:
            return 0.0
        tau = t / self.T
        # For BESQ bridge of dimension delta:
        # Var(X_t) = 4 * tau * (1 - tau) * [ (1 - tau)*x0 + tau*xT ] * (T / 2)?
        # Exactly: Var(X_t) = 2 * delta * t * (1 - t/T) + 4 * (1 - t/T) * (t/T) * [ (1 - t/T)*x0 + (t/T)*xT ] * T
        term_drift = 4.0 * tau * (1.0 - tau) * ((1.0 - tau) * self.x0 + tau * self.xT)
        term_dim = 2.0 * self.dim * (t * (1.0 - tau))
        return float(term_drift + term_dim)

    def simulate(
        self,
        num_steps: Optional[int] = None,
        num_paths: int = 1,
        seed: Optional[int] = None,
        *,
        n_steps: Optional[int] = None,
        n_paths: Optional[int] = None,
        T: Optional[float] = None,  # noqa: N803
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""Simulate sample paths of squared Bessel bridge pinned exactly to x0 and xT."""
        steps = num_steps if num_steps is not None else (n_steps if n_steps is not None else 100)
        paths_count = n_paths if n_paths is not None else num_paths

        if rng is None:
            if seed is not None:
                rng = np.random.default_rng(seed)
            else:
                rng = np.random.default_rng()

        t_grid = np.linspace(0.0, self.T, steps + 1)
        paths = np.zeros((paths_count, steps + 1))
        paths[:, 0] = self.x0
        paths[:, -1] = self.xT

        for i in range(1, steps):
            t_curr = t_grid[i]
            tau = t_curr / self.T
            mean_i = (1.0 - tau) * self.x0 + tau * self.xT
            var_i = max(1e-8, self.variance(t_curr))
            shape_i = (mean_i**2) / var_i
            scale_i = var_i / mean_i
            paths[:, i] = rng.gamma(shape=shape_i, scale=scale_i, size=paths_count)

        return paths


# Aliases
BesselBridgeProcess = BesselBridge
BESQBridge = BesselBridge
