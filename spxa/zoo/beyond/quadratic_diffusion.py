"""Quadratic Diffusion (Wong / Pearson Polynomial Diffusion) Process.

A diffusion with affine drift and quadratic variance in the Pearson diffusion family:
    dX(t) = (a + b * X(t)) dt + sqrt(c + d * X(t) + e * X(t)^2) dW(t)

Under stability condition b < 0 and 2*b + e < 0, the process possesses an exact stationary
moment hierarchy where polynomials are preserved under the infinitesimal generator:
    L(x^m) = lambda_m * x^m + lower order terms, with lambda_m = m*b + 0.5*m*(m-1)*e

References:
    Wong, E. (1964). The construction of a class of stationary Markoff processes.
        Proceedings of Symposia in Applied Mathematics, 16, 264-276.
    Forman, J. L., & Sørensen, M. (2008). The Pearson diffusions: A class of solvable diffusions
        for statistical modeling. Scandinavian Journal of Statistics, 35(3), 438-465.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties


class QuadraticDiffusion(Process):
    r"""Quadratic Pearson / Wong polynomial diffusion process.

    .. math::
        dX_t = (a + b X_t) dt + \sqrt{c + d X_t + e X_t^2} dW_t
    """

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        d: float,
        e: float,
        x0: float = 0.0,
    ) -> None:
        if b >= 0:
            raise ValueError(f"Stability requires mean-reversion drift slope b < 0, got {b}")
        if e < 0:
            raise ValueError(f"Quadratic variance coefficient e must be non-negative, got {e}")
        # Verify radicand positive at initial point x0
        rad_0 = c + d * x0 + e * (x0**2)
        if rad_0 <= 0:
            raise ValueError(f"Variance polynomial c + d*x0 + e*x0^2 must be positive at x0, got {rad_0}")

        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.e = float(e)
        self.x0 = float(x0)
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _triplet(self):
        raise ExactnessError(
            "QuadraticDiffusion has state-dependent quadratic volatility and cannot be represented by a Levy triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=(2.0 * self.b + self.e < 0.0),
            has_finite_mean=(self.b < 0.0),
            notes=["Pearson quadratic polynomial diffusion (Wong 1964)"],
        )

    def variance_radicand(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quadratic variance polynomial c + d*x + e*x^2."""
        return self.c + self.d * x + self.e * (x**2)

    def mean(self, t: float) -> float:
        """Conditional expectation E[X(t) | X(0)]."""
        if t <= 0:
            return self.x0
        eq = -self.a / self.b
        return eq + (self.x0 - eq) * math.exp(self.b * t)

    def stationary_mean(self) -> float:
        """Stationary unconditional mean mu_inf = -a / b."""
        return -self.a / self.b

    def stationary_second_moment(self) -> float:
        """Stationary second moment E[X_inf^2] (exists if 2*b + e < 0)."""
        lam2 = 2.0 * self.b + self.e
        if lam2 >= 0:
            return float("inf")
        mu = self.stationary_mean()
        num = 2.0 * self.a * mu + self.d * mu + self.c
        return -num / lam2

    def stationary_variance(self) -> float:
        """Stationary unconditional variance Var[X_inf]."""
        e_x2 = self.stationary_second_moment()
        if math.isinf(e_x2):
            return float("inf")
        mu = self.stationary_mean()
        return float(max(0.0, e_x2 - mu**2))

    def variance(self, t: float) -> float:
        """Conditional variance Var[X(t) | X(0)]."""
        if t <= 0:
            return 0.0
        lam2 = 2.0 * self.b + self.e
        if lam2 >= 0:
            return float("inf")
        # Solve linear system for E[X_t^2]
        mu_inf = self.stationary_mean()
        var_inf = self.stationary_variance()
        e_lam2t = math.exp(lam2 * t)
        return float(var_inf * (1.0 - e_lam2t))

    def simulate(
        self,
        t_max: float,
        num_steps: int,
        num_paths: int = 1,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate sample paths using Milstein discretization."""
        if seed is not None:
            np.random.seed(seed)

        dt = float(t_max) / num_steps
        sqrt_dt = math.sqrt(dt)

        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.x0

        a, b = self.a, self.b
        c, d, e = self.c, self.d, self.e

        for step in range(num_steps):
            x = paths[:, step]
            rad = np.maximum(1e-8, c + d * x + e * (x**2))
            vol = np.sqrt(rad)

            drift = a + b * x
            z = np.random.normal(0.0, 1.0, num_paths)

            # Milstein term: 0.5 * sigma * sigma' * (Z^2 - 1) * dt = 0.25 * (d + 2*e*x) * (Z^2 - 1) * dt
            milstein = 0.25 * (d + 2.0 * e * x) * (z**2 - 1.0) * dt

            paths[:, step + 1] = x + drift * dt + vol * sqrt_dt * z + milstein

        return paths


# Aliases
QuadraticPearsonDiffusion = QuadraticDiffusion
WongDiffusion = QuadraticDiffusion
