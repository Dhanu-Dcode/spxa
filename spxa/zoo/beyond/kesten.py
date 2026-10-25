"""Kesten Multiplicative Noise Diffusion Process.

The continuous-time limit of the Kesten (1973) stochastic recurrence with multiplicative noise:
    dX(t) = (a - b * X(t)) dt + sigma * X(t) dW(t)

The process produces endogenous heavy-tailed power-law behavior. The exact stationary distribution
is an Inverse-Gamma law with tail exponent alpha = 2*b / sigma^2 + 1 and scale beta = 2*a / sigma^2:
    P(X > x) ~ x^(-alpha)

References:
    Kesten, H. (1973). Random difference equations and renewal theory for products of random matrices.
        Acta Mathematica, 131(1), 207-248.
    Sornette, D., & Cont, R. (1997). Convergent multiplicative processes repelled from zero:
        Power laws without fine-tuning. Journal de Physique I, 7(3), 431-444.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy import stats

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties


class KestenProcess(Process):
    r"""Kesten multiplicative noise diffusion process with Pareto power-law tails.

    .. math::
        dX_t = (a - b X_t) dt + \sigma X_t dW_t
    """

    def __init__(
        self,
        a: float,
        b: float,
        sigma: float,
        x0: float = 1.0,
    ) -> None:
        if a <= 0:
            raise ValueError(f"Additive drift parameter a must be strictly positive: {a}")
        if b <= 0:
            raise ValueError(f"Mean-reversion parameter b must be strictly positive: {b}")
        if sigma <= 0:
            raise ValueError(f"Multiplicative noise sigma must be strictly positive: {sigma}")
        if x0 <= 0:
            raise ValueError(f"Initial state x0 must be strictly positive: {x0}")

        self.a = float(a)
        self.b = float(b)
        self.sigma = float(sigma)
        self.x0 = float(x0)
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _triplet(self):
        raise ExactnessError(
            "KestenProcess has state-dependent multiplicative noise sigma*X_t and cannot be represented by a Levy triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=(2.0 * self.b > self.sigma**2),
            has_finite_mean=(self.b > 0.0),
            tail_index=self.pareto_tail_index,
            notes=["Kesten multiplicative noise diffusion with Inverse-Gamma stationary law"],
        )

    @property
    def pareto_tail_index(self) -> float:
        r"""Exact Pareto stationary tail exponent \alpha = 2*b / \sigma^2 + 1."""
        return 2.0 * self.b / (self.sigma**2) + 1.0

    @property
    def invgamma_scale(self) -> float:
        r"""Inverse-Gamma scale parameter \beta = 2*a / \sigma^2."""
        return 2.0 * self.a / (self.sigma**2)

    def mean(self, t: float) -> float:
        """Conditional expectation E[X(t) | X(0)]."""
        if t <= 0:
            return self.x0
        eq = self.a / self.b
        return eq + (self.x0 - eq) * math.exp(-self.b * t)

    def stationary_mean(self) -> float:
        """Stationary unconditional mean E[X_inf] = a / b."""
        return self.a / self.b

    def stationary_variance(self) -> float:
        """Stationary unconditional variance (exists when 2*b > sigma^2)."""
        denom = 2.0 * self.b - self.sigma**2
        if denom <= 0:
            return float("inf")
        return (self.a**2 * self.sigma**2) / (self.b**2 * denom)

    def stationary_pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Stationary Inverse-Gamma probability density."""
        alpha = self.pareto_tail_index
        beta = self.invgamma_scale
        return stats.invgamma.pdf(x, a=alpha, scale=beta)

    def simulate(
        self,
        t_max: Optional[float] = None,
        num_steps: Optional[int] = None,
        num_paths: int = 1,
        seed: Optional[int] = None,
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        n_paths: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate strictly positive paths using log-Euler / Milstein discretization."""
        if seed is not None:
            np.random.seed(seed)

        final_t = t_max if t_max is not None else (T if T is not None else 1.0)
        steps = num_steps if num_steps is not None else (n_steps if n_steps is not None else 100)
        paths_count = n_paths if n_paths is not None else num_paths

        dt = float(final_t) / steps
        sqrt_dt = math.sqrt(dt)

        paths = np.zeros((paths_count, steps + 1))
        paths[:, 0] = self.x0
        log_x = np.full(paths_count, math.log(self.x0))

        a, b, sig = self.a, self.b, self.sigma

        for step in range(steps):
            x_curr = np.exp(log_x)
            # Drift of ln(X_t): a/X_t - b - 0.5*sigma^2
            drift = (a / x_curr) - b - 0.5 * (sig**2)
            z = np.random.normal(0.0, 1.0, paths_count)

            log_x = log_x + drift * dt + sig * sqrt_dt * z
            paths[:, step + 1] = np.exp(log_x)

        return paths


# Aliases
KestenDiffusion = KestenProcess
MultiplicativeNoiseDiffusion = KestenProcess
