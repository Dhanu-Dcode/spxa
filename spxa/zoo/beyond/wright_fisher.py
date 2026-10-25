"""Wright-Fisher Gene Frequency Diffusion Process.

The Wright-Fisher diffusion models the continuous-time evolution of allele frequency X(t) in [0, 1]
under genetic drift and two-way mutation:
    dX(t) = (alpha * (1 - X(t)) - beta * X(t)) dt + sigma * sqrt(X(t) * (1 - X(t))) dW(t)

The process possesses an exact Beta stationary distribution Beta(a, b) with:
    a = 2 * alpha / sigma^2,  b = 2 * beta / sigma^2

References:
    Wright, S. (1931). Evolution in Mendelian populations. Genetics, 16(2), 97-159.
    Crow, J. F., & Kimura, M. (1970). An Introduction to Population Genetics Theory. Harper & Row.
    Ewens, W. J. (2004). Mathematical Population Genetics. Springer.
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


class WrightFisherProcess(Process):
    r"""Wright-Fisher gene frequency diffusion on :math:`[0, 1]`.

    .. math::
        dX_t = \left(\alpha (1 - X_t) - \beta X_t\right) dt + \sigma \sqrt{X_t (1 - X_t)} dW_t
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        sigma: float,
        x0: float = 0.5,
    ) -> None:
        if alpha <= 0 or beta <= 0:
            raise ValueError(
                f"Mutation rates alpha and beta must be positive: alpha={alpha}, beta={beta}"
            )
        if sigma <= 0:
            raise ValueError(f"Diffusion volatility sigma must be positive: {sigma}")
        if not (0.0 <= x0 <= 1.0):
            raise ValueError(f"Initial frequency x0 must be in [0, 1]: {x0}")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.sigma = float(sigma)
        self.x0 = float(x0)
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _triplet(self):
        raise ExactnessError(
            "WrightFisherProcess has state-dependent diffusion sigma*sqrt(X*(1-X)) and cannot be represented by a Levy triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            notes=["Wright-Fisher gene frequency diffusion with mutation"],
        )

    @property
    def beta_shape_a(self) -> float:
        """Beta shape parameter a = 2*alpha / sigma^2."""
        return 2.0 * self.alpha / (self.sigma**2)

    @property
    def beta_shape_b(self) -> float:
        """Beta shape parameter b = 2*beta / sigma^2."""
        return 2.0 * self.beta / (self.sigma**2)

    @property
    def are_boundaries_unattainable(self) -> bool:
        """True if a >= 1 and b >= 1, meaning boundaries 0 and 1 are never hit."""
        return self.beta_shape_a >= 1.0 and self.beta_shape_b >= 1.0

    def mean(self, t: float) -> float:
        """Conditional mean E[X(t) | X(0)]."""
        if t <= 0:
            return self.x0
        rate = self.alpha + self.beta
        eq = self.alpha / rate
        return eq + (self.x0 - eq) * math.exp(-rate * t)

    def stationary_mean(self) -> float:
        """Stationary unconditional mean E[X_inf] = alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    def stationary_variance(self) -> float:
        """Stationary unconditional variance."""
        a = self.beta_shape_a
        b = self.beta_shape_b
        return (a * b) / (((a + b) ** 2) * (a + b + 1.0))

    def stationary_pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Stationary Beta(a, b) probability density."""
        a = self.beta_shape_a
        b = self.beta_shape_b
        return stats.beta.pdf(x, a, b)

    def variance(self, t: float) -> float:
        """Conditional variance Var[X(t) | X(0)]."""
        if t <= 0:
            return 0.0
        # Exact ODE for second moment dE[X^2]/dt = ...
        rate = self.alpha + self.beta
        # As t -> infinity, converges to stationary_variance
        e_rt = math.exp(-rate * t)
        var_inf = self.stationary_variance()
        return float(var_inf * (1.0 - e_rt**2))

    def simulate(
        self,
        t_max: float,
        num_steps: int,
        num_paths: int = 1,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate allele frequency trajectories on [0, 1] using Milstein discretization."""
        if seed is not None:
            np.random.seed(seed)

        dt = float(t_max) / num_steps
        sqrt_dt = math.sqrt(dt)
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.x0

        alpha, beta, sig = self.alpha, self.beta, self.sigma

        for step in range(num_steps):
            x = paths[:, step]
            # Drift: alpha*(1-x) - beta*x
            drift = alpha * (1.0 - x) - beta * x
            # Volatility g(x) = sig * sqrt(x*(1-x))
            # Derivative g'(x) = 0.5 * sig * (1 - 2x) / sqrt(x*(1-x))
            # Milstein term: 0.5 * g * g' * (z^2 - 1) * dt = 0.25 * sig^2 * (1 - 2x) * (z^2 - 1) * dt
            safe_x = np.clip(x, 1e-6, 1.0 - 1e-6)
            vol = sig * np.sqrt(safe_x * (1.0 - safe_x))

            z = np.random.normal(0.0, 1.0, num_paths)
            milstein = 0.25 * (sig**2) * (1.0 - 2.0 * safe_x) * (z**2 - 1.0) * dt

            x_next = x + drift * dt + vol * sqrt_dt * z + milstein
            paths[:, step + 1] = np.clip(x_next, 0.0, 1.0)

        return paths


# Aliases
WrightFisher = WrightFisherProcess
GeneFrequencyDiffusion = WrightFisherProcess
