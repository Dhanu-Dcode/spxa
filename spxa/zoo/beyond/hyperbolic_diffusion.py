"""Hyperbolic Diffusion Process.

A continuous-time diffusion possessing an exact Hyperbolic invariant distribution:
    dX(t) = mu(X(t)) dt + sigma * dW(t)
    mu(x) = 0.5 * sigma^2 * (beta - alpha * (x - mu_0) / sqrt(delta^2 + (x - mu_0)^2))

Unlike the linear drift of Ornstein-Uhlenbeck (which forces Gaussian tails), the drift of the
hyperbolic diffusion approaches constant slopes as |x| -> inf, generating exact exponential
tails P(X > x) ~ exp(-(alpha - beta)*x) without jumps.

References:
    Barndorff-Nielsen, O. E. (1978). Hyperbolic distributions and distributions on hyperbolae.
        Scandinavian Journal of Statistics, 5(3), 151-157.
    Bibby, B. M., & Sørensen, M. (2003). Hyperbolic processes in finance.
        In Handbook of Heavy Tailed Distributions in Finance, 211-248.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy import special

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties


class HyperbolicDiffusion(Process):
    r"""Hyperbolic invariant diffusion process with exponential tails.

    .. math::
        dX_t = \frac{1}{2}\sigma^2 \left( \beta - \alpha \frac{X_t - \mu_0}{\sqrt{\delta^2 + (X_t - \mu_0)^2}} \right) dt + \sigma dW_t
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        delta: float,
        mu0: float = 0.0,
        sigma: float = 1.0,
        x0: float = 0.0,
    ) -> None:
        if alpha <= 0 or delta <= 0:
            raise ValueError(f"Parameters alpha and delta must be positive: alpha={alpha}, delta={delta}")
        if abs(beta) >= alpha:
            raise ValueError(f"Stability requires |beta| < alpha: |{beta}| >= {alpha}")
        if sigma <= 0:
            raise ValueError(f"Diffusion volatility sigma must be positive: {sigma}")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.delta = float(delta)
        self.mu0 = float(mu0)
        self.sigma = float(sigma)
        self.x0 = float(x0)
        self.gamma = math.sqrt(self.alpha**2 - self.beta**2)
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _triplet(self):
        raise ExactnessError(
            "HyperbolicDiffusion has nonlinear bounded drift and cannot be represented by a Levy triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            notes=["Hyperbolic diffusion with invariant exponential tails (Barndorff-Nielsen 1978)"],
        )

    def drift(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Drift coefficient mu(x)."""
        diff = x - self.mu0
        hyp = np.sqrt(self.delta**2 + diff**2)
        return 0.5 * (self.sigma**2) * (self.beta - self.alpha * diff / hyp)

    def stationary_pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Exact stationary probability density of the Hyperbolic distribution."""
        diff = x - self.mu0
        hyp = np.sqrt(self.delta**2 + diff**2)
        exponent = -self.alpha * hyp + self.beta * diff

        # Normalization factor C = gamma / (2 * alpha * delta * K_1(delta * gamma))
        dg = self.delta * self.gamma
        k1 = special.kv(1, dg)
        c = self.gamma / (2.0 * self.alpha * self.delta * k1)
        return c * np.exp(exponent)

    def stationary_mean(self) -> float:
        """Exact stationary mean using modified Bessel functions."""
        dg = self.delta * self.gamma
        k1 = special.kv(1, dg)
        k2 = special.kv(2, dg)
        return float(self.mu0 + (self.delta * self.beta / self.gamma) * (k2 / k1))

    def stationary_variance(self) -> float:
        """Exact stationary variance using modified Bessel functions."""
        dg = self.delta * self.gamma
        k1 = special.kv(1, dg)
        k2 = special.kv(2, dg)
        k3 = special.kv(3, dg)
        term1 = (self.delta / self.gamma) * (k2 / k1)
        term2 = ((self.delta * self.beta / self.gamma) ** 2) * ((k3 / k1) - (k2 / k1) ** 2)
        return float(term1 + term2)

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
        sig = self.sigma

        for step in range(num_steps):
            x_curr = paths[:, step]
            mu_curr = self.drift(x_curr)
            z = np.random.normal(0.0, 1.0, num_paths)
            paths[:, step + 1] = x_curr + mu_curr * dt + sig * sqrt_dt * z

        return paths


# Aliases
HyperbolicOU = HyperbolicDiffusion
HyperbolicInvariantDiffusion = HyperbolicDiffusion
