"""Stochastic Logistic (Verhulst) Growth Diffusion Process.

The Verhulst logistic growth model with multiplicative environmental noise:
    dX(t) = r * X(t) * (1 - X(t) / K) dt + sigma * X(t) dW(t)

When r > 0.5 * sigma^2, the process possesses an exact Gamma stationary distribution:
    X_inf ~ Gamma(shape=alpha, scale=1/beta)
    alpha = 2 * r / sigma^2 - 1,  beta = 2 * r / (sigma^2 * K)
with expected equilibrium:
    E[X_inf] = K * (1 - 0.5 * sigma^2 / r)

References:
    Verhulst, P. F. (1838). Notice sur la loi que la population suit dans son accroissement.
        Corr. Math. Phys., 10, 113-121.
    Goel, N. S., & Richter-Dyn, N. (1974). Stochastic Models in Biology. Academic Press.
    Dennis, B., & Patil, G. P. (1984). The gamma distribution and weighted multimodal distributions
        as models of population abundance. Mathematical Biosciences, 68(2), 187-212.
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


class LogisticDiffusion(Process):
    r"""Stochastic Logistic (Verhulst) population growth diffusion process.

    .. math::
        dX_t = r X_t \left(1 - \frac{X_t}{K}\right) dt + \sigma X_t dW_t
    """

    def __init__(
        self,
        r: float,
        k: float,
        sigma: float,
        x0: float = 1.0,
    ) -> None:
        if r <= 0:
            raise ValueError(f"Growth rate r must be strictly positive: {r}")
        if k <= 0:
            raise ValueError(f"Carrying capacity k must be strictly positive: {k}")
        if sigma <= 0:
            raise ValueError(f"Environmental volatility sigma must be strictly positive: {sigma}")
        if x0 <= 0:
            raise ValueError(f"Initial population x0 must be strictly positive: {x0}")

        self.r = float(r)
        self.k = float(k)
        self.sigma = float(sigma)
        self.x0 = float(x0)
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _triplet(self):
        raise ExactnessError(
            "LogisticDiffusion has non-linear drift r*X*(1 - X/K) and cannot be represented by a Levy triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            notes=["Verhulst stochastic logistic growth diffusion with Gamma stationary law"],
        )

    @property
    def is_persistent(self) -> bool:
        """True if r > 0.5 * sigma^2, guaranteeing a non-trivial stationary distribution."""
        return self.r > 0.5 * (self.sigma**2)

    @property
    def gamma_shape(self) -> float:
        """Shape parameter alpha = 2*r / sigma^2 - 1 for the stationary Gamma law."""
        return 2.0 * self.r / (self.sigma**2) - 1.0

    @property
    def gamma_rate(self) -> float:
        """Rate parameter beta = 2*r / (sigma^2 * K) for the stationary Gamma law."""
        return 2.0 * self.r / ((self.sigma**2) * self.k)

    def stationary_mean(self) -> float:
        """Stationary expected population E[X_inf] = K * (1 - 0.5 * sigma^2 / r)."""
        if not self.is_persistent:
            return 0.0
        return self.k * (1.0 - 0.5 * (self.sigma**2) / self.r)

    def stationary_variance(self) -> float:
        """Stationary population variance Var[X_inf]."""
        if not self.is_persistent:
            return 0.0
        alpha = self.gamma_shape
        beta = self.gamma_rate
        return alpha / (beta**2)

    def stationary_pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Stationary probability density (Gamma distribution)."""
        if not self.is_persistent:
            return 0.0
        alpha = self.gamma_shape
        scale = 1.0 / self.gamma_rate
        return stats.gamma.pdf(x, a=alpha, scale=scale)

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
        """Simulate positive population paths using log-transformed predictor-corrector Euler scheme."""
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

        r, k, sig = self.r, self.k, self.sigma

        for step in range(steps):
            x_curr = np.exp(log_x)
            # Drift of log(X_t): r - 0.5*sig^2 - (r/K)*X_t
            drift = r - 0.5 * (sig**2) - (r / k) * x_curr
            z = np.random.normal(0.0, 1.0, paths_count)

            log_x = log_x + drift * dt + sig * sqrt_dt * z
            paths[:, step + 1] = np.exp(log_x)

        return paths


# Aliases
LogisticGrowthProcess = LogisticDiffusion
VerhulstDiffusion = LogisticDiffusion
