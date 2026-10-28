r"""Mean-Reverting Ornstein-Uhlenbeck Process with Compound Poisson Jumps (Jump-OU).

The Jump-OU model couples Gaussian mean-reverting drift with a compound Poisson jump process,
widely used in energy, electricity, and commodity markets to capture mean reversion and sudden spikes:
    dX(t) = k * (theta - X(t)) dt + sigma * dW(t) + dJ(t)
    J(t) = \sum_{i=1}^{N(t)} Y_i, with N(t) ~ Poisson(lambda_J * t), Y_i ~ Normal(mu_J, sigma_J^2)

References:
    Lucia, J. J., & Schwartz, E. S. (2002). Electricity prices and power derivatives:
        Evidence from the Nordic Power Exchange. Review of Derivatives Research, 5(1), 5-50.
    Benth, F. E., Benth, J. Š., & Koekebakker, S. (2008). Stochastic Modelling of Electricity
        and Related Markets. World Scientific.
"""

from __future__ import annotations

import cmath
import math
from typing import Optional, Union

import numpy as np
from scipy import integrate

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties


class JumpOUProcess(Process):
    r"""Mean-reverting Ornstein-Uhlenbeck process with compound Poisson jumps.

    .. math::
        dX_t = \kappa (\theta - X_t) dt + \sigma dW_t + dJ_t, \quad J_t = \sum_{i=1}^{N_t} Y_i
    """

    def __init__(
        self,
        k: float,
        theta: float,
        sigma: float,
        lambda_j: float,
        mu_j: float,
        sigma_j: float,
        x0: float = 0.0,
    ) -> None:
        if k <= 0:
            raise ValueError(f"Mean-reversion speed k must be strictly positive: {k}")
        if sigma <= 0:
            raise ValueError(f"Volatility sigma must be strictly positive: {sigma}")
        if lambda_j < 0:
            raise ValueError(f"Jump intensity lambda_j must be non-negative: {lambda_j}")
        if sigma_j < 0:
            raise ValueError(f"Jump size volatility sigma_j must be non-negative: {sigma_j}")

        self.k = float(k)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.lambda_j = float(lambda_j)
        self.mu_j = float(mu_j)
        self.sigma_j = float(sigma_j)
        self.x0 = float(x0)
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _triplet(self):
        raise ExactnessError(
            "JumpOUProcess has state-dependent drift k*(theta - X_t) and cannot be represented by a Levy triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            notes=["Ornstein-Uhlenbeck process with compound Poisson jumps (Jump-OU)"],
        )

    def mean(self, t: float) -> float:
        """Conditional expectation E[X(t) | X(0)]."""
        if t <= 0:
            return self.x0
        e_kt = math.exp(-self.k * t)
        ou_mean = self.x0 * e_kt + self.theta * (1.0 - e_kt)
        jump_mean = (self.lambda_j * self.mu_j / self.k) * (1.0 - e_kt)
        return ou_mean + jump_mean

    def variance(self, t: float) -> float:
        """Conditional variance Var[X(t) | X(0)]."""
        if t <= 0:
            return 0.0
        e_2kt = math.exp(-2.0 * self.k * t)
        var_diff = (self.sigma**2 / (2.0 * self.k)) * (1.0 - e_2kt)
        second_moment_y = self.sigma_j**2 + self.mu_j**2
        var_jumps = (self.lambda_j * second_moment_y / (2.0 * self.k)) * (1.0 - e_2kt)
        return var_diff + var_jumps

    def stationary_mean(self) -> float:
        """Long-run unconditional mean."""
        return self.theta + (self.lambda_j * self.mu_j / self.k)

    def stationary_variance(self) -> float:
        """Long-run unconditional variance."""
        second_moment_y = self.sigma_j**2 + self.mu_j**2
        return (self.sigma**2 + self.lambda_j * second_moment_y) / (2.0 * self.k)

    def characteristic_function(self, u: Union[float, complex], t: float) -> complex:
        r"""Characteristic function E[exp(i u X_t)]."""
        u_c = complex(u)
        i = 1j
        if t <= 0:
            return cmath.exp(i * u_c * self.x0)

        e_kt = math.exp(-self.k * t)
        mean_ou = self.x0 * e_kt + self.theta * (1.0 - e_kt)
        var_diff = (self.sigma**2 / (4.0 * self.k)) * (1.0 - math.exp(-2.0 * self.k * t))
        cf_diff = cmath.exp(i * u_c * mean_ou - var_diff * (u_c**2))

        # Jump integral: int_0^t lambda_j * [exp(i * u * exp(-k*s) * mu_j - 0.5 * sigma_j^2 * u^2 * exp(-2*k*s)) - 1] ds
        def jump_integrand(s: float) -> complex:
            u_s = u_c * math.exp(-self.k * s)
            phi_y = cmath.exp(i * u_s * self.mu_j - 0.5 * (self.sigma_j**2) * (u_s**2))
            return self.lambda_j * (phi_y - 1.0)

        # Numerical quadrature for jump exponent
        int_re, _ = integrate.quad(lambda s: jump_integrand(s).real, 0.0, t)
        int_im, _ = integrate.quad(lambda s: jump_integrand(s).imag, 0.0, t)

        return cf_diff * cmath.exp(complex(int_re, int_im))

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
        """Simulate sample paths with exact Gaussian OU step and Poisson jumps."""
        if seed is not None:
            np.random.seed(seed)

        final_t = t_max if t_max is not None else (T if T is not None else 1.0)
        steps = num_steps if num_steps is not None else (n_steps if n_steps is not None else 100)
        paths_count = n_paths if n_paths is not None else num_paths

        dt = float(final_t) / steps
        exp_kdt = math.exp(-self.k * dt)
        ou_std = math.sqrt((self.sigma**2 / (2.0 * self.k)) * (1.0 - math.exp(-2.0 * self.k * dt)))

        paths = np.zeros((paths_count, steps + 1))
        paths[:, 0] = self.x0

        for step in range(steps):
            x_prev = paths[:, step]
            z = np.random.normal(0.0, 1.0, paths_count)
            x_diff = x_prev * exp_kdt + self.theta * (1.0 - exp_kdt) + ou_std * z

            num_jumps = np.random.poisson(self.lambda_j * dt, paths_count)
            total_jumps = np.zeros(paths_count)
            for path_idx, n_j in enumerate(num_jumps):
                if n_j > 0:
                    total_jumps[path_idx] = np.sum(np.random.normal(self.mu_j, self.sigma_j, n_j))

            paths[:, step + 1] = x_diff + total_jumps

        return paths


# Aliases
JumpOU = JumpOUProcess
ExpOUJump = JumpOUProcess
EnergyOU = JumpOUProcess
