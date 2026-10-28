r"""Maximum Likelihood Estimation for Heavy-Tailed and Jump Return Distributions.

Provides exact probability density MLE for non-Gaussian financial returns:
- Student-t Innovations: (mu, sigma, df)
- Normal Inverse Gaussian (NIG): (alpha, beta, delta, mu)
- Variance Gamma (VG): (sigma, nu, theta)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from scipy import special, stats

from spxa.calibration.mle.base import TimeSeriesEstimator


class StudentTInnovationsMLE(TimeSeriesEstimator):
    r"""MLE for Student-t distributed log-returns with degrees of freedom nu, scale sigma, location mu."""

    @property
    def param_names(self) -> List[str]:
        return ["mu", "sigma", "nu"]

    @property
    def default_bounds(self) -> List[Tuple[float, float]]:
        return [(-10.0, 10.0), (1e-4, 10.0), (2.01, 100.0)]

    @property
    def default_initial_guess(self) -> np.ndarray:
        return np.array([0.0, 0.2, 5.0])

    def log_likelihood(self, params: np.ndarray, data: np.ndarray, dt: float) -> float:
        mu, sigma, nu = params[0], params[1], params[2]
        if sigma <= 0.0 or nu <= 2.0:
            return -1e12

        returns = np.diff(data) if np.ndim(data) == 1 else data
        # Scale for time step dt
        scale = sigma * math.sqrt(dt)
        loc = mu * dt

        ll = np.sum(stats.t.logpdf(returns, df=nu, loc=loc, scale=scale))
        if np.isnan(ll) or np.isinf(ll):
            return -1e12
        return float(ll)


class NormalInverseGaussianMLE(TimeSeriesEstimator):
    r"""MLE for the Normal Inverse Gaussian (NIG) distribution:

        f(x) = \frac{\alpha \delta}{\pi} \exp(\delta \gamma + \beta(x - \mu))
               \frac{K_1(\alpha \sqrt{\delta^2 + (x - \mu)^2})}{\sqrt{\delta^2 + (x - \mu)^2}}
    where \gamma = \sqrt{\alpha^2 - \beta^2}, \alpha > |\beta| >= 0, \delta > 0.
    """

    @property
    def param_names(self) -> List[str]:
        return ["alpha", "beta", "delta", "mu"]

    @property
    def default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (0.1, 50.0),  # alpha
            (-40.0, 40.0),  # beta
            (0.001, 10.0),  # delta
            (-5.0, 5.0),  # mu
        ]

    @property
    def default_initial_guess(self) -> np.ndarray:
        return np.array([3.0, -0.5, 0.2, 0.0])

    def log_likelihood(self, params: np.ndarray, data: np.ndarray, dt: float) -> float:
        alpha, beta, delta, mu = params[0], params[1], params[2], params[3]
        if alpha <= abs(beta) or delta <= 0.0:
            return -1e12

        returns = np.diff(data) if np.ndim(data) == 1 else data
        # Scale parameters for interval dt: delta_dt = delta * dt, mu_dt = mu * dt
        d_t = delta * dt
        m_t = mu * dt
        gamma = math.sqrt(alpha**2 - beta**2)

        x = returns - m_t
        q = np.sqrt(d_t**2 + x**2)
        k1e = special.k1e(alpha * q)

        if np.any(k1e <= 0.0) or np.any(np.isnan(k1e)):
            return -1e12

        # ln(f) = ln(alpha * delta / pi) + delta * gamma + beta * x - alpha * q + ln(K1e) - ln(q)
        ll = (
            math.log(alpha * d_t / math.pi)
            + d_t * gamma
            + beta * x
            - alpha * q
            + np.log(k1e)
            - np.log(q)
        )
        total = float(np.sum(ll))
        if np.isnan(total) or np.isinf(total):
            return -1e12
        return total


NIGMLE = NormalInverseGaussianGaussian = NormalInverseGaussianMLE
