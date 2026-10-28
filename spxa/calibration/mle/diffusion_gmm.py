r"""Multi-Moment GMM Estimators for Continuous-Time Diffusions.

Matches empirical stationary moments and lagged autocovariances to exact theoretical formulas:
- Ornstein-Uhlenbeck GMM: matches mean \theta, variance \frac{\sigma^2}{2\kappa}, and autocovariance
- Cox-Ingersoll-Ross GMM: matches mean \theta, variance \frac{\sigma^2 \theta}{2\kappa}, and autocovariance
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from spxa.calibration.mle.gmm_base import GMMEstimator


class OrnsteinUhlenbeckGMM(GMMEstimator):
    r"""GMM estimator for the Ornstein-Uhlenbeck process using stationary moments:

    1. E[X_t] = theta
    2. E[(X_t - theta)^2] = sigma^2 / (2 * kappa)
    3. E[(X_t - theta) * (X_{t-1} - theta)] = (sigma^2 / (2 * kappa)) * exp(-kappa * dt)
    4. E[(X_t - theta)^3] = 0 (Gaussian symmetry)
    """

    def __init__(self, dt: float = 1.0) -> None:
        self.dt = float(dt)

    @property
    def param_names(self) -> List[str]:
        return ["kappa", "theta", "sigma"]

    @property
    def default_bounds(self) -> List[Tuple[float, float]]:
        return [(0.01, 20.0), (-1e3, 1e3), (0.01, 10.0)]

    @property
    def default_initial_guess(self) -> np.ndarray:
        return np.array([1.0, 0.0, 0.2])

    def moment_conditions(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        kappa, theta, sigma = params[0], params[1], params[2]
        x_curr = data[1:]
        x_prev = data[:-1]

        stat_var = (sigma**2) / (2.0 * kappa)
        cov_lag1 = stat_var * math.exp(-kappa * self.dt)

        m1 = x_curr - theta
        m2 = (x_curr - theta) ** 2 - stat_var
        m3 = (x_curr - theta) * (x_prev - theta) - cov_lag1
        m4 = (x_curr - theta) ** 3  # Skewness is zero

        return np.column_stack([m1, m2, m3, m4])


class CIRGMM(GMMEstimator):
    r"""GMM estimator for the Cox-Ingersoll-Ross (CIR) process:

    1. E[X_t] = theta
    2. E[(X_t - theta)^2] = (sigma^2 * theta) / (2 * kappa)
    3. E[(X_t - theta) * (X_{t-1} - theta)] = ((sigma^2 * theta) / (2 * kappa)) * exp(-kappa * dt)
    """

    def __init__(self, dt: float = 1.0) -> None:
        self.dt = float(dt)

    @property
    def param_names(self) -> List[str]:
        return ["kappa", "theta", "sigma"]

    @property
    def default_bounds(self) -> List[Tuple[float, float]]:
        return [(0.01, 20.0), (0.001, 1.0), (0.01, 2.0)]

    @property
    def default_initial_guess(self) -> np.ndarray:
        return np.array([1.0, 0.05, 0.15])

    def moment_conditions(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        kappa, theta, sigma = params[0], params[1], params[2]
        x_curr = data[1:]
        x_prev = data[:-1]

        stat_var = (sigma**2 * theta) / (2.0 * kappa)
        cov_lag1 = stat_var * math.exp(-kappa * self.dt)

        m1 = x_curr - theta
        m2 = (x_curr - theta) ** 2 - stat_var
        m3 = (x_curr - theta) * (x_prev - theta) - cov_lag1

        return np.column_stack([m1, m2, m3])
