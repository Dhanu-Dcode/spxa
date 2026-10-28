r"""Exact Maximum Likelihood Estimators for Gaussian and Log-Normal Diffusions.

Provides exact closed-form score estimators and likelihood functions for:
- Ornstein-Uhlenbeck / Vasicek process: dX_t = \kappa * (\theta - X_t) dt + \sigma dW_t
- Arithmetic Brownian Motion: dX_t = \mu dt + \sigma dW_t
- Geometric Brownian Motion: dS_t = \mu S_t dt + \sigma S_t dW_t
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from spxa.calibration.mle.base import EstimationResult, TimeSeriesEstimator


class OrnsteinUhlenbeckMLE(TimeSeriesEstimator):
    r"""Exact Maximum Likelihood Estimator for the Ornstein-Uhlenbeck (Vasicek) diffusion:

        dX_t = \kappa * (\theta - X_t) dt + \sigma dW_t

    Conditional distribution:
        X_{i+1} | X_i ~ N( X_i * exp(-\kappa * dt) + \theta * (1 - exp(-\kappa * dt)),
                           \frac{\sigma^2}{2\kappa} * (1 - exp(-2\kappa * dt)) )
    """

    @property
    def param_names(self) -> List[str]:
        return ["kappa", "theta", "sigma"]

    @property
    def default_bounds(self) -> List[Tuple[float, float]]:
        return [(1e-4, 50.0), (-1e4, 1e4), (1e-4, 10.0)]

    @property
    def default_initial_guess(self) -> np.ndarray:
        return np.array([1.0, 0.0, 0.2])

    def log_likelihood(self, params: np.ndarray, data: np.ndarray, dt: float) -> float:
        kappa, theta, sigma = params[0], params[1], params[2]
        if kappa <= 0.0 or sigma <= 0.0:
            return -1e12

        x_prev = data[:-1]
        x_curr = data[1:]
        n = len(x_prev)

        e_k = math.exp(-kappa * dt)
        mean_cond = x_prev * e_k + theta * (1.0 - e_k)
        var_cond = (sigma**2 / (2.0 * kappa)) * (1.0 - math.exp(-2.0 * kappa * dt))
        if var_cond <= 0.0:
            return -1e12

        diff = x_curr - mean_cond
        ll = -0.5 * n * math.log(2.0 * math.pi * var_cond) - 0.5 * np.sum(diff**2) / var_cond
        return float(ll)

    def analytical_fit(self, data: np.ndarray, dt: float) -> EstimationResult:
        """Compute exact closed-form OLS-equivalent maximum likelihood estimates."""
        x = np.asarray(data, dtype=float)
        n = len(x) - 1
        x_prev = x[:-1]
        x_curr = x[1:]

        sx = np.sum(x_prev)
        sy = np.sum(x_curr)
        sxx = np.sum(x_prev**2)
        syy = np.sum(x_curr**2)
        sxy = np.sum(x_prev * x_curr)

        # AR(1) linear regression: x_{i+1} = a * x_i + b + eps
        denom = n * sxx - sx**2
        if denom == 0.0:
            return self.fit(data, dt)

        a = (n * sxy - sx * sy) / denom
        b = (sy - a * sx) / n

        # Boundary checks
        if a <= 0.0 or a >= 1.0:
            return self.fit(data, dt)

        kappa = -math.log(a) / dt
        theta = b / (1.0 - a)

        # Residual variance
        var_eps = (1.0 / n) * (
            syy - 2.0 * a * sxy - 2.0 * b * sy + (a**2) * sxx + 2.0 * a * b * sx + n * (b**2)
        )
        var_eps = max(1e-12, var_eps)
        sigma = math.sqrt(var_eps * 2.0 * kappa / (1.0 - a**2))

        p_opt = np.array([kappa, theta, sigma])
        return self.fit(data, dt, initial_params=p_opt)


VasicekMLE = OrnsteinUhlenbeckMLE


class BrownianMotionMLE(TimeSeriesEstimator):
    r"""Exact Maximum Likelihood Estimator for Arithmetic Brownian Motion:

    dX_t = \mu dt + \sigma dW_t
    """

    @property
    def param_names(self) -> List[str]:
        return ["mu", "sigma"]

    @property
    def default_bounds(self) -> List[Tuple[float, float]]:
        return [(-1e4, 1e4), (1e-4, 50.0)]

    @property
    def default_initial_guess(self) -> np.ndarray:
        return np.array([0.0, 0.2])

    def log_likelihood(self, params: np.ndarray, data: np.ndarray, dt: float) -> float:
        mu, sigma = params[0], params[1]
        if sigma <= 0.0:
            return -1e12

        dx = np.diff(data)
        n = len(dx)
        var = (sigma**2) * dt
        diff = dx - mu * dt
        ll = -0.5 * n * math.log(2.0 * math.pi * var) - 0.5 * np.sum(diff**2) / var
        return float(ll)

    def analytical_fit(self, data: np.ndarray, dt: float) -> EstimationResult:
        """Compute exact closed-form score estimators."""
        x = np.asarray(data, dtype=float)
        dx = np.diff(x)
        len(dx)

        mu = float(np.mean(dx) / dt)
        var = float(np.mean((dx - mu * dt) ** 2) / dt)
        sigma = math.sqrt(max(1e-12, var))

        return self.fit(data, dt, initial_params=np.array([mu, sigma]))


class GeometricBrownianMotionMLE(TimeSeriesEstimator):
    r"""Exact Maximum Likelihood Estimator for Geometric Brownian Motion:

    dS_t = \mu S_t dt + \sigma S_t dW_t
    """

    @property
    def param_names(self) -> List[str]:
        return ["mu", "sigma"]

    @property
    def default_bounds(self) -> List[Tuple[float, float]]:
        return [(-1e4, 1e4), (1e-4, 50.0)]

    @property
    def default_initial_guess(self) -> np.ndarray:
        return np.array([0.05, 0.20])

    def log_likelihood(self, params: np.ndarray, data: np.ndarray, dt: float) -> float:
        mu, sigma = params[0], params[1]
        if sigma <= 0.0:
            return -1e12
        if np.any(data <= 0.0):
            return -1e12

        log_returns = np.diff(np.log(data))
        n = len(log_returns)
        # r ~ N((mu - 0.5 * sigma^2) * dt, sigma^2 * dt)
        mean_r = (mu - 0.5 * (sigma**2)) * dt
        var_r = (sigma**2) * dt

        diff = log_returns - mean_r
        ll = -0.5 * n * math.log(2.0 * math.pi * var_r) - 0.5 * np.sum(diff**2) / var_r
        return float(ll)

    def analytical_fit(self, data: np.ndarray, dt: float) -> EstimationResult:
        """Compute exact closed-form log-return estimators."""
        s = np.asarray(data, dtype=float)
        log_ret = np.diff(np.log(s))

        sample_mean = float(np.mean(log_ret))
        sample_var = float(np.var(log_ret))

        sigma = math.sqrt(max(1e-12, sample_var / dt))
        mu = float(sample_mean / dt + 0.5 * (sigma**2))

        return self.fit(data, dt, initial_params=np.array([mu, sigma]))


BlackScholesMLE = GeometricBrownianMotionMLE
