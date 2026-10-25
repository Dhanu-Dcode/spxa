r"""Heston Latent Variance Filter for recovering instantaneous volatility from returns."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from spxa.calibration.filter.base import FilteredTrajectory, StateSpaceModel
from spxa.calibration.filter.ukf import UnscentedKalmanFilter


class HestonStateSpaceModel(StateSpaceModel):
    r"""Continuous-discrete state space model for Heston log-variance tracking:

    State: Latent log-variance x_t = ln(v_t)
        v = exp(x_t)
        v_next = v + \kappa * (\theta - v) * dt
        x_next = ln(max(1e-6, v_next))
        Var[v] = \sigma^2 * v * dt -> Var[x] \approx \sigma^2 * dt / v

    Observation: Realized return volatility proxy
        y_k \approx \sqrt{v} = exp(0.5 * x_k)
    """

    def __init__(
        self,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma: float = 0.3,
        rho: float = -0.7,
        rate: float = 0.0,
    ) -> None:
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.rate = float(rate)

    @property
    def dim_state(self) -> int:
        return 1

    @property
    def dim_obs(self) -> int:
        return 1

    def transition(self, state: np.ndarray, dt: float) -> np.ndarray:
        x = float(state[0])
        v = math.exp(np.clip(x, -15.0, 5.0))
        v_next = max(1e-6, v + self.kappa * (self.theta - v) * dt)
        return np.array([math.log(v_next)])

    def transition_covariance(self, state: np.ndarray, dt: float) -> np.ndarray:
        x = float(state[0])
        v = math.exp(np.clip(x, -15.0, 5.0))
        # Delta method: Var[ln v] = (1/v)^2 * Var[v] = (1/v^2) * (sigma^2 * v * dt) = sigma^2 * dt / v
        q = (self.sigma**2) * dt / max(1e-4, v)
        return np.array([[max(1e-6, min(10.0, q))]])

    def observation(self, state: np.ndarray) -> np.ndarray:
        x = float(state[0])
        vol = math.exp(0.5 * np.clip(x, -15.0, 5.0))
        return np.array([vol])

    def observation_covariance(self) -> np.ndarray:
        return np.array([[0.005]])


class HestonVarianceFilter:
    r"""Filter for tracking unobservable instantaneous variance v_t from return volatility."""

    def __init__(
        self,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma: float = 0.3,
        rho: float = -0.7,
        rate: float = 0.0,
    ) -> None:
        self.model = HestonStateSpaceModel(
            kappa=kappa, theta=theta, sigma=sigma, rho=rho, rate=rate
        )
        self.ukf = UnscentedKalmanFilter(self.model)

    def filter_returns(
        self,
        returns: np.ndarray,
        dt: float,
        v0: Optional[float] = None,
    ) -> FilteredTrajectory:
        r"""Filter variance trajectory using local realized volatility proxy |r_t| / sqrt(dt)."""
        ret = np.asarray(returns, dtype=float)
        # Realized volatility proxy
        vol_proxy = (math.sqrt(math.pi / 2.0) * np.abs(ret)) / math.sqrt(dt)

        init_v = v0 if v0 is not None else self.model.theta
        init_state = np.array([math.log(max(1e-6, init_v))])
        init_cov = np.array([[0.1]])

        log_var_traj = self.ukf.filter(
            observations=vol_proxy,
            dt=dt,
            initial_state=init_state,
            initial_covariance=init_cov,
        )

        # Convert filtered log-variance state back to variance: v_t = exp(x_t)
        var_states = np.exp(log_var_traj.filtered_states)
        # Approximate variance covariance: Var[v] = v^2 * Var[x]
        var_covs = (var_states[:, :, None] ** 2) * log_var_traj.filtered_covariances

        return FilteredTrajectory(
            times=log_var_traj.times,
            filtered_states=var_states,
            filtered_covariances=var_covs,
            innovations=log_var_traj.innovations,
            log_likelihood=log_var_traj.log_likelihood,
            state_names=["variance"],
        )
