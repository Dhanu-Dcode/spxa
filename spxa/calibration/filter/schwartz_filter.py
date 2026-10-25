r"""Schwartz (1997) Two-Factor Commodity Model Convenience Yield Filter.

Tracks the unobserved stochastic convenience yield \delta_t and spot commodity price S_t
from observed market futures term structures F(t, T_1), ..., F(t, T_M):
    d ln S_t = (r - \delta_t - 0.5 * \sigma_1^2) dt + \sigma_1 dW_1
    d \delta_t = \kappa * (\alpha - \delta_t) dt + \sigma_2 dW_2
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from spxa.calibration.filter.base import FilteredTrajectory, StateSpaceModel
from spxa.calibration.filter.ukf import UnscentedKalmanFilter


class SchwartzTwoFactorStateSpace(StateSpaceModel):
    r"""Linear-Gaussian state-space model for Schwartz 2-factor commodity dynamics.

    State: [ln(S_t), \delta_t]^T
    Observation: Log-futures prices ln F(t, T_m) for maturities T_1, ..., T_M.
    """

    def __init__(
        self,
        maturities: Sequence[float],
        kappa: float = 1.2,
        alpha: float = 0.05,
        sigma1: float = 0.25,
        sigma2: float = 0.30,
        rho: float = 0.60,
        rate: float = 0.03,
        meas_noise_std: float = 0.01,
    ) -> None:
        self.maturities = np.asarray(maturities, dtype=float)
        self.kappa = float(kappa)
        self.alpha = float(alpha)
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)
        self.rho = float(rho)
        self.rate = float(rate)
        self.meas_var = float(meas_noise_std**2)

    @property
    def dim_state(self) -> int:
        return 2

    @property
    def dim_obs(self) -> int:
        return len(self.maturities)

    def transition(self, state: np.ndarray, dt: float) -> np.ndarray:
        ln_s = float(state[0])
        delta = float(state[1])

        # Drift update
        ln_s_next = ln_s + (self.rate - delta - 0.5 * (self.sigma1**2)) * dt
        delta_next = delta + self.kappa * (self.alpha - delta) * dt
        return np.array([ln_s_next, delta_next])

    def transition_covariance(self, state: np.ndarray, dt: float) -> np.ndarray:
        cov11 = (self.sigma1**2) * dt
        cov22 = (self.sigma2**2) * dt
        cov12 = self.rho * self.sigma1 * self.sigma2 * dt
        return np.array([[cov11, cov12], [cov12, cov22]])

    def observation(self, state: np.ndarray) -> np.ndarray:
        ln_s = float(state[0])
        delta = float(state[1])
        k = self.kappa

        log_futures = np.empty(len(self.maturities), dtype=float)
        for i, tau in enumerate(self.maturities):
            # B(tau) = (1 - exp(-k * tau)) / k
            b_tau = (1.0 - math.exp(-k * tau)) / k if k > 1e-6 else tau
            # Deterministic adjustment A(tau)
            # A(tau) = (r - alpha + 0.5 * sigma2^2 / k^2 - sigma1*sigma2*rho / k) * tau + ...
            a_tau = (self.rate - self.alpha) * tau + 0.5 * (self.sigma2**2 / (k**2)) * tau
            log_futures[i] = ln_s - delta * b_tau + a_tau
        return log_futures

    def observation_covariance(self) -> np.ndarray:
        return self.meas_var * np.eye(len(self.maturities))


class SchwartzTwoFactorFilter:
    r"""Convenience yield tracker for Schwartz two-factor commodity models."""

    def __init__(
        self,
        maturities: Sequence[float],
        kappa: float = 1.2,
        alpha: float = 0.05,
        sigma1: float = 0.25,
        sigma2: float = 0.30,
        rho: float = 0.60,
        rate: float = 0.03,
    ) -> None:
        self.model = SchwartzTwoFactorStateSpace(
            maturities=maturities,
            kappa=kappa,
            alpha=alpha,
            sigma1=sigma1,
            sigma2=sigma2,
            rho=rho,
            rate=rate,
        )
        self.ukf = UnscentedKalmanFilter(self.model)

    def filter_futures(
        self,
        futures_matrix: np.ndarray,
        dt: float,
        initial_spot: float = 50.0,
        initial_yield: Optional[float] = None,
    ) -> FilteredTrajectory:
        r"""Filter convenience yield and spot path from matrix of futures prices: shape (n_steps, n_maturities)."""
        f_arr = np.asarray(futures_matrix, dtype=float)
        log_f = np.log(np.maximum(1e-6, f_arr))

        init_y = initial_yield if initial_yield is not None else self.model.alpha
        init_state = np.array([math.log(initial_spot), init_y])
        init_cov = np.diag([0.05, 0.05])

        traj = self.ukf.filter(
            observations=log_f,
            dt=dt,
            initial_state=init_state,
            initial_covariance=init_cov,
        )
        traj.state_names = ["log_spot", "convenience_yield"]
        return traj
