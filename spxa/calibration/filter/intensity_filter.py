r"""Default Hazard Rate Intensity Filter for credit risk models.

Tracks latent default intensity \lambda_t from observed credit default swap (CDS) spreads
or corporate bond yield spreads under Duffie-Singleton (1999) dynamics:
    d \lambda_t = \kappa * (\theta - \lambda_t) dt + \sigma * \sqrt{\lambda_t} dW_t
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from spxa.calibration.filter.base import FilteredTrajectory, StateSpaceModel
from spxa.calibration.filter.ukf import UnscentedKalmanFilter


class IntensityStateSpaceModel(StateSpaceModel):
    r"""State-space model for latent default intensity \lambda_t."""

    def __init__(
        self,
        maturities: Sequence[float],
        kappa: float = 0.5,
        theta: float = 0.02,
        sigma: float = 0.10,
        loss_given_default: float = 0.60,
        meas_noise: float = 0.0005,
    ) -> None:
        self.maturities = np.asarray(maturities, dtype=float)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.lgd = float(loss_given_default)
        self.meas_var = float(meas_noise**2)

    @property
    def dim_state(self) -> int:
        return 1

    @property
    def dim_obs(self) -> int:
        return len(self.maturities)

    def transition(self, state: np.ndarray, dt: float) -> np.ndarray:
        lam = max(1e-5, float(state[0]))
        lam_next = lam + self.kappa * (self.theta - lam) * dt
        return np.array([max(1e-5, lam_next)])

    def transition_covariance(self, state: np.ndarray, dt: float) -> np.ndarray:
        lam = max(1e-5, float(state[0]))
        q = (self.sigma**2) * lam * dt
        return np.array([[max(1e-8, q)]])

    def observation(self, state: np.ndarray) -> np.ndarray:
        lam = max(1e-5, float(state[0]))
        # Par spread approx: s(T) = LGD * lambda_avg(T)
        # lambda_avg(T) = theta + (lambda - theta) * (1 - exp(-kappa * T)) / (kappa * T)
        spreads = np.empty(len(self.maturities), dtype=float)
        for i, t in enumerate(self.maturities):
            factor = (
                (1.0 - math.exp(-self.kappa * t)) / (self.kappa * t)
                if self.kappa * t > 1e-6
                else 1.0
            )
            lam_avg = self.theta + (lam - self.theta) * factor
            spreads[i] = self.lgd * lam_avg
        return spreads

    def observation_covariance(self) -> np.ndarray:
        return self.meas_var * np.eye(len(self.maturities))


class DefaultIntensityFilter:
    r"""Filter for tracking latent credit default hazard rates \lambda_t from credit spreads."""

    def __init__(
        self,
        maturities: Sequence[float],
        kappa: float = 0.5,
        theta: float = 0.02,
        sigma: float = 0.10,
        loss_given_default: float = 0.60,
    ) -> None:
        self.model = IntensityStateSpaceModel(
            maturities=maturities,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            loss_given_default=loss_given_default,
        )
        self.ukf = UnscentedKalmanFilter(self.model)

    def filter_spreads(
        self,
        spreads_matrix: np.ndarray,
        dt: float,
        initial_lambda: float = 0.02,
    ) -> FilteredTrajectory:
        r"""Filter hazard rate path from observed CDS spread curves."""
        s_mat = np.asarray(spreads_matrix, dtype=float)
        init_state = np.array([initial_lambda])
        init_cov = np.array([[0.001]])

        traj = self.ukf.filter(
            observations=s_mat,
            dt=dt,
            initial_state=init_state,
            initial_covariance=init_cov,
        )
        traj.state_names = ["hazard_rate"]
        return traj
