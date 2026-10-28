r"""Unscented Kalman Filter (UKF) for nonlinear continuous-discrete state-space models.

Propagates mean and covariance through the Merwe-Wan scaled unscented transformation
without requiring analytical Jacobian derivations.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from spxa.calibration.filter.base import FilteredTrajectory, StateSpaceModel
from spxa.calibration.filter.sigma_points import MerweScaledSigmaPoints


class UnscentedKalmanFilter:
    r"""Unscented Kalman Filter (UKF) for non-linear state estimation and innovation log-likelihood."""

    def __init__(
        self,
        model: StateSpaceModel,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self.model = model
        self.sigma_points = MerweScaledSigmaPoints(
            dim=model.dim_state, alpha=alpha, beta=beta, kappa=kappa
        )

    def filter(
        self,
        observations: Sequence[float] | np.ndarray,
        dt: float,
        initial_state: Optional[np.ndarray] = None,
        initial_covariance: Optional[np.ndarray] = None,
    ) -> FilteredTrajectory:
        r"""Run the UKF forward across sequential observation vectors."""
        y_obs = np.asarray(observations, dtype=float)
        if y_obs.ndim == 1 and self.model.dim_obs == 1:
            y_obs = y_obs[:, None]

        n_steps, dim_obs = y_obs.shape
        dim_state = self.model.dim_state

        # Initial conditions
        x_curr = (
            np.asarray(initial_state, dtype=float)
            if initial_state is not None
            else np.zeros(dim_state)
        )
        p_curr = (
            np.asarray(initial_covariance, dtype=float)
            if initial_covariance is not None
            else np.eye(dim_state)
        )

        filtered_states = np.empty((n_steps, dim_state), dtype=float)
        filtered_covs = np.empty((n_steps, dim_state, dim_state), dtype=float)
        innovations = np.empty((n_steps, dim_obs), dtype=float)
        total_ll = 0.0

        times = np.arange(n_steps) * dt
        r_mat = self.model.observation_covariance()

        for k in range(n_steps):
            y_k = y_obs[k]

            # ----------------------------------------------------
            # 1. Prediction Step
            # ----------------------------------------------------
            # Generate sigma points around (x_curr, p_curr)
            chi = self.sigma_points.generate(x_curr, p_curr)
            # Propagate through nonlinear state transition
            chi_pred = np.empty_like(chi)
            for i in range(len(chi)):
                chi_pred[i] = self.model.transition(chi[i], dt)

            # Process noise covariance
            q_mat = self.model.transition_covariance(x_curr, dt)
            x_pred, p_pred = self.sigma_points.reconstruct(chi_pred, noise_cov=q_mat)

            # ----------------------------------------------------
            # 2. Measurement Update Step
            # ----------------------------------------------------
            # Generate sigma points around (x_pred, p_pred)
            chi_meas = self.sigma_points.generate(x_pred, p_pred)
            # Propagate through measurement function
            gamma_meas = np.empty((self.sigma_points.n_points, dim_obs), dtype=float)
            for i in range(len(chi_meas)):
                gamma_meas[i] = self.model.observation(chi_meas[i])

            # Predicted measurement and innovation covariance
            y_pred = np.sum(self.sigma_points.wm[:, None] * gamma_meas, axis=0)
            diff_y = gamma_meas - y_pred[None, :]
            p_yy = diff_y.T @ (self.sigma_points.wc[:, None] * diff_y) + r_mat

            # Cross-covariance P_xy
            diff_x = chi_meas - x_pred[None, :]
            p_xy = diff_x.T @ (self.sigma_points.wc[:, None] * diff_y)

            # Kalman gain K = P_xy @ P_yy^{-1}
            try:
                k_gain = p_xy @ np.linalg.pinv(p_yy)
            except np.linalg.LinAlgError:
                k_gain = np.zeros((dim_state, dim_obs))

            # Innovation residual
            v_k = y_k - y_pred
            x_curr = x_pred + k_gain @ v_k
            p_curr = p_pred - k_gain @ p_yy @ k_gain.T

            # Symmetrize and regularize covariance
            p_curr = 0.5 * (p_curr + p_curr.T)

            # Innovation log-likelihood contribution
            try:
                sign, logdet = np.linalg.slogdet(p_yy)
                if sign > 0:
                    quad = float(v_k.T @ np.linalg.pinv(p_yy) @ v_k)
                    ll_k = -0.5 * (dim_obs * math.log(2.0 * math.pi) + logdet + quad)
                    total_ll += ll_k
            except Exception:
                pass

            filtered_states[k] = x_curr
            filtered_covs[k] = p_curr
            innovations[k] = v_k

        return FilteredTrajectory(
            times=times,
            filtered_states=filtered_states,
            filtered_covariances=filtered_covs,
            innovations=innovations,
            log_likelihood=total_ll,
        )


UKF = UnscentedKalmanFilter
