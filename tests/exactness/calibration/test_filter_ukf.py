"""Tests for Unscented Kalman Filter (UKF) state estimation."""

import numpy as np

from spxa.calibration.filter.base import FilteredTrajectory, StateSpaceModel
from spxa.calibration.filter.ukf import UKF, UnscentedKalmanFilter


class LinearTrackingModel(StateSpaceModel):
    """Linear state space model where Kalman Filter is optimal."""

    @property
    def dim_state(self):
        return 1

    @property
    def dim_obs(self):
        return 1

    def transition(self, state, dt):
        return 0.8 * state

    def transition_covariance(self, state, dt):
        return np.array([[0.04]])

    def observation(self, state):
        return 1.0 * state

    def observation_covariance(self):
        return np.array([[0.01]])


class TestFilterUKF:
    def test_ukf_aliases_and_initialization(self):
        assert UKF is UnscentedKalmanFilter
        model = LinearTrackingModel()
        ukf = UnscentedKalmanFilter(model)
        assert ukf.model is model

    def test_ukf_tracks_hidden_state(self):
        rng = np.random.default_rng(42)
        n = 100
        dt = 1.0
        model = LinearTrackingModel()

        # Simulate true latent state and noisy observations
        true_x = np.empty(n)
        true_x[0] = 5.0
        for i in range(1, n):
            true_x[i] = 0.8 * true_x[i - 1] + rng.normal(0.0, 0.2)

        # Observations: y_k = x_k + v_k (std=0.1)
        obs = true_x + rng.normal(0.0, 0.1, size=n)

        ukf = UKF(model)
        traj = ukf.filter(obs, dt=dt, initial_state=np.array([4.0]))

        assert isinstance(traj, FilteredTrajectory)
        assert len(traj.filtered_states) == n
        # Filtered states should track true states closely: correlation > 0.95
        corr = np.corrcoef(traj.filtered_states[:, 0], true_x)[0, 1]
        assert corr > 0.95
        # RMSE between filtered and true state should be small
        rmse = np.sqrt(np.mean((traj.filtered_states[:, 0] - true_x) ** 2))
        assert rmse < 0.20
        assert np.isfinite(traj.log_likelihood)
