"""Tests for StateSpaceModel base class and FilteredTrajectory container."""

import numpy as np

from spxa.calibration.filter.base import FilteredTrajectory, StateSpaceModel


class Dummy1DModel(StateSpaceModel):
    @property
    def dim_state(self):
        return 1

    @property
    def dim_obs(self):
        return 1

    def transition(self, state, dt):
        return 0.9 * state

    def transition_covariance(self, state, dt):
        return np.array([[0.05 * dt]])

    def observation(self, state):
        return 2.0 * state

    def observation_covariance(self):
        return np.array([[0.01]])


class TestFilterBase:
    def test_filtered_trajectory_confidence_bounds(self):
        n = 50
        times = np.linspace(0.0, 1.0, n)
        states = np.zeros((n, 1))
        covs = np.zeros((n, 1, 1))
        covs[:, 0, 0] = 0.04  # std = 0.2
        innovs = np.zeros((n, 1))

        traj = FilteredTrajectory(
            times=times,
            filtered_states=states,
            filtered_covariances=covs,
            innovations=innovs,
            log_likelihood=120.5,
            state_names=["x"],
        )

        lb, ub = traj.confidence_bounds(state_idx=0, z=2.0)
        np.testing.assert_allclose(lb, -0.4, atol=1e-7)
        np.testing.assert_allclose(ub, 0.4, atol=1e-7)

    def test_state_space_model_dimensions(self):
        m = Dummy1DModel()
        assert m.dim_state == 1
        assert m.dim_obs == 1
        np.testing.assert_allclose(m.transition(np.array([10.0]), 0.1), [9.0])
        np.testing.assert_allclose(m.observation(np.array([5.0])), [10.0])
