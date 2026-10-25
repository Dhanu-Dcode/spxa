"""Tests for DefaultIntensityFilter hazard rate tracking."""

import numpy as np

from spxa.calibration.filter.base import FilteredTrajectory
from spxa.calibration.filter.intensity_filter import (
    DefaultIntensityFilter,
    IntensityStateSpaceModel,
)


class TestFilterIntensity:
    def test_intensity_model_dimensions(self):
        mats = [1.0, 3.0, 5.0]
        model = IntensityStateSpaceModel(maturities=mats)
        assert model.dim_state == 1
        assert model.dim_obs == 3

    def test_intensity_filter_tracking(self):
        rng = np.random.default_rng(42)
        n = 60
        dt = 0.05
        mats = [1.0, 3.0, 5.0]

        model = IntensityStateSpaceModel(maturities=mats, kappa=0.8, theta=0.03, sigma=0.08)

        # Simulate true intensity path
        true_lam = np.empty(n)
        true_lam[0] = 0.02
        for t in range(1, n):
            true_lam[t] = model.transition(np.array([true_lam[t - 1]]), dt)[0] + rng.normal(
                0.0, 0.002
            )

        # Simulate observed CDS spreads
        obs_spreads = np.empty((n, 3))
        for t in range(n):
            obs_spreads[t] = model.observation(np.array([true_lam[t]])) + rng.normal(
                0.0, 0.0002, size=3
            )

        intensity_filter = DefaultIntensityFilter(
            maturities=mats, kappa=0.8, theta=0.03, sigma=0.08
        )
        traj = intensity_filter.filter_spreads(obs_spreads, dt=dt, initial_lambda=0.02)

        assert isinstance(traj, FilteredTrajectory)
        assert len(traj.filtered_states) == n
        corr = np.corrcoef(traj.filtered_states[:, 0], true_lam)[0, 1]
        assert corr > 0.85
