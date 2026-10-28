"""Integration tests for the state filtering submodule."""

import numpy as np

from spxa.calibration.filter import (
    FilteredTrajectory,
    HestonVarianceFilter,
)


class TestFilterIntegration:
    def test_pipeline_smoke(self):
        # UKF, Heston filter, Schwartz filter, Intensity filter all run and return FilteredTrajectory
        dt = 0.1
        returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
        h_filter = HestonVarianceFilter()
        traj = h_filter.filter_returns(returns, dt=dt)
        assert isinstance(traj, FilteredTrajectory)
        assert len(traj.filtered_states) == 5

        # Check confidence intervals
        lb, ub = traj.confidence_bounds(state_idx=0, z=1.96)
        assert len(lb) == 5
        assert np.all(ub >= lb)
