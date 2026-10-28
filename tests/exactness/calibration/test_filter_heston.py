"""Tests for Heston stochastic volatility latent variance filter."""

import numpy as np

from spxa.calibration.filter.base import FilteredTrajectory
from spxa.calibration.filter.heston_filter import HestonVarianceFilter
from spxa.zoo import HestonProcess


class TestFilterHeston:
    def test_heston_variance_filter_tracking(self):
        rng = np.random.default_rng(42)
        n = 200
        dt = 1.0 / 252.0  # Daily

        kappa, theta, sigma, rho = 3.0, 0.04, 0.3, -0.6
        heston = HestonProcess(v0=0.04, kappa=kappa, theta=theta, sigma=sigma, rho=rho, s0=100.0)

        # Simulate joint asset and variance paths
        paths, var_paths = heston.simulate_joint(n_steps=n, n_paths=1, T=n * dt, rng=rng)
        s_path = paths[0]
        true_var = var_paths[0, 1:]  # variance at steps 1..n
        log_ret = np.diff(np.log(s_path))

        h_filter = HestonVarianceFilter(kappa=kappa, theta=theta, sigma=sigma, rho=rho)
        traj = h_filter.filter_returns(log_ret, dt=dt, v0=0.04)

        assert isinstance(traj, FilteredTrajectory)
        assert len(traj.filtered_states) == n
        # Filtered variance is positive
        assert np.all(traj.filtered_states > 0.0)
        # Correlation with true variance should be positive
        corr = np.corrcoef(traj.filtered_states[:, 0], true_var)[0, 1]
        assert corr > 0.40
