"""Tests for Schwartz two-factor commodity convenience yield filter."""

import numpy as np

from spxa.calibration.filter.base import FilteredTrajectory
from spxa.calibration.filter.schwartz_filter import (
    SchwartzTwoFactorFilter,
    SchwartzTwoFactorStateSpace,
)


class TestFilterSchwartz:
    def test_schwartz_dimensions(self):
        mats = [0.25, 0.5, 1.0]
        ss = SchwartzTwoFactorStateSpace(maturities=mats)
        assert ss.dim_state == 2
        assert ss.dim_obs == 3

    def test_schwartz_filter_tracking(self):
        rng = np.random.default_rng(42)
        n = 80
        dt = 0.05
        mats = [0.25, 0.5, 1.0]

        kappa, alpha, sigma1, sigma2, rho, rate = 1.5, 0.08, 0.20, 0.25, 0.5, 0.03
        model = SchwartzTwoFactorStateSpace(
            maturities=mats,
            kappa=kappa,
            alpha=alpha,
            sigma1=sigma1,
            sigma2=sigma2,
            rho=rho,
            rate=rate,
        )

        # Simulate true state trajectory
        true_states = np.empty((n, 2))
        true_states[0] = [np.log(60.0), 0.08]
        for t in range(1, n):
            true_states[t] = model.transition(true_states[t - 1], dt) + rng.normal(
                0.0, 0.02, size=2
            )

        # Simulate observed futures
        obs_futures = np.empty((n, 3))
        for t in range(n):
            obs_log_f = model.observation(true_states[t]) + rng.normal(0.0, 0.005, size=3)
            obs_futures[t] = np.exp(obs_log_f)

        s_filter = SchwartzTwoFactorFilter(
            maturities=mats,
            kappa=kappa,
            alpha=alpha,
            sigma1=sigma1,
            sigma2=sigma2,
            rho=rho,
            rate=rate,
        )
        traj = s_filter.filter_futures(obs_futures, dt=dt, initial_spot=60.0, initial_yield=0.08)

        assert isinstance(traj, FilteredTrajectory)
        assert traj.filtered_states.shape == (n, 2)
        # Check spot and convenience yield correlation with truth
        corr_spot = np.corrcoef(traj.filtered_states[:, 0], true_states[:, 0])[0, 1]
        corr_yield = np.corrcoef(traj.filtered_states[:, 1], true_states[:, 1])[0, 1]
        assert corr_spot > 0.95
        assert corr_yield > 0.70
