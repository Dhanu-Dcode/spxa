"""Tests for exact Gaussian diffusion maximum likelihood estimators."""

import numpy as np
import pytest

from spxa.calibration.mle.gaussian import (
    BlackScholesMLE,
    BrownianMotionMLE,
    GeometricBrownianMotionMLE,
    OrnsteinUhlenbeckMLE,
    VasicekMLE,
)


class TestMLEGaussian:
    def test_ou_mle_fit_and_aliases(self):
        assert VasicekMLE is OrnsteinUhlenbeckMLE
        rng = np.random.default_rng(42)
        dt = 0.05
        n = 1500

        true_kappa = 2.0
        true_theta = 0.05
        true_sigma = 0.15

        # Simulate exact OU discrete steps
        e_k = np.exp(-true_kappa * dt)
        std_cond = np.sqrt(
            (true_sigma**2 / (2.0 * true_kappa)) * (1.0 - np.exp(-2.0 * true_kappa * dt))
        )
        x = np.empty(n + 1)
        x[0] = 0.04
        for i in range(n):
            m = x[i] * e_k + true_theta * (1.0 - e_k)
            x[i + 1] = rng.normal(m, std_cond)

        est = OrnsteinUhlenbeckMLE()
        res_analytical = est.analytical_fit(x, dt=dt)

        assert res_analytical.success is True
        assert res_analytical.params["kappa"] == pytest.approx(true_kappa, rel=0.15)
        assert res_analytical.params["theta"] == pytest.approx(true_theta, rel=0.15)
        assert res_analytical.params["sigma"] == pytest.approx(true_sigma, rel=0.10)

    def test_brownian_motion_mle(self):
        rng = np.random.default_rng(123)
        dt = 0.1
        n = 800
        true_mu = 0.3
        true_sigma = 0.25

        dx = rng.normal(true_mu * dt, true_sigma * np.sqrt(dt), size=n)
        x = np.concatenate([[10.0], 10.0 + np.cumsum(dx)])

        est = BrownianMotionMLE()
        res = est.analytical_fit(x, dt=dt)

        assert res.success is True
        assert res.params["mu"] == pytest.approx(true_mu, rel=0.15)
        assert res.params["sigma"] == pytest.approx(true_sigma, rel=0.10)

    def test_geometric_brownian_motion_mle(self):
        assert BlackScholesMLE is GeometricBrownianMotionMLE
        rng = np.random.default_rng(456)
        dt = 1.0 / 252.0  # Daily
        n = 1000
        true_mu = 0.08
        true_sigma = 0.20

        # Simulate GBM
        drift_r = (true_mu - 0.5 * true_sigma**2) * dt
        vol_r = true_sigma * np.sqrt(dt)
        r_steps = rng.normal(drift_r, vol_r, size=n)
        s = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(r_steps)]))

        est = GeometricBrownianMotionMLE()
        res = est.analytical_fit(s, dt=dt)

        assert res.success is True
        assert res.params["mu"] == pytest.approx(true_mu, abs=0.05)
        assert res.params["sigma"] == pytest.approx(true_sigma, abs=0.03)
