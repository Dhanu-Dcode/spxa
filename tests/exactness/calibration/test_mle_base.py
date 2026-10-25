"""Tests for TimeSeriesEstimator base class and EstimationResult container."""

import numpy as np
import pytest

from spxa.calibration.mle.base import EstimationResult, TimeSeriesEstimator


class DummyNormalEstimator(TimeSeriesEstimator):
    """Test estimator fitting mean mu and variance sigma^2 of independent Gaussian steps."""

    @property
    def param_names(self):
        return ["mu", "sigma"]

    @property
    def default_bounds(self):
        return [(-10.0, 10.0), (0.01, 10.0)]

    @property
    def default_initial_guess(self):
        return np.array([0.0, 1.0])

    def log_likelihood(self, params, data, dt):
        mu, sigma = params[0], params[1]
        dx = np.diff(data)
        # dx ~ N(mu * dt, sigma^2 * dt)
        var = (sigma**2) * dt
        z = dx - mu * dt
        ll = -0.5 * len(dx) * np.log(2.0 * np.pi * var) - 0.5 * np.sum(z**2) / var
        return float(ll)


class TestMLEBase:
    def test_mle_base_fit_and_diagnostics(self):
        rng = np.random.default_rng(42)
        dt = 0.1
        n = 500
        true_mu = 1.5
        true_sigma = 0.5

        # Generate Brownian motion path with drift
        dx = rng.normal(true_mu * dt, true_sigma * np.sqrt(dt), size=n)
        x = np.concatenate([[0.0], np.cumsum(dx)])

        est = DummyNormalEstimator()
        res = est.fit(x, dt=dt)

        assert isinstance(res, EstimationResult)
        assert res.success is True
        assert res.params["mu"] == pytest.approx(true_mu, abs=0.1)
        assert res.params["sigma"] == pytest.approx(true_sigma, abs=0.05)
        assert res.standard_errors["mu"] > 0.0
        assert res.standard_errors["sigma"] > 0.0
        assert np.isfinite(res.aic)
        assert np.isfinite(res.bic)

        summary = res.summary()
        assert "DummyNormalEstimator" in summary
        assert "CONVERGED" in summary
