"""Tests for Empirical Characteristic Function (ECF) estimator."""

import numpy as np
import pytest

from spxa.calibration.mle.ecf import ECFEstimator, ECFResult


class TestMLEECF:
    def test_ecf_gaussian_recovery(self):
        # Test recovering mu and sigma of Gaussian from sample characteristic function
        # phi(u) = exp(i * u * mu - 0.5 * sigma^2 * u^2)
        def gaussian_cf(u, theta):
            mu, sigma = theta[0], theta[1]
            return np.exp(1j * u * mu - 0.5 * (sigma**2) * (u**2))

        rng = np.random.default_rng(42)
        true_mu = 0.5
        true_sigma = 1.2
        samples = rng.normal(true_mu, true_sigma, size=1500)

        estimator = ECFEstimator(
            char_func=gaussian_cf,
            param_names=["mu", "sigma"],
            default_bounds=[(-5.0, 5.0), (0.1, 5.0)],
            default_initial_guess=[0.0, 1.0],
            u_grid=np.linspace(-3.0, 3.0, 31),
        )

        res = estimator.fit(samples)
        assert isinstance(res, ECFResult)
        assert res.success is True
        assert res.params["mu"] == pytest.approx(true_mu, abs=0.1)
        assert res.params["sigma"] == pytest.approx(true_sigma, abs=0.1)
        assert res.loss < 1e-3
        assert "ECF Estimation Report" in res.summary()
