"""Tests for Newey-West HAC covariance estimator."""

import numpy as np

from spxa.calibration.mle.hac import newey_west_hac


class TestMLEHAC:
    def test_hac_positive_definite(self):
        rng = np.random.default_rng(42)
        n = 500
        # AR(1) autocorrelated series: x_t = 0.6 * x_{t-1} + eps
        eps = rng.normal(size=n)
        x = np.empty(n)
        x[0] = eps[0]
        for t in range(1, n):
            x[t] = 0.6 * x[t - 1] + eps[t]

        # Moments matrix with 2 dimensions
        m = np.column_stack([x, x**2 - 1.0])
        omega = newey_west_hac(m, lags=5)

        assert omega.shape == (2, 2)
        # Check symmetry
        np.testing.assert_allclose(omega, omega.T, atol=1e-10)
        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(omega)
        assert np.all(eigvals > 0.0)

    def test_hac_automatic_lag_length(self):
        # 1000 observations -> lag = floor(4 * (10)^(2/9)) = floor(4 * 1.668) = 6
        m = np.random.normal(size=(1000, 3))
        omega = newey_west_hac(m, lags=None)
        assert omega.shape == (3, 3)
