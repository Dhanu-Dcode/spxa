"""Tests for residual diagnostics, Ljung-Box test, and Jarque-Bera test."""

import numpy as np
import pytest

from spxa.calibration.diagnostics.residuals import ResidualDiagnostics, compute_residual_diagnostics


class TestDiagResiduals:
    def test_residual_diagnostics_gaussian_white_noise(self):
        rng = np.random.default_rng(42)
        # Independent Gaussian white noise
        e = rng.normal(0.0, 1.0, size=1000)

        diag = compute_residual_diagnostics(e, lags=10)
        assert isinstance(diag, ResidualDiagnostics)

        assert diag.mean == pytest.approx(0.0, abs=0.08)
        assert diag.std == pytest.approx(1.0, abs=0.08)
        assert diag.skewness == pytest.approx(0.0, abs=0.15)
        assert diag.kurtosis == pytest.approx(3.0, abs=0.30)

        # Ljung-Box and Jarque-Bera should fail to reject null for Gaussian white noise
        assert diag.ljung_box_pvalue > 0.05
        assert diag.jarque_bera_pvalue > 0.05
        assert "No Autocorrelation" in diag.summary()
        assert "Normal" in diag.summary()

    def test_residual_diagnostics_autocorrelation_detected(self):
        rng = np.random.default_rng(123)
        n = 500
        # Highly persistent AR(1)
        e = np.empty(n)
        e[0] = 0.0
        for t in range(1, n):
            e[t] = 0.7 * e[t - 1] + rng.normal(0.0, 1.0)

        diag = compute_residual_diagnostics(e, lags=5)
        # Should detect autocorrelation (reject null)
        assert diag.ljung_box_pvalue < 1e-4
        assert "Autocorrelation Detected" in diag.summary()
