"""Tests for SABR model surface calibration and Hagan implied volatility formula."""

import numpy as np
import pytest

from spxa.calibration.surface.data import MarketSurface
from spxa.calibration.surface.sabr import SABRCalibrator, sabr_hagan_implied_vol


class TestSABRCalibrator:
    def test_sabr_hagan_implied_vol_atm_and_wings(self):
        f = 100.0
        t = 1.0
        alpha = 0.20
        beta = 1.0
        rho = -0.30
        nu = 0.40

        atm_vol = sabr_hagan_implied_vol(f, 100.0, t, alpha, beta, rho, nu)
        assert 0.15 < atm_vol < 0.25

        otm_put_vol = sabr_hagan_implied_vol(f, 80.0, t, alpha, beta, rho, nu)
        otm_call_vol = sabr_hagan_implied_vol(f, 120.0, t, alpha, beta, rho, nu)

        # With negative rho=-0.3, skew should be downward sloping (put vol > call vol)
        assert otm_put_vol > otm_call_vol

    def test_sabr_calibration_recovery(self):
        # Generate synthetic quotes with known SABR parameters
        true_alpha = 0.22
        true_beta = 1.0  # Log-normal SABR
        true_rho = -0.40
        true_nu = 0.35

        f0 = 100.0
        t = 0.5
        strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
        vols = [
            sabr_hagan_implied_vol(f0, k, t, true_alpha, true_beta, true_rho, true_nu)
            for k in strikes
        ]

        surf = MarketSurface.from_grid(strikes, [t], np.array([vols]), spot=f0)

        # Calibrate with fixed beta = 1.0
        calibrator = SABRCalibrator(fixed_beta=1.0)
        res = calibrator.fit(surf)

        assert res.success is True
        assert res.rmse < 1e-4
        assert res.calibrated_params["alpha"] == pytest.approx(true_alpha, abs=1e-3)
        assert res.calibrated_params["rho"] == pytest.approx(true_rho, abs=1e-2)
        assert res.calibrated_params["nu"] == pytest.approx(true_nu, abs=1e-2)
