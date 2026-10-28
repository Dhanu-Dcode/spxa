"""Tests for CEV-Heston local-stochastic volatility surface calibrator."""

import numpy as np
import pytest

from spxa.calibration.surface.cev_heston import CEVHestonCalibrator, cev_heston_approx_implied_vol
from spxa.calibration.surface.data import MarketSurface
from spxa.calibration.surface.heston import bs_implied_vol, heston_call_price


class TestCEVHestonCalibrator:
    def test_cev_heston_reduces_to_heston_at_gamma_one(self):
        s0 = 100.0
        k = 100.0
        t = 1.0
        r = 0.02
        v0, kappa, theta, sigma, rho = 0.04, 2.0, 0.04, 0.3, -0.6

        vol_cev = cev_heston_approx_implied_vol(
            s0, k, t, r, v0, kappa, theta, sigma, rho, gamma=1.0
        )
        call_heston = heston_call_price(s0, k, t, r, v0, kappa, theta, sigma, rho)
        vol_heston = bs_implied_vol(call_heston, s0, k, t, r)

        assert vol_cev == pytest.approx(vol_heston, rel=1e-5)

    def test_cev_heston_calibrator_fit(self):
        s0 = 100.0
        t = 0.5
        r = 0.01
        true_params = [0.04, 2.0, 0.04, 0.25, -0.6, 0.8]
        strikes = [95.0, 100.0, 105.0]

        vols = [cev_heston_approx_implied_vol(s0, k, t, r, *true_params) for k in strikes]
        surf = MarketSurface.from_grid(strikes, [t], np.array([vols]), spot=s0, rate=r)
        calibrator = CEVHestonCalibrator(enforce_feller=False)

        res = calibrator.fit(surf, initial_params=true_params)
        assert res.success is True
        assert res.rmse < 1e-4
