"""Tests for Heston model surface calibration and Fourier pricer."""

import numpy as np
import pytest

from spxa.calibration.surface.data import MarketSurface
from spxa.calibration.surface.heston import (
    HestonCalibrator,
    bs_call_price,
    bs_implied_vol,
    heston_call_price,
)


class TestHestonCalibrator:
    def test_bs_call_and_implied_vol_roundtrip(self):
        s0 = 100.0
        k = 105.0
        t = 1.0
        r = 0.03
        vol = 0.25

        price = bs_call_price(s0, k, t, r, vol)
        recovered_vol = bs_implied_vol(price, s0, k, t, r)
        assert recovered_vol == pytest.approx(vol, rel=1e-5)

    def test_heston_call_price_positive_and_convex(self):
        s0 = 100.0
        t = 0.5
        r = 0.02
        v0 = 0.04
        kappa = 2.0
        theta = 0.04
        sigma = 0.3
        rho = -0.5

        strikes = [90.0, 100.0, 110.0]
        prices = [heston_call_price(s0, k, t, r, v0, kappa, theta, sigma, rho) for k in strikes]
        assert prices[0] > prices[1] > prices[2]
        # Convexity: C(K1) - 2C(K2) + C(K3) >= 0 for equal spacing
        assert prices[0] - 2.0 * prices[1] + prices[2] >= 0.0

    def test_heston_calibrator_fit(self):
        # Generate small surface
        s0 = 100.0
        t = 0.25
        r = 0.01
        true_params = [0.04, 2.0, 0.04, 0.25, -0.6]

        strikes = [95.0, 100.0, 105.0]
        vols = []
        for k in strikes:
            c = heston_call_price(s0, k, t, r, *true_params)
            v = bs_implied_vol(c, s0, k, t, r)
            vols.append(v)

        surf = MarketSurface.from_grid(strikes, [t], np.array([vols]), spot=s0, rate=r)
        calibrator = HestonCalibrator(enforce_feller=True)

        res = calibrator.fit(surf, initial_params=[0.04, 1.8, 0.04, 0.28, -0.55])
        assert res.success is True
        assert res.rmse < 0.01
        assert "HestonCalibrator" in res.summary()
