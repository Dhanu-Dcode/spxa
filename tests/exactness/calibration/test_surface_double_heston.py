"""Tests for Double Heston two-factor surface calibrator and characteristic function."""

import numpy as np
import pytest

from spxa.calibration.surface.data import MarketSurface
from spxa.calibration.surface.double_heston import DoubleHestonCalibrator, double_heston_call_price
from spxa.calibration.surface.heston import bs_implied_vol, heston_call_price


class TestDoubleHestonCalibrator:
    def test_double_heston_reduces_to_single_heston_when_factor2_inactive(self):
        s0 = 100.0
        k = 100.0
        t = 1.0
        r = 0.03

        # Factor 1 parameters
        v0_1, k1, th1, sig1, rho1 = 0.04, 2.0, 0.04, 0.3, -0.6
        # Factor 2 negligible parameters (variance near 0)
        v0_2, k2, th2, sig2, rho2 = 1e-6, 1.0, 1e-6, 1e-4, 0.0

        p_double = double_heston_call_price(
            s0, k, t, r, np.array([v0_1, k1, th1, sig1, rho1, v0_2, k2, th2, sig2, rho2])
        )
        p_single = heston_call_price(s0, k, t, r, v0_1, k1, th1, sig1, rho1)

        assert p_double == pytest.approx(p_single, rel=1e-3)

    def test_double_heston_calibrator_fit(self):
        s0 = 100.0
        t = 0.5
        r = 0.01

        # Known parameters
        true_params = np.array(
            [
                0.02,
                5.0,
                0.02,
                0.30,
                -0.70,
                0.02,
                0.5,
                0.02,
                0.15,
                -0.40,
            ]
        )
        strikes = [95.0, 100.0, 105.0]
        vols = []
        for k in strikes:
            c = double_heston_call_price(s0, k, t, r, true_params)
            vols.append(bs_implied_vol(c, s0, k, t, r))

        surf = MarketSurface.from_grid(strikes, [t], np.array([vols]), spot=s0, rate=r)
        calibrator = DoubleHestonCalibrator(enforce_feller=False)

        res = calibrator.fit(surf, initial_params=true_params)
        assert res.success is True
        assert res.rmse < 1e-4
