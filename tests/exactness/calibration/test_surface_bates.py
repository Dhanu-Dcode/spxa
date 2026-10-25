"""Tests for Bates jump-diffusion surface calibrator and characteristic function."""

import numpy as np
import pytest

from spxa.calibration.surface.bates import BatesCalibrator, bates_call_price
from spxa.calibration.surface.data import MarketSurface
from spxa.calibration.surface.heston import bs_implied_vol, heston_call_price


class TestBatesCalibrator:
    def test_bates_reduces_to_heston_when_jump_rate_is_zero(self):
        s0 = 100.0
        k = 100.0
        t = 1.0
        r = 0.02
        v0 = 0.04
        kappa = 1.5
        theta = 0.04
        sigma = 0.3
        rho = -0.5

        # Pure Heston price
        price_heston = heston_call_price(s0, k, t, r, v0, kappa, theta, sigma, rho)

        # Bates with jump_rate = 0.0
        price_bates = bates_call_price(
            s0, k, t, r, v0, kappa, theta, sigma, rho, jump_rate=0.0, jump_mean=-0.05, jump_std=0.10
        )
        assert price_bates == pytest.approx(price_heston, rel=1e-5)

    def test_bates_jumps_increase_wing_implied_vol(self):
        s0 = 100.0
        t = 0.25
        r = 0.0
        v0 = 0.04
        kappa = 2.0
        theta = 0.04
        sigma = 0.2
        rho = 0.0

        # OTM Put strike K = 80
        p_no_jump = bates_call_price(
            s0, 80.0, t, r, v0, kappa, theta, sigma, rho, jump_rate=0.0, jump_mean=0.0, jump_std=0.1
        )
        p_with_jump = bates_call_price(
            s0,
            80.0,
            t,
            r,
            v0,
            kappa,
            theta,
            sigma,
            rho,
            jump_rate=2.0,
            jump_mean=-0.1,
            jump_std=0.15,
        )
        # Adding downward jumps increases deep OTM option prices
        assert p_with_jump > p_no_jump

    def test_bates_calibrator_fit(self):
        s0 = 100.0
        t = 0.5
        r = 0.02
        true_params = [0.04, 2.0, 0.04, 0.25, -0.6, 0.5, -0.05, 0.10]
        strikes = [90.0, 100.0, 110.0]

        vols = []
        for k in strikes:
            c = bates_call_price(s0, k, t, r, *true_params)
            vols.append(bs_implied_vol(c, s0, k, t, r))

        surf = MarketSurface.from_grid(strikes, [t], np.array([vols]), spot=s0, rate=r)
        calibrator = BatesCalibrator(enforce_feller=False)

        res = calibrator.fit(surf, initial_params=true_params)
        assert res.success is True
        assert res.rmse < 1e-4
