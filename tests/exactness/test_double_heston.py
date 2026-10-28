"""Exactness tests for Double Heston Two-Factor Stochastic Volatility Model."""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.double_heston import (
    DoubleHestonProcess,
    DoubleHeston,
    TwoFactorHeston,
)


@pytest.mark.exactness
class TestDoubleHestonExactness:
    def test_double_heston_exactness_level_and_aliases(self):
        process = DoubleHestonProcess(
            s0=100.0,
            v1_0=0.02,
            v2_0=0.02,
            k1=2.0,
            theta1=0.02,
            sigma1=0.2,
            rho1=-0.5,
            k2=0.5,
            theta2=0.03,
            sigma2=0.1,
            rho2=-0.3,
            r=0.03,
            q=0.01,
        )
        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert DoubleHeston is DoubleHestonProcess
        assert TwoFactorHeston is DoubleHestonProcess

    def test_double_heston_triplet_raises_exactness_error(self):
        process = DoubleHestonProcess(
            s0=100.0,
            v1_0=0.02,
            v2_0=0.02,
            k1=2.0,
            theta1=0.02,
            sigma1=0.2,
            rho1=-0.5,
            k2=0.5,
            theta2=0.03,
            sigma2=0.1,
            rho2=-0.3,
        )
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_double_heston_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="Initial spot price"):
            DoubleHestonProcess(
                s0=-10.0,
                v1_0=0.02,
                v2_0=0.02,
                k1=2.0,
                theta1=0.02,
                sigma1=0.2,
                rho1=-0.5,
                k2=0.5,
                theta2=0.03,
                sigma2=0.1,
                rho2=-0.3,
            )
        with pytest.raises(ValueError, match="rho1 and rho2"):
            DoubleHestonProcess(
                s0=100.0,
                v1_0=0.02,
                v2_0=0.02,
                k1=2.0,
                theta1=0.02,
                sigma1=0.2,
                rho1=-1.5,
                k2=0.5,
                theta2=0.03,
                sigma2=0.1,
                rho2=-0.3,
            )

    def test_double_heston_variance_aggregations(self):
        process = DoubleHestonProcess(
            s0=100.0,
            v1_0=0.015,
            v2_0=0.025,
            k1=3.0,
            theta1=0.02,
            sigma1=0.25,
            rho1=-0.6,
            k2=0.4,
            theta2=0.04,
            sigma2=0.15,
            rho2=-0.2,
        )
        assert process.total_initial_variance == pytest.approx(0.04, rel=1e-7)
        assert process.total_long_run_variance == pytest.approx(0.06, rel=1e-7)
        assert process.mean_variance(0.0) == pytest.approx(0.04, rel=1e-7)

    def test_double_heston_characteristic_function(self):
        process = DoubleHestonProcess(
            s0=100.0,
            v1_0=0.02,
            v2_0=0.02,
            k1=2.0,
            theta1=0.02,
            sigma1=0.2,
            rho1=-0.5,
            k2=0.5,
            theta2=0.02,
            sigma2=0.1,
            rho2=-0.3,
            r=0.02,
            q=0.0,
        )
        # CF at u=0 must be 1.0
        cf_0 = process.characteristic_function(0.0, t=1.0)
        assert abs(cf_0 - 1.0) < 1e-6

        # Modulus <= 1 for real u
        for u_val in [0.5, 1.0, 2.0, 5.0]:
            cf_val = process.characteristic_function(u_val, t=0.5)
            assert abs(cf_val) <= 1.000001

    def test_double_heston_option_pricing_put_call_parity(self):
        process = DoubleHestonProcess(
            s0=100.0,
            v1_0=0.02,
            v2_0=0.02,
            k1=2.5,
            theta1=0.02,
            sigma1=0.3,
            rho1=-0.7,
            k2=0.6,
            theta2=0.03,
            sigma2=0.15,
            rho2=-0.2,
            r=0.04,
            q=0.01,
        )
        t = 0.5
        for strike in [90.0, 100.0, 110.0]:
            call = process.european_call(strike, t)
            put = process.european_put(strike, t)
            parity_forward = process.s0 * math.exp(-process.q * t) - strike * math.exp(
                -process.r * t
            )
            assert (call - put) == pytest.approx(parity_forward, abs=1e-4)

    def test_double_heston_simulation_martingale(self):
        process = DoubleHestonProcess(
            s0=100.0,
            v1_0=0.02,
            v2_0=0.02,
            k1=2.0,
            theta1=0.02,
            sigma1=0.2,
            rho1=-0.5,
            k2=0.8,
            theta2=0.02,
            sigma2=0.1,
            rho2=-0.2,
            r=0.03,
            q=0.01,
        )
        s_paths, v1_paths, v2_paths = process.simulate(
            t_max=0.5, num_steps=25, num_paths=1500, seed=123
        )
        assert s_paths.shape == (1500, 26)
        assert np.all(v1_paths >= 0.0)
        assert np.all(v2_paths >= 0.0)

        sim_mean_s = np.mean(s_paths[:, -1])
        analytic_mean_s = process.mean_spot(0.5)
        assert sim_mean_s == pytest.approx(analytic_mean_s, rel=0.03)
