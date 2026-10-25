"""Exactness tests for Two-Factor Cox-Ingersoll-Ross (CIR2) Process."""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.two_factor_cir import (
    TwoFactorCIRProcess,
    TwoFactorCIR,
    CIR2PlusPlus,
)


@pytest.mark.exactness
class TestTwoFactorCIRExactness:
    def test_cir2_exactness_level_and_aliases(self):
        process = TwoFactorCIRProcess(
            k_x=0.5, theta_x=0.03, sigma_x=0.08,
            k_y=0.2, theta_y=0.02, sigma_y=0.05,
            x0=0.02, y0=0.01, phi=0.005,
        )
        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert TwoFactorCIR is TwoFactorCIRProcess
        assert CIR2PlusPlus is TwoFactorCIRProcess

    def test_cir2_triplet_raises_exactness_error(self):
        process = TwoFactorCIRProcess(
            k_x=0.5, theta_x=0.03, sigma_x=0.08,
            k_y=0.2, theta_y=0.02, sigma_y=0.05,
        )
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_cir2_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="Factor X parameters"):
            TwoFactorCIRProcess(k_x=-0.5, theta_x=0.03, sigma_x=0.08, k_y=0.2, theta_y=0.02, sigma_y=0.05)
        with pytest.raises(ValueError, match="Initial factor values"):
            TwoFactorCIRProcess(k_x=0.5, theta_x=0.03, sigma_x=0.08, k_y=0.2, theta_y=0.02, sigma_y=0.05, x0=-0.01)

    def test_cir2_properties_and_feller(self):
        process = TwoFactorCIRProcess(
            k_x=1.0, theta_x=0.04, sigma_x=0.1,  # 2*1*0.04 = 0.08 > 0.01 -> Feller = 8
            k_y=0.5, theta_y=0.02, sigma_y=0.1,  # 2*0.5*0.02 = 0.02 > 0.01 -> Feller = 2
            x0=0.03, y0=0.01, phi=0.005,
        )
        assert process.feller_ratio_x == pytest.approx(8.0, rel=1e-5)
        assert process.feller_ratio_y == pytest.approx(2.0, rel=1e-5)
        assert process.r0 == pytest.approx(0.045, rel=1e-5)

    def test_cir2_moments_and_stationary_limits(self):
        process = TwoFactorCIRProcess(
            k_x=0.8, theta_x=0.03, sigma_x=0.06,
            k_y=0.3, theta_y=0.02, sigma_y=0.05,
            x0=0.01, y0=0.04, phi=0.002,
        )
        # At t=0
        assert process.mean(0.0) == pytest.approx(process.r0, rel=1e-6)
        assert process.variance(0.0) == pytest.approx(0.0, abs=1e-12)

        # As t -> infinity
        t_large = 100.0
        assert process.mean(t_large) == pytest.approx(process.stationary_mean(), rel=1e-3)
        assert process.variance(t_large) == pytest.approx(process.stationary_variance(), rel=1e-3)

    def test_cir2_zero_coupon_bond_properties(self):
        process = TwoFactorCIRProcess(
            k_x=0.6, theta_x=0.03, sigma_x=0.08,
            k_y=0.2, theta_y=0.02, sigma_y=0.05,
            x0=0.02, y0=0.02, phi=0.0,
        )
        # P(0) = 1
        assert process.zero_coupon_bond(0.0) == pytest.approx(1.0, rel=1e-7)

        # Monotonicity with maturity
        maturities = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        p = process.zero_coupon_bond(maturities)
        assert np.all(p > 0.0)
        assert np.all(p <= 1.0)
        assert np.all(np.diff(p) < 0.0)  # Strictly decreasing prices with maturity

        # Yield curve limits
        yields = process.yield_curve(maturities)
        assert yields[0] == pytest.approx(process.r0, rel=0.1)

    def test_cir2_simulation_paths_and_mean(self):
        process = TwoFactorCIRProcess(
            k_x=0.8, theta_x=0.03, sigma_x=0.05,
            k_y=0.4, theta_y=0.02, sigma_y=0.04,
            x0=0.02, y0=0.02, phi=0.001,
        )
        paths = process.simulate(t_max=1.0, num_steps=50, num_paths=2000, seed=42)
        assert paths.shape == (2000, 51)
        assert np.all(paths >= 0.0)  # Always non-negative rates

        sim_mean_1y = np.mean(paths[:, -1])
        analytical_mean_1y = process.mean(1.0)
        assert sim_mean_1y == pytest.approx(analytical_mean_1y, rel=0.05)
