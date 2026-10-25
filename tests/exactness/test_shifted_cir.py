"""Exactness tests for Shifted Cox-Ingersoll-Ross (CIR++) Process."""

import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.shifted_cir import (
    ShiftedCIRProcess,
    ShiftedCIR,
    CIRPlusPlus,
)


@pytest.mark.exactness
class TestShiftedCIRExactness:
    def test_shifted_cir_exactness_level_and_aliases(self):
        process = ShiftedCIRProcess(k=0.5, theta=0.04, sigma=0.1, x0=0.03, phi0=0.01)
        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert ShiftedCIR is ShiftedCIRProcess
        assert CIRPlusPlus is ShiftedCIRProcess

    def test_shifted_cir_triplet_raises_exactness_error(self):
        process = ShiftedCIRProcess(k=0.5, theta=0.04, sigma=0.1)
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_shifted_cir_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="Parameters must be positive"):
            ShiftedCIRProcess(k=-0.5, theta=0.04, sigma=0.1)
        with pytest.raises(ValueError, match="Initial factor x0"):
            ShiftedCIRProcess(k=0.5, theta=0.04, sigma=0.1, x0=-0.01)

    def test_shifted_cir_properties_and_feller(self):
        process = ShiftedCIRProcess(k=1.2, theta=0.03, sigma=0.1, x0=0.02, phi0=0.005, phi1=0.01, alpha=0.5)
        # 2 * 1.2 * 0.03 / 0.01 = 7.2
        assert process.feller_ratio == pytest.approx(7.2, rel=1e-5)
        assert process.phi(0.0) == pytest.approx(0.015, rel=1e-5)
        assert process.r0 == pytest.approx(0.035, rel=1e-5)

    def test_shifted_cir_phi_integral(self):
        process = ShiftedCIRProcess(k=0.5, theta=0.03, sigma=0.05, phi0=0.01, phi1=0.02, alpha=0.3)
        # Analytic integral from 0 to T: phi0 * T + phi1 * (1 - exp(-alpha*T)) / alpha
        t = 2.0
        expected = 0.01 * 2.0 + 0.02 * (1.0 - np.exp(-0.6)) / 0.3
        assert process.phi_integral(t) == pytest.approx(expected, rel=1e-6)

    def test_shifted_cir_conditional_moments(self):
        process = ShiftedCIRProcess(k=0.6, theta=0.04, sigma=0.08, x0=0.02, phi0=0.01)
        assert process.mean(0.0) == pytest.approx(process.r0, rel=1e-6)
        assert process.variance(0.0) == pytest.approx(0.0, abs=1e-12)

        t_far = 50.0
        assert process.mean(t_far) == pytest.approx(process.theta + process.phi(t_far), rel=1e-3)
        assert process.variance(t_far) == pytest.approx(process.theta * process.sigma**2 / (2.0 * process.k), rel=1e-3)

    def test_shifted_cir_bond_pricing_and_yield_curve(self):
        process = ShiftedCIRProcess(k=0.5, theta=0.03, sigma=0.05, x0=0.02, phi0=0.01, phi1=0.005, alpha=0.2)
        assert process.zero_coupon_bond(0.0) == pytest.approx(1.0, rel=1e-7)

        maturities = np.array([0.5, 1.0, 2.0, 5.0])
        p = process.zero_coupon_bond(maturities)
        assert np.all(p > 0.0)
        assert np.all(p <= 1.0)
        assert np.all(np.diff(p) < 0.0)

        yields = process.yield_curve(maturities)
        assert yields[0] == pytest.approx(process.r0, rel=0.1)

    def test_shifted_cir_european_bond_call(self):
        process = ShiftedCIRProcess(k=0.8, theta=0.03, sigma=0.05, x0=0.03, phi0=0.01)
        call_atm = process.european_bond_call(maturity=1.0, bond_maturity=2.0, strike=0.95)
        assert call_atm > 0.0
        call_deep_itm = process.european_bond_call(maturity=1.0, bond_maturity=2.0, strike=0.5)
        call_deep_otm = process.european_bond_call(maturity=1.0, bond_maturity=2.0, strike=1.5)
        assert call_deep_itm > call_atm > call_deep_otm
        assert call_deep_otm == pytest.approx(0.0, abs=1e-10)

    def test_shifted_cir_simulation_convergence(self):
        process = ShiftedCIRProcess(k=0.7, theta=0.03, sigma=0.04, x0=0.02, phi0=0.008)
        paths = process.simulate(t_max=1.0, num_steps=40, num_paths=2000, seed=42)
        assert paths.shape == (2000, 41)
        assert np.all(paths >= 0.0)

        sim_mean_1y = np.mean(paths[:, -1])
        analytic_mean_1y = process.mean(1.0)
        assert sim_mean_1y == pytest.approx(analytic_mean_1y, rel=0.05)
