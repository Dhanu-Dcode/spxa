"""Exactness tests for Cheyette Quasi-Gaussian HJM Process."""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.cheyette import (
    Cheyette,
    CheyetteModel,
    CheyetteProcess,
    QuasiGaussianHJM,
)


class TestCheyetteExactness:
    def test_cheyette_exactness_level_and_aliases(self):
        process = CheyetteProcess(kappa=0.6, sigma=0.015, r0=0.03)
        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert Cheyette is CheyetteProcess
        assert CheyetteModel is CheyetteProcess
        assert QuasiGaussianHJM is CheyetteProcess

    def test_cheyette_triplet_raises_exactness_error(self):
        process = CheyetteProcess()
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_cheyette_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="kappa must be positive"):
            CheyetteProcess(kappa=-0.5)
        with pytest.raises(ValueError, match="sigma must be positive"):
            CheyetteProcess(sigma=-0.01)
        with pytest.raises(ValueError, match="y0 must be non-negative"):
            CheyetteProcess(y0=-0.001)

    def test_cheyette_g_factor_and_deterministic_y(self):
        process = CheyetteProcess(kappa=1.0, sigma=0.02, r0=0.03, y0=0.0)
        # G(0, 0) = 0
        assert process.g_factor(0.0, 0.0) == 0.0
        # G(0, 1) = (1 - e^-1) / 1
        expected_g = 1.0 - math.exp(-1.0)
        assert process.g_factor(0.0, 1.0) == pytest.approx(expected_g, rel=1e-7)

        # Deterministic y_t: at t=0, y=0; at t -> infty, y = sigma^2 / (2*kappa)
        assert process.y_deterministic(0.0) == 0.0
        stat_y = process.stationary_y()
        expected_stat_y = (0.02**2) / 2.0
        assert stat_y == pytest.approx(expected_stat_y, rel=1e-7)
        assert process.y_deterministic(50.0) == pytest.approx(stat_y, rel=1e-4)

    def test_cheyette_conditional_moments(self):
        process = CheyetteProcess(kappa=0.8, sigma=0.012, r0=0.025, x0=0.0, y0=0.0)
        # Mean of x_t at t=0 is 0
        assert process.mean_x(0.0) == 0.0
        # Mean of short rate at t=0 is r0
        assert process.mean_short_rate(0.0) == pytest.approx(0.025, rel=1e-7)

        # Variance of x_t at t=0 is 0
        assert process.variance_x(0.0) == 0.0
        # Variance of x_t at t=1 matches formula
        var_1 = process.variance_x(1.0)
        expected_var_1 = (0.012**2 / (2.0 * 0.8)) * (1.0 - math.exp(-1.6))
        assert var_1 == pytest.approx(expected_var_1, rel=1e-7)

    def test_cheyette_bond_pricing_and_european_option(self):
        process = CheyetteProcess(kappa=0.5, sigma=0.01, r0=0.03, x0=0.0, y0=0.0)
        # P(0, 0) = 1.0
        assert process.zero_coupon_bond(0.0) == pytest.approx(1.0, rel=1e-7)

        # P(0, 2) with flat rate = exp(-0.03 * 2) = exp(-0.06)
        # (since x0 = 0 and y0 = 0, g*x0 = 0, g^2*y0 = 0)
        assert process.zero_coupon_bond(2.0) == pytest.approx(math.exp(-0.06), rel=1e-7)

        # Yield curve at tau=2 should be 0.03
        assert process.yield_curve(2.0) == pytest.approx(0.03, rel=1e-7)

        # European bond option call pricing
        call_price = process.european_bond_call(maturity=1.0, bond_maturity=3.0, strike=0.90)
        assert call_price > 0.0
        # Call price must be bounded by forward bond price * P(0, T_opt) = P(0, T_bond)
        assert call_price < process.zero_coupon_bond(3.0)

    def test_cheyette_simulation_convergence(self):
        process = CheyetteProcess(kappa=0.7, sigma=0.015, r0=0.03, x0=0.0, y0=0.0)
        r_paths, x_paths, y_paths = process.simulate(
            t_max=1.0, num_steps=50, num_paths=2000, seed=42, return_states=True
        )
        assert r_paths.shape == (2000, 51)
        assert x_paths.shape == (2000, 51)
        assert y_paths.shape == (2000, 51)

        # Check mean of r_paths at t=1.0 matches analytical mean_short_rate(1.0)
        mc_r_mean = np.mean(r_paths[:, -1])
        analytic_r_mean = process.mean_short_rate(1.0)
        assert mc_r_mean == pytest.approx(analytic_r_mean, rel=0.05)

        # Check variance of x_paths at t=1.0 matches analytical variance_x(1.0)
        mc_x_var = np.var(x_paths[:, -1])
        analytic_x_var = process.variance_x(1.0)
        assert mc_x_var == pytest.approx(analytic_x_var, rel=0.10)
