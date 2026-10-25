"""Exactness tests for CEV-Heston Local-Stochastic Volatility Process."""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.cev_heston import (
    CEVHeston,
    CEVHestonProcess,
    HestonCEV,
    SLVCEV,
)


class TestCEVHestonExactness:
    def test_cev_heston_exactness_level_and_aliases(self):
        process = CEVHestonProcess(s0=100.0, v0=0.04, gamma=0.8, kappa=2.0, theta=0.04, sigma_v=0.3)
        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert CEVHeston is CEVHestonProcess
        assert HestonCEV is CEVHestonProcess
        assert SLVCEV is CEVHestonProcess

    def test_cev_heston_triplet_raises_exactness_error(self):
        process = CEVHestonProcess()
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_cev_heston_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="Initial spot price"):
            CEVHestonProcess(s0=-10.0)
        with pytest.raises(ValueError, match="Initial variance"):
            CEVHestonProcess(v0=-0.04)
        with pytest.raises(ValueError, match="CEV elasticity gamma"):
            CEVHestonProcess(gamma=0.0)
        with pytest.raises(ValueError, match="Variance parameters"):
            CEVHestonProcess(kappa=-1.0)
        with pytest.raises(ValueError, match="Correlation rho"):
            CEVHestonProcess(rho=1.5)

    def test_cev_heston_feller_and_variance_moments(self):
        process = CEVHestonProcess(s0=100.0, v0=0.05, gamma=0.7, kappa=2.5, theta=0.04, sigma_v=0.2)
        # Feller ratio = 2 * 2.5 * 0.04 / 0.04 = 5.0 > 1.0
        assert process.feller_ratio == pytest.approx(5.0, rel=1e-7)
        assert process.satisfies_feller is True

        # Expected variance at t=0 and t -> infty
        assert process.expected_variance(0.0) == pytest.approx(0.05, rel=1e-7)
        assert process.expected_variance(100.0) == pytest.approx(0.04, rel=1e-4)

        # Variance of variance at t=0
        assert process.variance_of_variance(0.0) == pytest.approx(0.0, abs=1e-9)
        # Stationary variance of variance = theta * sigma_v^2 / (2 * kappa)
        stat_var = 0.04 * 0.04 / 5.0
        assert process.stationary_variance_var() == pytest.approx(stat_var, rel=1e-7)
        assert process.variance_of_variance(100.0) == pytest.approx(stat_var, rel=1e-3)

    def test_cev_heston_limit_to_heston_char_func_and_pricing(self):
        # When gamma = 1.0, it reduces to standard Heston
        process = CEVHestonProcess(
            s0=100.0,
            v0=0.04,
            gamma=1.0,
            kappa=2.0,
            theta=0.04,
            sigma_v=0.3,
            rho=-0.7,
            r=0.03,
            q=0.01,
        )
        assert process.is_heston is True

        # Characteristic function at u=0 should equal 1.0
        cf_0 = process.characteristic_function_heston(0.0, t=1.0)
        assert cf_0.real == pytest.approx(1.0, rel=1e-7)
        assert abs(cf_0.imag) < 1e-7

        # European call and put prices with put-call parity
        call_atm = process.european_call(strike=100.0, maturity=1.0)
        put_atm = process.european_put(strike=100.0, maturity=1.0)
        fwd_parity = 100.0 * math.exp(-0.01) - 100.0 * math.exp(-0.03)
        assert call_atm - put_atm == pytest.approx(fwd_parity, rel=1e-5)

    def test_cev_heston_simulation_and_martingale_property(self):
        # Under r=0, q=0, E[S_t] = S_0
        process = CEVHestonProcess(
            s0=100.0,
            v0=0.04,
            gamma=0.75,
            kappa=2.0,
            theta=0.04,
            sigma_v=0.2,
            rho=-0.5,
            r=0.0,
            q=0.0,
        )
        s_paths, v_paths = process.simulate(
            t_max=0.5, num_steps=50, num_paths=2000, seed=42, return_variance=True
        )
        assert s_paths.shape == (2000, 51)
        assert v_paths.shape == (2000, 51)
        assert np.all(v_paths >= 0.0)
        assert np.all(s_paths >= 0.0)

        # Average spot at t=0.5 should be close to 100
        mean_s_t = np.mean(s_paths[:, -1])
        assert mean_s_t == pytest.approx(100.0, rel=0.05)

        # Average variance at t=0.5 should match analytical expected variance
        mean_v_t = np.mean(v_paths[:, -1])
        analytic_v_t = process.expected_variance(0.5)
        assert mean_v_t == pytest.approx(analytic_v_t, rel=0.08)
