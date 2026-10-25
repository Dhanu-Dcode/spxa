"""Exactness tests for Quadratic Rough Heston Process."""

import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.quadratic_rough_heston import (
    QRHeston,
    QuadraticRoughHeston,
    QuadraticRoughHestonProcess,
    RoughZumbach,
)


class TestQuadraticRoughHestonExactness:
    def test_qrh_exactness_level_and_aliases(self):
        process = QuadraticRoughHestonProcess(hurst=0.1, lam=1.2, nu=0.3, a=1.0, b=0.0, c=0.01)
        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert QuadraticRoughHeston is QuadraticRoughHestonProcess
        assert QRHeston is QuadraticRoughHestonProcess
        assert RoughZumbach is QuadraticRoughHestonProcess

    def test_qrh_triplet_raises_exactness_error(self):
        process = QuadraticRoughHestonProcess()
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_qrh_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="Hurst parameter"):
            QuadraticRoughHestonProcess(hurst=0.6)
        with pytest.raises(ValueError, match="Mean-reversion parameter"):
            QuadraticRoughHestonProcess(lam=-1.0)
        with pytest.raises(ValueError, match="Vol-of-vol"):
            QuadraticRoughHestonProcess(nu=-0.1)
        with pytest.raises(ValueError, match="Quadratic scale parameter"):
            QuadraticRoughHestonProcess(a=-1.0)
        with pytest.raises(ValueError, match="Variance floor"):
            QuadraticRoughHestonProcess(c=-0.05)

    def test_qrh_initial_variance_and_zumbach_factor(self):
        process = QuadraticRoughHestonProcess(
            hurst=0.15, lam=1.0, nu=0.25, a=2.0, b=0.05, c=0.02, z0=0.15
        )
        # V0 = 2 * (0.15 - 0.05)^2 + 0.02 = 2 * 0.01 + 0.02 = 0.04
        assert process.initial_variance == pytest.approx(0.04, rel=1e-7)

        # Zumbach factor dV/dZ = 2 * a * (z0 - b) = 4 * 0.10 = 0.4
        assert process.zumbach_feedback_factor == pytest.approx(0.40, rel=1e-7)

        # Variance function mapping
        assert process.variance_from_z(0.05) == pytest.approx(0.02, rel=1e-7)
        assert process.variance_from_z(0.25) == pytest.approx(2.0 * 0.04 + 0.02, rel=1e-7)

    def test_qrh_asymptotic_skew_scaling(self):
        process = QuadraticRoughHestonProcess(
            hurst=0.1, lam=1.5, nu=0.3, a=1.0, b=0.0, c=0.01, z0=0.15, rho=-0.7
        )
        # For H=0.1, skew scales as tau^{0.1 - 0.5} = tau^{-0.4} (blow-up at 0)
        skew_short = process.asymptotic_atm_skew(0.01)
        skew_long = process.asymptotic_atm_skew(1.0)
        assert skew_short > skew_long
        # Ratio should be (0.01 / 1.0)^-0.4 = 100^0.4 ~ 6.30957
        expected_ratio = (0.01 / 1.0) ** (-0.4)
        assert skew_short / skew_long == pytest.approx(expected_ratio, rel=1e-5)

    def test_qrh_simulation_paths_and_positivity(self):
        process = QuadraticRoughHestonProcess(
            hurst=0.1, lam=1.0, nu=0.2, a=1.0, b=0.0, c=0.01, z0=0.1, s0=100.0, rho=-0.5
        )
        s_paths, z_paths, v_paths = process.simulate(
            t_max=0.5, num_steps=40, num_paths=1500, seed=42, return_variance=True
        )
        assert s_paths.shape == (1500, 41)
        assert z_paths.shape == (1500, 41)
        assert v_paths.shape == (1500, 41)

        # Spot price and variance strictly positive
        assert np.all(s_paths > 0.0)
        assert np.all(v_paths >= process.c)

        # Average spot price under r=0, q=0 stays close to S0
        mc_s_mean = np.mean(s_paths[:, -1])
        assert mc_s_mean == pytest.approx(100.0, rel=0.05)
