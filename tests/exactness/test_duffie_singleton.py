"""Exactness tests for Duffie-Singleton Hazard-Rate Default Intensity Credit Process."""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.duffie_singleton import (
    DefaultIntensityDiffusion,
    DuffieSingleton,
    DuffieSingletonProcess,
    HazardRateDiffusion,
)


class TestDuffieSingletonExactness:
    def test_ds_exactness_level_and_aliases(self):
        process = DuffieSingletonProcess(kappa=0.8, theta=0.03, sigma=0.1, lambda0=0.02)
        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert DuffieSingleton is DuffieSingletonProcess
        assert DefaultIntensityDiffusion is DuffieSingletonProcess
        assert HazardRateDiffusion is DuffieSingletonProcess

    def test_ds_triplet_raises_exactness_error(self):
        process = DuffieSingletonProcess()
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_ds_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="Hazard rate parameters"):
            DuffieSingletonProcess(kappa=-0.5)
        with pytest.raises(ValueError, match="Initial hazard intensity"):
            DuffieSingletonProcess(lambda0=-0.01)
        with pytest.raises(ValueError, match="Recovery rate must lie in"):
            DuffieSingletonProcess(recovery=1.5)

    def test_ds_feller_and_hazard_moments(self):
        process = DuffieSingletonProcess(kappa=1.0, theta=0.03, sigma=0.1, lambda0=0.02)
        # Feller ratio = 2 * 1.0 * 0.03 / 0.01 = 6.0 > 1.0
        assert process.feller_ratio == pytest.approx(6.0, rel=1e-7)
        assert process.satisfies_feller is True

        # Hazard conditional mean
        assert process.mean_hazard_rate(0.0) == pytest.approx(0.02, rel=1e-7)
        assert process.mean_hazard_rate(100.0) == pytest.approx(0.03, rel=1e-4)

        # Hazard conditional variance
        assert process.variance_hazard_rate(0.0) == 0.0
        stat_var = 0.03 * 0.01 / 2.0
        assert process.stationary_hazard_var() == pytest.approx(stat_var, rel=1e-7)
        assert process.variance_hazard_rate(100.0) == pytest.approx(stat_var, rel=1e-4)

    def test_ds_survival_and_defaultable_bond(self):
        process = DuffieSingletonProcess(
            kappa=0.7, theta=0.03, sigma=0.08, lambda0=0.02, recovery=0.40, r=0.03
        )
        # S(0, 0) = 1.0, P_def(0, 0) = 1.0
        assert process.survival_probability(0.0) == pytest.approx(1.0, rel=1e-7)
        assert process.default_probability(0.0) == pytest.approx(0.0, abs=1e-9)
        assert process.defaultable_bond(0.0) == pytest.approx(1.0, rel=1e-7)

        # Survival probability is strictly decreasing in tau
        s_1 = process.survival_probability(1.0)
        s_2 = process.survival_probability(2.0)
        assert 0.0 < s_2 < s_1 < 1.0

        # Defaultable bond price is strictly less than risk-free bond
        p_def = process.defaultable_bond(2.0)
        p_rf = math.exp(-0.03 * 2.0)
        assert 0.0 < p_def < p_rf

        # Credit spread is positive
        spread = process.credit_spread(2.0)
        assert spread > 0.0

        # CDS par spread is positive
        cds_spread = process.cds_par_spread(2.0)
        assert cds_spread > 0.0

    def test_ds_simulation_and_survival_matching(self):
        process = DuffieSingletonProcess(
            kappa=1.2, theta=0.03, sigma=0.08, lambda0=0.02, recovery=0.40, r=0.03
        )
        lambda_paths, default_times = process.simulate(
            t_max=1.0, num_steps=50, num_paths=3000, seed=42, return_default_times=True
        )
        assert lambda_paths.shape == (3000, 51)
        assert len(default_times) == 3000
        assert np.all(lambda_paths >= 0.0)

        # Monte Carlo mean hazard rate at t=1.0
        mc_lambda_1 = np.mean(lambda_paths[:, -1])
        analytic_lambda_1 = process.mean_hazard_rate(1.0)
        assert mc_lambda_1 == pytest.approx(analytic_lambda_1, rel=0.08)

        # Monte Carlo survival fraction to T=1.0 vs analytical S(0, 1.0)
        # Paths that did not default by t=1.0
        survived_mc = np.mean(np.isnan(default_times) | (default_times > 1.0))
        analytic_s = process.survival_probability(1.0)
        assert survived_mc == pytest.approx(analytic_s, abs=0.03)
