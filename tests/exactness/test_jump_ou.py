"""Exactness tests for Mean-Reverting Ornstein-Uhlenbeck with Jumps (Jump-OU)."""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.jump_ou import (
    JumpOUProcess,
    JumpOU,
    EnergyOU,
)


@pytest.mark.exactness
class TestJumpOUExactness:
    def test_jump_ou_exactness_level_and_aliases(self):
        process = JumpOUProcess(k=1.2, theta=2.0, sigma=0.5, lambda_j=1.0, mu_j=0.2, sigma_j=0.3, x0=1.0)
        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert JumpOU is JumpOUProcess
        assert EnergyOU is JumpOUProcess

    def test_jump_ou_triplet_raises_exactness_error(self):
        process = JumpOUProcess(k=1.2, theta=2.0, sigma=0.5, lambda_j=1.0, mu_j=0.2, sigma_j=0.3)
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_jump_ou_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="speed k"):
            JumpOUProcess(k=-1.0, theta=2.0, sigma=0.5, lambda_j=1.0, mu_j=0.2, sigma_j=0.3)
        with pytest.raises(ValueError, match="[Vv]olatility sigma"):
            JumpOUProcess(k=1.0, theta=2.0, sigma=-0.5, lambda_j=1.0, mu_j=0.2, sigma_j=0.3)

    def test_jump_ou_moments_and_stationary_limits(self):
        process = JumpOUProcess(k=1.5, theta=2.0, sigma=0.4, lambda_j=2.0, mu_j=0.3, sigma_j=0.2, x0=1.0)
        assert process.mean(0.0) == pytest.approx(1.0, rel=1e-7)
        assert process.variance(0.0) == pytest.approx(0.0, abs=1e-12)

        t_large = 40.0
        assert process.mean(t_large) == pytest.approx(process.stationary_mean(), rel=1e-3)
        assert process.variance(t_large) == pytest.approx(process.stationary_variance(), rel=1e-3)

    def test_jump_ou_characteristic_function(self):
        process = JumpOUProcess(k=1.0, theta=1.5, sigma=0.3, lambda_j=1.5, mu_j=0.2, sigma_j=0.1, x0=1.0)
        cf_0 = process.characteristic_function(0.0, t=1.0)
        assert abs(cf_0 - 1.0) < 1e-6

        for u in [0.5, 1.0, 2.0]:
            assert abs(process.characteristic_function(u, t=0.5)) <= 1.000001

    def test_jump_ou_simulation_convergence(self):
        process = JumpOUProcess(k=2.0, theta=1.0, sigma=0.3, lambda_j=2.0, mu_j=0.2, sigma_j=0.1, x0=1.0)
        paths = process.simulate(t_max=1.0, num_steps=50, num_paths=2000, seed=42)
        assert paths.shape == (2000, 51)

        sim_mean = np.mean(paths[:, -1])
        sim_var = np.var(paths[:, -1])
        analytic_mean = process.mean(1.0)
        analytic_var = process.variance(1.0)

        assert sim_mean == pytest.approx(analytic_mean, rel=0.05)
        assert sim_var == pytest.approx(analytic_var, rel=0.1)
