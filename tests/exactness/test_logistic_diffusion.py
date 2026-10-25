"""Exactness tests for Stochastic Logistic (Verhulst) Diffusion Process."""

import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.logistic_diffusion import (
    LogisticDiffusion,
    LogisticGrowthProcess,
    VerhulstDiffusion,
)


@pytest.mark.exactness
class TestLogisticDiffusionExactness:
    def test_logistic_exactness_level_and_aliases(self):
        process = LogisticDiffusion(r=1.5, k=100.0, sigma=0.4, x0=50.0)
        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert LogisticGrowthProcess is LogisticDiffusion
        assert VerhulstDiffusion is LogisticDiffusion

    def test_logistic_triplet_raises_exactness_error(self):
        process = LogisticDiffusion(r=1.0, k=50.0, sigma=0.2)
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_logistic_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="Growth rate r"):
            LogisticDiffusion(r=-1.0, k=50.0, sigma=0.2)
        with pytest.raises(ValueError, match="Carrying capacity"):
            LogisticDiffusion(r=1.0, k=-50.0, sigma=0.2)
        with pytest.raises(ValueError, match="Initial population"):
            LogisticDiffusion(r=1.0, k=50.0, sigma=0.2, x0=-5.0)

    def test_logistic_gamma_stationary_distribution(self):
        # r = 1.0, k = 10.0, sigma = 0.5 -> 2*r/sig^2 = 8.0
        # alpha = 7.0, beta = 8.0 / 10 = 0.8
        process = LogisticDiffusion(r=1.0, k=10.0, sigma=0.5, x0=10.0)
        assert process.is_persistent is True
        assert process.gamma_shape == pytest.approx(7.0, rel=1e-5)
        assert process.gamma_rate == pytest.approx(0.8, rel=1e-5)

        # Expected equilibrium: 10 * (1 - 0.125) = 8.75
        assert process.stationary_mean() == pytest.approx(8.75, rel=1e-5)
        # Variance: 7.0 / (0.8^2) = 10.9375
        assert process.stationary_variance() == pytest.approx(10.9375, rel=1e-5)

    def test_logistic_simulation_positivity_and_convergence(self):
        process = LogisticDiffusion(r=2.0, k=20.0, sigma=0.4, x0=10.0)
        paths = process.simulate(t_max=3.0, num_steps=60, num_paths=2000, seed=42)
        assert paths.shape == (2000, 61)

        # Strict positivity
        assert np.all(paths > 0.0)

        # Convergence towards stationary mean: 20 * (1 - 0.5 * 0.16 / 2) = 20 * (1 - 0.04) = 19.2
        sim_final_mean = np.mean(paths[:, -1])
        assert sim_final_mean == pytest.approx(process.stationary_mean(), rel=0.05)
