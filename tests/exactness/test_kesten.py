"""Exactness tests for Kesten Multiplicative Noise Diffusion Process."""

import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.kesten import (
    KestenProcess,
    KestenDiffusion,
    MultiplicativeNoiseDiffusion,
)


@pytest.mark.exactness
class TestKestenExactness:
    def test_kesten_exactness_level_and_aliases(self):
        process = KestenProcess(a=1.0, b=2.0, sigma=0.5, x0=1.0)
        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert KestenDiffusion is KestenProcess
        assert MultiplicativeNoiseDiffusion is KestenProcess

    def test_kesten_triplet_raises_exactness_error(self):
        process = KestenProcess(a=1.0, b=2.0, sigma=0.5)
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_kesten_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="parameter a"):
            KestenProcess(a=-1.0, b=2.0, sigma=0.5)
        with pytest.raises(ValueError, match="parameter b"):
            KestenProcess(a=1.0, b=-2.0, sigma=0.5)
        with pytest.raises(ValueError, match="Initial state x0"):
            KestenProcess(a=1.0, b=2.0, sigma=0.5, x0=-0.5)

    def test_kesten_power_law_index_and_moments(self):
        # a = 2.0, b = 1.0, sigma = 0.5 -> 2*b/sig^2 = 2.0 / 0.25 = 8.0
        # tail index alpha = 9.0, beta = 2*2 / 0.25 = 16.0
        process = KestenProcess(a=2.0, b=1.0, sigma=0.5, x0=1.0)
        assert process.pareto_tail_index == pytest.approx(9.0, rel=1e-5)
        assert process.invgamma_scale == pytest.approx(16.0, rel=1e-5)

        # Stationary mean: a / b = 2.0
        assert process.stationary_mean() == pytest.approx(2.0, rel=1e-5)
        # Stationary variance: (4.0 * 0.25) / (1.0 * (2 - 0.25)) = 1.0 / 1.75 = 0.571428...
        expected_var = (4.0 * 0.25) / (1.75)
        assert process.stationary_variance() == pytest.approx(expected_var, rel=1e-5)

    def test_kesten_conditional_mean_convergence(self):
        process = KestenProcess(a=3.0, b=1.5, sigma=0.4, x0=0.5)
        assert process.mean(0.0) == pytest.approx(0.5, rel=1e-7)
        assert process.mean(50.0) == pytest.approx(2.0, rel=1e-4)

    def test_kesten_simulation_positivity_and_mean(self):
        process = KestenProcess(a=2.0, b=1.0, sigma=0.3, x0=1.0)
        paths = process.simulate(t_max=2.0, num_steps=40, num_paths=2000, seed=42)
        assert paths.shape == (2000, 41)
        assert np.all(paths > 0.0)

        sim_mean = np.mean(paths[:, -1])
        analytic_mean = process.mean(2.0)
        assert sim_mean == pytest.approx(analytic_mean, rel=0.05)
