"""Exactness tests for Quadratic Pearson / Wong Diffusion Process."""

import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.quadratic_diffusion import (
    QuadraticDiffusion,
    QuadraticPearsonDiffusion,
    WongDiffusion,
)


@pytest.mark.exactness
class TestQuadraticDiffusionExactness:
    def test_qd_exactness_level_and_aliases(self):
        process = QuadraticDiffusion(a=1.0, b=-2.0, c=0.5, d=0.1, e=0.2, x0=0.0)
        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert QuadraticPearsonDiffusion is QuadraticDiffusion
        assert WongDiffusion is QuadraticDiffusion

    def test_qd_triplet_raises_exactness_error(self):
        process = QuadraticDiffusion(a=1.0, b=-2.0, c=0.5, d=0.0, e=0.1)
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_qd_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="b < 0"):
            QuadraticDiffusion(a=1.0, b=1.0, c=0.5, d=0.0, e=0.1)
        with pytest.raises(ValueError, match="e must be non-negative"):
            QuadraticDiffusion(a=1.0, b=-2.0, c=0.5, d=0.0, e=-0.1)
        with pytest.raises(ValueError, match="must be positive at x0"):
            QuadraticDiffusion(a=1.0, b=-2.0, c=-1.0, d=0.0, e=0.0, x0=0.0)

    def test_qd_moment_hierarchy_and_stationary_limits(self):
        # a = 2.0, b = -2.0 -> mu_inf = 1.0
        # c = 1.0, d = 0.0, e = 0.5 -> lam2 = 2*(-2) + 0.5 = -3.5 < 0
        process = QuadraticDiffusion(a=2.0, b=-2.0, c=1.0, d=0.0, e=0.5, x0=0.0)

        # Stationary mean: -2 / (-2) = 1.0
        assert process.stationary_mean() == pytest.approx(1.0, rel=1e-7)

        # num = 2*a*mu + d*mu + c = 2*2*1 + 0 + 1 = 5.0
        # lam2 = -3.5
        # second moment: -5.0 / (-3.5) = 10 / 7 = 1.4285714...
        assert process.stationary_second_moment() == pytest.approx(10.0 / 7.0, rel=1e-5)

        # Stationary variance = 10/7 - 1 = 3/7 = 0.4285714...
        assert process.stationary_variance() == pytest.approx(3.0 / 7.0, rel=1e-5)

        # Convergence as t -> inf
        assert process.mean(50.0) == pytest.approx(1.0, rel=1e-4)
        assert process.variance(50.0) == pytest.approx(3.0 / 7.0, rel=1e-4)

    def test_qd_simulation_convergence(self):
        process = QuadraticDiffusion(a=1.5, b=-1.5, c=0.8, d=0.0, e=0.2, x0=0.0)
        paths = process.simulate(t_max=2.0, num_steps=40, num_paths=2000, seed=42)
        assert paths.shape == (2000, 41)

        sim_mean_2y = np.mean(paths[:, -1])
        analytic_mean_2y = process.mean(2.0)
        assert sim_mean_2y == pytest.approx(analytic_mean_2y, rel=0.05)
