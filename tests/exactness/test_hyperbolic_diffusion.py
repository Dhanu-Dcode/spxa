"""Exactness tests for Hyperbolic Diffusion Process."""

import numpy as np
import pytest
from scipy import integrate

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.hyperbolic_diffusion import (
    HyperbolicDiffusion,
    HyperbolicOU,
    HyperbolicInvariantDiffusion,
)


@pytest.mark.exactness
class TestHyperbolicDiffusionExactness:
    def test_hd_exactness_level_and_aliases(self):
        process = HyperbolicDiffusion(alpha=2.0, beta=0.5, delta=1.0, mu0=0.0, sigma=0.5)
        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert HyperbolicOU is HyperbolicDiffusion
        assert HyperbolicInvariantDiffusion is HyperbolicDiffusion

    def test_hd_triplet_raises_exactness_error(self):
        process = HyperbolicDiffusion(alpha=2.0, beta=0.0, delta=1.0)
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_hd_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="positive"):
            HyperbolicDiffusion(alpha=-2.0, beta=0.0, delta=1.0)
        with pytest.raises(ValueError, match=r"\|beta\| < alpha"):
            HyperbolicDiffusion(alpha=2.0, beta=2.5, delta=1.0)

    def test_hd_drift_boundedness_and_centering(self):
        process = HyperbolicDiffusion(alpha=3.0, beta=0.0, delta=1.0, mu0=0.0, sigma=1.0)
        # At x=0, drift is zero when beta=0 and mu0=0
        assert process.drift(0.0) == pytest.approx(0.0, abs=1e-12)

        # As x -> +inf, drift -> 0.5 * sigma^2 * (beta - alpha) = 0.5 * (0 - 3) = -1.5
        assert process.drift(1e5) == pytest.approx(-1.5, rel=1e-4)
        # As x -> -inf, drift -> 0.5 * sigma^2 * (beta + alpha) = 0.5 * (0 + 3) = +1.5
        assert process.drift(-1e5) == pytest.approx(1.5, rel=1e-4)

    def test_hd_stationary_density_integration(self):
        process = HyperbolicDiffusion(alpha=2.5, beta=0.8, delta=1.2, mu0=0.5, sigma=0.8)
        # Integral of PDF over R must equal 1.0
        integral, _ = integrate.quad(process.stationary_pdf, -30.0, 30.0)
        assert integral == pytest.approx(1.0, rel=1e-4)

        # Mean from PDF matches stationary_mean formula
        mean_int, _ = integrate.quad(lambda x: x * process.stationary_pdf(x), -30.0, 30.0)
        assert mean_int == pytest.approx(process.stationary_mean(), rel=1e-3)

    def test_hd_simulation_mean_reversion(self):
        process = HyperbolicDiffusion(alpha=2.0, beta=0.0, delta=1.0, mu0=0.0, sigma=0.5, x0=3.0)
        paths = process.simulate(t_max=5.0, num_steps=50, num_paths=1500, seed=42)
        assert paths.shape == (1500, 51)

        # Pulls from initial state x0=3.0 toward stationary mean 0.0
        sim_final_mean = np.mean(paths[:, -1])
        assert sim_final_mean < process.x0 - 0.8  # Significant reversion toward 0
