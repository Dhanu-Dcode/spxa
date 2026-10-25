"""Exactness tests for Wright-Fisher Gene Frequency Diffusion Process."""

import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond.wright_fisher import (
    WrightFisherProcess,
    WrightFisher,
    GeneFrequencyDiffusion,
)


@pytest.mark.exactness
class TestWrightFisherExactness:
    def test_wf_exactness_level_and_aliases(self):
        process = WrightFisherProcess(alpha=1.0, beta=1.5, sigma=0.5, x0=0.4)
        assert process.exactness_level == ExactnessLevel.MOMENT_PROPAGATION
        assert WrightFisher is WrightFisherProcess
        assert GeneFrequencyDiffusion is WrightFisherProcess

    def test_wf_triplet_raises_exactness_error(self):
        process = WrightFisherProcess(alpha=1.0, beta=1.0, sigma=0.5)
        with pytest.raises(ExactnessError):
            _ = process.triplet

    def test_wf_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="Mutation rates"):
            WrightFisherProcess(alpha=-1.0, beta=1.0, sigma=0.5)
        with pytest.raises(ValueError, match="Initial frequency"):
            WrightFisherProcess(alpha=1.0, beta=1.0, sigma=0.5, x0=1.5)

    def test_wf_beta_shapes_and_boundary(self):
        process = WrightFisherProcess(alpha=0.5, beta=0.5, sigma=0.5, x0=0.5)
        # 2 * 0.5 / 0.25 = 4.0
        assert process.beta_shape_a == pytest.approx(4.0, rel=1e-5)
        assert process.beta_shape_b == pytest.approx(4.0, rel=1e-5)
        assert process.are_boundaries_unattainable is True

    def test_wf_moments_and_stationary_limits(self):
        process = WrightFisherProcess(alpha=2.0, beta=3.0, sigma=0.4, x0=0.2)
        assert process.mean(0.0) == pytest.approx(0.2, rel=1e-7)

        # Expected equilibrium is alpha / (alpha + beta) = 2 / 5 = 0.4
        assert process.stationary_mean() == pytest.approx(0.4, rel=1e-7)
        assert process.mean(50.0) == pytest.approx(0.4, rel=1e-4)

    def test_wf_simulation_boundaries_and_mean(self):
        process = WrightFisherProcess(alpha=1.5, beta=1.5, sigma=0.3, x0=0.5)
        paths = process.simulate(t_max=1.0, num_steps=40, num_paths=2000, seed=42)
        assert paths.shape == (2000, 41)

        # All trajectories stay in [0, 1]
        assert np.all(paths >= 0.0)
        assert np.all(paths <= 1.0)

        sim_mean_1y = np.mean(paths[:, -1])
        analytic_mean_1y = process.mean(1.0)
        assert sim_mean_1y == pytest.approx(analytic_mean_1y, abs=0.03)
