"""
Exactness and mathematical consistency tests for Nelson (1990) continuous-time GARCH diffusion.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import ContinuousGARCH, GARCHDiffusion, GARCHDiffusionProcess


class TestGARCHDiffusionExactness:
    """Mathematical exactness and consistency test suite for GARCHDiffusion."""

    def test_garch_exactness_level_and_aliases(self) -> None:
        """GARCHDiffusion is MOMENT_PROPAGATION and identical to ContinuousGARCH, GARCHDiffusionProcess."""
        p = GARCHDiffusion(kappa=2.5, theta=0.04, sigma=0.3, v0=0.05)
        assert p.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert ContinuousGARCH is GARCHDiffusion
        assert GARCHDiffusionProcess is GARCHDiffusion
        assert "GARCHDiffusion" in repr(p)
        assert "kappa=2.5" in repr(p)
        assert "theta=0.04" in repr(p)

    def test_garch_triplet_raises_exactness_error(self) -> None:
        """Calling triplet raises ExactnessError with mathematical explanation."""
        p = GARCHDiffusion()
        with pytest.raises(ExactnessError, match="triplet"):
            _ = p.triplet

    def test_garch_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on non-positive parameters."""
        with pytest.raises(ValueError, match="kappa"):
            GARCHDiffusion(kappa=-1.0)
        with pytest.raises(ValueError, match="theta"):
            GARCHDiffusion(theta=0.0)
        with pytest.raises(ValueError, match="sigma"):
            GARCHDiffusion(sigma=-0.2)
        with pytest.raises(ValueError, match="v0"):
            GARCHDiffusion(v0=-0.01)

    def test_garch_process_properties(self) -> None:
        """Process properties reflect non-Lévy Markovian diffusion structure."""
        p = GARCHDiffusion()
        props = p.properties
        assert not props.has_stationary_increments
        assert not props.has_independent_increments
        assert not props.is_martingale
        assert not props.is_subordinator

    def test_garch_stationary_distribution(self) -> None:
        """Stationary Inverse-Gamma parameters and moments match analytical formulas."""
        k, th, sig = 3.0, 0.04, 0.35
        p = GARCHDiffusion(kappa=k, theta=th, sigma=sig)

        expected_shape = 1.0 + 2.0 * k / (sig**2)
        expected_scale = 2.0 * k * th / (sig**2)
        assert math.isclose(p.stationary_shape, expected_shape, rel_tol=1e-9)
        assert math.isclose(p.stationary_scale, expected_scale, rel_tol=1e-9)

        assert math.isclose(p.stationary_mean, th, rel_tol=1e-9)

        expected_var = (sig**2 * th**2) / (2.0 * k - sig**2)
        assert math.isclose(p.stationary_variance, expected_var, rel_tol=1e-9)

        # Density check
        dens = p.stationary_density(th)
        assert dens > 0.0

    def test_garch_conditional_moments_evolution(self) -> None:
        """Conditional mean and variance evolve smoothly and converge to stationary values."""
        k, th, sig, v0 = 2.0, 0.04, 0.25, 0.08
        p = GARCHDiffusion(kappa=k, theta=th, sigma=sig, v0=v0)

        # At t=0
        assert math.isclose(p.conditional_mean(0.0), v0, rel_tol=1e-9)
        assert math.isclose(p.conditional_variance(0.0), 0.0, abs_tol=1e-12)

        # Intermediate time
        m1 = p.conditional_mean(1.0)
        assert th < m1 < v0

        # Long time convergence
        t_long = 30.0
        assert math.isclose(p.conditional_mean(t_long), th, rel_tol=1e-5)
        assert math.isclose(p.conditional_variance(t_long), p.stationary_variance, rel_tol=1e-4)

    def test_garch_simulation_positivity(self) -> None:
        """Log-Euler simulation produces strictly positive variance sample paths."""
        p = GARCHDiffusion(kappa=3.0, theta=0.04, sigma=0.4, v0=0.04)
        t_grid = np.linspace(0, 1.0, 11)
        rng = np.random.default_rng(99)

        paths = p.simulate(t_grid, n_paths=100, rng=rng)
        assert paths.shape == (100, 11)
        assert np.allclose(paths[:, 0], 0.04)
        assert np.all(paths > 0)
