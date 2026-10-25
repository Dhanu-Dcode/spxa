"""
Exactness and mathematical consistency tests for Bilinear Diffusion process.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import AffineDiffusion, BilinearDiffusion, BilinearDiffusionProcess


class TestBilinearDiffusionExactness:
    """Mathematical exactness and consistency test suite for BilinearDiffusion."""

    def test_bilinear_exactness_level_and_aliases(self) -> None:
        """BilinearDiffusion is MOMENT_PROPAGATION and identical to BilinearDiffusionProcess, AffineDiffusion."""
        p = BilinearDiffusion(a=0.5, b=-1.0, c=0.2, d=0.3, x0=0.5)
        assert p.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert BilinearDiffusionProcess is BilinearDiffusion
        assert AffineDiffusion is BilinearDiffusion
        assert "BilinearDiffusion" in repr(p)
        assert "a=0.5" in repr(p)
        assert "b=-1.0" in repr(p)

    def test_bilinear_triplet_raises_exactness_error(self) -> None:
        """Calling triplet raises ExactnessError with mathematical explanation."""
        p = BilinearDiffusion()
        with pytest.raises(ExactnessError, match="triplet"):
            _ = p.triplet

    def test_bilinear_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on invalid inputs."""
        with pytest.raises(ValueError, match="Slope 'b'"):
            BilinearDiffusion(b=0.5)  # b must be < 0
        with pytest.raises(ValueError, match="Slope 'b'"):
            BilinearDiffusion(b=0.0)
        with pytest.raises(ValueError, match="Additive diffusion 'c'"):
            BilinearDiffusion(c=-0.1)
        with pytest.raises(ValueError, match="Multiplicative diffusion 'd'"):
            BilinearDiffusion(d=-0.1)
        with pytest.raises(ValueError, match="At least one of c or d"):
            BilinearDiffusion(c=0.0, d=0.0)
        with pytest.raises(ValueError, match="Initial value x0"):
            BilinearDiffusion(a=0.5, b=-1.0, c=0.2, d=0.5, x0=-1.0)  # lower bound is -0.4

    def test_bilinear_process_properties(self) -> None:
        """Process properties reflect non-Lévy Markovian diffusion structure."""
        p = BilinearDiffusion()
        props = p.properties
        assert not props.has_stationary_increments
        assert not props.has_independent_increments
        assert not props.is_martingale
        assert not props.is_subordinator

    def test_bilinear_moments_and_stationary_convergence(self) -> None:
        """Conditional moments match analytical formulas and converge to stationary law."""
        a, b, c, d, x0 = 0.6, -1.2, 0.2, 0.4, 0.8
        p = BilinearDiffusion(a=a, b=b, c=c, d=d, x0=x0)

        # Long-term mean theta = -a / b
        theta = -a / b
        assert math.isclose(p.theta, theta, rel_tol=1e-9)
        assert math.isclose(p.stationary_mean, theta, rel_tol=1e-9)

        # At t=0
        assert math.isclose(p.conditional_mean(0.0), x0, rel_tol=1e-9)
        assert math.isclose(p.conditional_variance(0.0), 0.0, abs_tol=1e-12)

        # Long term convergence
        t_long = 25.0
        assert math.isclose(p.conditional_mean(t_long), theta, rel_tol=1e-5)

        # Stationary variance check
        stat_var = p.stationary_variance
        assert math.isclose(p.conditional_variance(t_long), stat_var, rel_tol=1e-3)

    def test_bilinear_simulation(self) -> None:
        """Milstein simulation respects state boundary and produces correct output shape."""
        p = BilinearDiffusion(a=0.5, b=-1.0, c=0.2, d=0.3, x0=0.5)
        t_grid = np.linspace(0, 1.0, 11)
        rng = np.random.default_rng(42)

        paths = p.simulate(t_grid, n_paths=100, rng=rng)
        assert paths.shape == (100, 11)
        assert np.allclose(paths[:, 0], 0.5)
        assert np.all(paths >= p.lower_bound)
