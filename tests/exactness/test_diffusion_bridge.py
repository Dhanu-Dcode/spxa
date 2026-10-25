"""
Exactness and mathematical consistency tests for General Diffusion Bridge.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import DiffusionBridge, DoobBridge, GeneralDiffusionBridge


class TestGeneralDiffusionBridgeExactness:
    """Mathematical exactness and consistency test suite for GeneralDiffusionBridge."""

    def test_bridge_exactness_level_and_aliases(self) -> None:
        """GeneralDiffusionBridge is MOMENT_PROPAGATION and identical to DiffusionBridge, DoobBridge."""
        p = GeneralDiffusionBridge(a=0.0, b=2.0, T=1.0, model="brownian", sigma=0.5)
        assert p.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert DiffusionBridge is GeneralDiffusionBridge
        assert DoobBridge is GeneralDiffusionBridge
        assert "GeneralDiffusionBridge" in repr(p)
        assert "brownian" in repr(p)

    def test_bridge_triplet_raises_exactness_error(self) -> None:
        """Calling triplet raises ExactnessError with mathematical explanation."""
        p = GeneralDiffusionBridge()
        with pytest.raises(ExactnessError, match="triplet"):
            _ = p.triplet

    def test_bridge_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on invalid inputs."""
        with pytest.raises(ValueError, match="Terminal horizon T"):
            GeneralDiffusionBridge(T=-1.0)
        with pytest.raises(ValueError, match="sigma"):
            GeneralDiffusionBridge(sigma=-0.2)
        with pytest.raises(ValueError, match="Unknown bridge model"):
            GeneralDiffusionBridge(model="invalid_model")
        with pytest.raises(ValueError, match="GBM bridge"):
            GeneralDiffusionBridge(model="gbm", a=-1.0, b=2.0)
        with pytest.raises(ValueError, match="kappa"):
            GeneralDiffusionBridge(model="ou", kappa=-1.0)

    def test_bridge_process_properties(self) -> None:
        """Process properties reflect conditioned non-homogeneous diffusion structure."""
        p = GeneralDiffusionBridge()
        props = p.properties
        assert not props.has_stationary_increments
        assert not props.has_independent_increments
        assert not props.is_martingale
        assert props.has_finite_mean

    def test_bridge_brownian_moments_and_variance(self) -> None:
        """Brownian bridge conditional mean is linear and variance is parabolic."""
        a, b, T, sig = 1.0, 5.0, 2.0, 0.4
        p = GeneralDiffusionBridge(a=a, b=b, T=T, model="brownian", sigma=sig)

        assert math.isclose(p.conditional_mean(0.0), a, rel_tol=1e-9)
        assert math.isclose(p.conditional_mean(T), b, rel_tol=1e-9)
        assert math.isclose(p.conditional_variance(0.0), 0.0, abs_tol=1e-12)
        assert math.isclose(p.conditional_variance(T), 0.0, abs_tol=1e-12)

        # Midpoint
        t_mid = 1.0
        assert math.isclose(p.conditional_mean(t_mid), (a + b) / 2.0, rel_tol=1e-9)
        expected_var = (sig**2) * t_mid * (T - t_mid) / T
        assert math.isclose(p.conditional_variance(t_mid), expected_var, rel_tol=1e-9)

    def test_bridge_ou_moments(self) -> None:
        """OU bridge conditional mean matches hyperbolic sine interpolation."""
        a, b, T, sig, k, th = 0.5, 2.5, 1.5, 0.3, 1.2, 1.0
        p = GeneralDiffusionBridge(a=a, b=b, T=T, model="ou", sigma=sig, kappa=k, theta=th)

        assert math.isclose(p.conditional_mean(0.0), a, rel_tol=1e-9)
        assert math.isclose(p.conditional_mean(T), b, rel_tol=1e-9)
        assert math.isclose(p.conditional_variance(0.0), 0.0, abs_tol=1e-12)
        assert math.isclose(p.conditional_variance(T), 0.0, abs_tol=1e-12)

        t_mid = 0.75
        assert p.conditional_variance(t_mid) > 0.0

    def test_bridge_simulation_pinning(self) -> None:
        """Simulated sample paths are strictly pinned at both boundaries."""
        a, b, T = 2.0, 8.0, 2.0
        p = GeneralDiffusionBridge(a=a, b=b, T=T, model="brownian", sigma=0.5)
        t_grid = np.linspace(0, T, 11)
        rng = np.random.default_rng(99)

        paths = p.simulate(t_grid, n_paths=100, rng=rng)
        assert paths.shape == (100, 11)
        assert np.allclose(paths[:, 0], a)
        assert np.allclose(paths[:, -1], b)
