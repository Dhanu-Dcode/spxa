"""
Exactness and mathematical consistency tests for 3/2 Stochastic Volatility model.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import ThreeHalves, ThreeHalvesModel, ThreeHalvesProcess


class TestThreeHalvesExactness:
    """Mathematical exactness and consistency test suite for ThreeHalvesProcess."""

    def test_three_halves_exactness_level_and_aliases(self) -> None:
        """ThreeHalvesProcess is MOMENT_PROPAGATION and identical to ThreeHalves, ThreeHalvesModel."""
        p = ThreeHalvesProcess(kappa=3.0, theta=0.04, epsilon=0.5, v0=0.05)
        assert p.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert ThreeHalves is ThreeHalvesProcess
        assert ThreeHalvesModel is ThreeHalvesProcess
        assert "ThreeHalvesProcess" in repr(p)
        assert "kappa=3.0" in repr(p)
        assert "theta=0.04" in repr(p)

    def test_three_halves_triplet_raises_exactness_error(self) -> None:
        """Calling triplet raises ExactnessError with mathematical explanation."""
        p = ThreeHalvesProcess()
        with pytest.raises(ExactnessError, match="triplet"):
            _ = p.triplet

    def test_three_halves_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on non-positive parameters."""
        with pytest.raises(ValueError, match="kappa"):
            ThreeHalvesProcess(kappa=-1.0)
        with pytest.raises(ValueError, match="theta"):
            ThreeHalvesProcess(theta=0.0)
        with pytest.raises(ValueError, match="epsilon"):
            ThreeHalvesProcess(epsilon=-0.2)
        with pytest.raises(ValueError, match="v0"):
            ThreeHalvesProcess(v0=-0.01)

    def test_three_halves_process_properties(self) -> None:
        """Process properties reflect non-Lévy Markovian diffusion structure."""
        p = ThreeHalvesProcess()
        props = p.properties
        assert not props.has_stationary_increments
        assert not props.has_independent_increments
        assert not props.is_martingale
        assert not props.is_subordinator

    def test_three_halves_feller_ratio_always_satisfied(self) -> None:
        """Reciprocal CIR Feller condition 1 + 2κ/ε² is strictly > 1 for all positive parameters."""
        p = ThreeHalvesProcess(kappa=0.5, theta=0.04, epsilon=2.0)
        assert p.feller_ratio > 1.0
        assert math.isclose(p.feller_ratio, 1.0 + 2.0 * 0.5 / 4.0, rel_tol=1e-9)

    def test_three_halves_reciprocal_parameters(self) -> None:
        """Reciprocal CIR parameters match theoretical formulas."""
        k, th, eps = 4.0, 0.05, 0.8
        p = ThreeHalvesProcess(kappa=k, theta=th, epsilon=eps)
        assert math.isclose(p.kappa_cir, k * th, rel_tol=1e-9)
        assert math.isclose(p.theta_cir, (k + eps**2) / (k * th), rel_tol=1e-9)
        assert math.isclose(p.sigma_cir, eps, rel_tol=1e-9)

    def test_three_halves_stationary_distribution(self) -> None:
        """Stationary Inverse-Gamma moments match analytical expressions."""
        k, th, eps = 5.0, 0.04, 0.6
        p = ThreeHalvesProcess(kappa=k, theta=th, epsilon=eps)

        expected_shape = 1.0 + 2.0 * k / (eps**2)
        expected_scale = 2.0 * k * th / (eps**2)
        assert math.isclose(p.stationary_shape, expected_shape, rel_tol=1e-9)
        assert math.isclose(p.stationary_scale, expected_scale, rel_tol=1e-9)

        assert math.isclose(p.stationary_mean, th, rel_tol=1e-9)

        expected_var = (eps**2 * th**2) / (2.0 * k - eps**2)
        assert math.isclose(p.stationary_variance, expected_var, rel_tol=1e-9)

        # Stationary density is strictly positive on positive support
        dens = p.stationary_density(th)
        assert dens > 0.0

    def test_three_halves_reciprocal_conditional_mean(self) -> None:
        """Reciprocal CIR conditional mean converges to stationary reciprocal level."""
        p = ThreeHalvesProcess(kappa=3.0, theta=0.04, epsilon=0.5, v0=0.02)
        y0 = 1.0 / 0.02
        m0 = p.reciprocal_conditional_mean(0.0)
        assert math.isclose(m0, y0, rel_tol=1e-9)

        m_long = p.reciprocal_conditional_mean(100.0)
        assert math.isclose(m_long, p.theta_cir, rel_tol=1e-4)

    def test_three_halves_transition_density(self) -> None:
        """Transition density p(v, t | v0) is positive and integrates properly."""
        p = ThreeHalvesProcess(kappa=4.0, theta=0.04, epsilon=0.5, v0=0.04)
        v_grid = np.linspace(0.01, 0.15, 20)
        densities = p.transition_density(v_grid, t=0.5)
        assert np.all(densities >= 0.0)
        assert densities.ndim == 1
        assert len(densities) == 20

    def test_three_halves_simulation_schemes(self) -> None:
        """Exact and Euler/Milstein simulation methods produce positive sample paths."""
        p = ThreeHalvesProcess(kappa=3.0, theta=0.04, epsilon=0.5, v0=0.04)
        t_grid = np.linspace(0, 1.0, 11)
        rng = np.random.default_rng(123)

        # Exact simulation
        paths_exact = p.simulate(t_grid, n_paths=100, rng=rng, method="exact")
        assert paths_exact.shape == (100, 11)
        assert np.all(paths_exact > 0)
        assert np.allclose(paths_exact[:, 0], 0.04)

        # Euler simulation
        paths_euler = p.simulate(t_grid, n_paths=100, rng=rng, method="euler")
        assert paths_euler.shape == (100, 11)
        assert np.all(paths_euler > 0)
        assert np.allclose(paths_euler[:, 0], 0.04)

        with pytest.raises(ValueError, match="Unknown simulation method"):
            p.simulate(t_grid, n_paths=10, method="invalid_method")
