"""
Exactness and mathematical consistency tests for Rough Heston model.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import RoughHeston, RoughHestonProcess, RoughVolatility


class TestRoughHestonExactness:
    """Mathematical exactness and consistency test suite for RoughHestonProcess."""

    def test_rough_heston_exactness_level_and_aliases(self) -> None:
        """RoughHestonProcess is MOMENT_PROPAGATION and identical to RoughHeston, RoughVolatility."""
        p = RoughHestonProcess(hurst=0.1, lam=1.2, theta=0.04, nu=0.3, rho=-0.6)
        assert p.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert RoughHeston is RoughHestonProcess
        assert RoughVolatility is RoughHestonProcess
        assert "RoughHestonProcess" in repr(p)
        assert "H=0.1" in repr(p)

    def test_rough_heston_triplet_raises_exactness_error(self) -> None:
        """Calling triplet raises ExactnessError with mathematical explanation."""
        p = RoughHestonProcess()
        with pytest.raises(ExactnessError, match="triplet"):
            _ = p.triplet

    def test_rough_heston_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on invalid Hurst or model parameters."""
        with pytest.raises(ValueError, match="Hurst parameter H"):
            RoughHestonProcess(hurst=-0.1)
        with pytest.raises(ValueError, match="Hurst parameter H"):
            RoughHestonProcess(hurst=0.6)  # must be in (0, 0.5)
        with pytest.raises(ValueError, match="lam"):
            RoughHestonProcess(lam=-1.0)
        with pytest.raises(ValueError, match="theta"):
            RoughHestonProcess(theta=0.0)
        with pytest.raises(ValueError, match="nu"):
            RoughHestonProcess(nu=-0.1)
        with pytest.raises(ValueError, match="rho"):
            RoughHestonProcess(rho=1.5)

    def test_rough_heston_process_properties(self) -> None:
        """Process properties reflect fractional Hurst self-similarity."""
        p = RoughHestonProcess(hurst=0.15)
        props = p.properties
        assert not props.has_stationary_increments
        assert not props.has_independent_increments
        assert not props.is_martingale
        assert props.self_similarity_index == 0.15
        assert props.hurst_index == 0.15

    def test_rough_heston_mittag_leffler_and_mean_variance(self) -> None:
        """Mittag-Leffler expansion evaluates properly and expected variance starts at v0."""
        p = RoughHestonProcess(hurst=0.1, lam=1.0, theta=0.04, nu=0.3, rho=-0.5, v0=0.08)
        assert math.isclose(p.mittag_leffler(0.0), 1.0, rel_tol=1e-9)
        assert math.isclose(p.expected_variance(0.0), 0.08, rel_tol=1e-9)

        # Variance decays toward theta
        ev1 = p.expected_variance(1.0)
        assert 0.04 < ev1 < 0.08

    def test_rough_heston_asymptotic_skew_power_law(self) -> None:
        """Asymptotic ATM skew exhibits T^{H - 0.5} power-law explosion as T -> 0."""
        p = RoughHestonProcess(hurst=0.1, lam=1.5, theta=0.04, nu=0.3, rho=-0.7)
        skew_short = p.asymptotic_atm_skew(0.01)
        skew_medium = p.asymptotic_atm_skew(0.25)
        skew_long = p.asymptotic_atm_skew(1.0)

        # Skew must blow up as T -> 0 (H - 0.5 = -0.4 < 0)
        assert skew_short > skew_medium > skew_long > 0.0

    def test_rough_heston_simulation(self) -> None:
        """Vectorized fractional convolution simulation produces positive price and variance paths."""
        p = RoughHestonProcess(
            hurst=0.15, lam=1.2, theta=0.04, nu=0.25, rho=-0.5, s0=100.0, v0=0.04
        )
        t_grid = np.linspace(0, 1.0, 11)
        rng = np.random.default_rng(42)

        s_paths, v_paths = p.simulate(t_grid, n_paths=40, rng=rng, return_vol=True)
        assert s_paths.shape == (40, 11)
        assert v_paths.shape == (40, 11)
        assert np.allclose(s_paths[:, 0], 100.0)
        assert np.allclose(v_paths[:, 0], 0.04)
        assert np.all(s_paths > 0)
        assert np.all(v_paths > 0)
