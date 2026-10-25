"""
Exactness and mathematical consistency tests for Two-Factor Hull-White (G2++) short-rate model.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import G2PlusPlus, G2PlusPlusProcess, TwoFactorHullWhite


class TestG2PlusPlusExactness:
    """Mathematical exactness and consistency test suite for G2PlusPlusProcess."""

    def test_g2pp_exactness_level_and_aliases(self) -> None:
        """G2PlusPlusProcess is MOMENT_PROPAGATION and identical to G2PlusPlus, TwoFactorHullWhite."""
        p = G2PlusPlusProcess(a=0.1, b=0.05, sigma=0.015, eta=0.01, rho=-0.3, r0=0.03)
        assert p.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert G2PlusPlus is G2PlusPlusProcess
        assert TwoFactorHullWhite is G2PlusPlusProcess
        assert "G2PlusPlusProcess" in repr(p)
        assert "a=0.1" in repr(p)
        assert "sigma=0.015" in repr(p)

    def test_g2pp_triplet_raises_exactness_error(self) -> None:
        """Calling triplet raises ExactnessError with mathematical explanation."""
        p = G2PlusPlus()
        with pytest.raises(ExactnessError, match="triplet"):
            _ = p.triplet

    def test_g2pp_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on invalid inputs."""
        with pytest.raises(ValueError, match="Speed 'a'"):
            G2PlusPlus(a=-0.1)
        with pytest.raises(ValueError, match="Speed 'b'"):
            G2PlusPlus(b=0.0)
        with pytest.raises(ValueError, match="Volatility 'sigma'"):
            G2PlusPlus(sigma=-0.01)
        with pytest.raises(ValueError, match="Volatility 'eta'"):
            G2PlusPlus(eta=0.0)
        with pytest.raises(ValueError, match="Correlation 'rho'"):
            G2PlusPlus(rho=1.5)
        with pytest.raises(ValueError, match="Correlation 'rho'"):
            G2PlusPlus(rho=-1.2)

    def test_g2pp_process_properties(self) -> None:
        """Process properties reflect non-Lévy Markovian diffusion structure."""
        p = G2PlusPlus()
        props = p.properties
        assert not props.has_stationary_increments
        assert not props.has_independent_increments
        assert not props.is_martingale
        assert not props.is_subordinator

    def test_g2pp_phi_deterministic_shift(self) -> None:
        """Deterministic shift φ(t) matches r0 at t=0 and is monotonically increasing for positive variance."""
        p = G2PlusPlus(r0=0.03)
        assert math.isclose(p.phi(0.0), 0.03, rel_tol=1e-9)
        assert p.phi(1.0) > 0.03
        assert p.int_phi(0.0, 0.0) == 0.0
        assert p.int_phi(0.0, 1.0) > 0.0

    def test_g2pp_factor_covariance(self) -> None:
        """Factor covariance matrix matches analytical formulas."""
        p = G2PlusPlus(a=0.1, b=0.05, sigma=0.02, eta=0.01, rho=-0.2)
        cov0 = p.factor_covariance(0.0)
        assert np.allclose(cov0, 0.0)

        t = 2.0
        cov = p.factor_covariance(t)
        var_x_exact = (0.02**2 / (2.0 * 0.1)) * (1.0 - math.exp(-2.0 * 0.1 * t))
        var_y_exact = (0.01**2 / (2.0 * 0.05)) * (1.0 - math.exp(-2.0 * 0.05 * t))
        cov_xy_exact = -0.2 * (0.02 * 0.01 / 0.15) * (1.0 - math.exp(-0.15 * t))

        assert math.isclose(cov[0, 0], var_x_exact, rel_tol=1e-9)
        assert math.isclose(cov[1, 1], var_y_exact, rel_tol=1e-9)
        assert math.isclose(cov[0, 1], cov_xy_exact, rel_tol=1e-9)
        assert math.isclose(cov[1, 0], cov_xy_exact, rel_tol=1e-9)

        # Variance of short rate equals sum of covariance elements
        var_r = p.conditional_variance(t)
        assert math.isclose(var_r, cov[0, 0] + cov[1, 1] + 2.0 * cov[0, 1], rel_tol=1e-9)

    def test_g2pp_bond_price_properties(self) -> None:
        """Zero-coupon bond price P(t, T) satisfies P(t, t) = 1 and decreases with maturity."""
        p = G2PlusPlus(a=0.1, b=0.05, sigma=0.015, eta=0.01, rho=-0.3, r0=0.04)
        assert p.bond_price(0.0, 0.0) == 1.0
        assert p.bond_price(1.0, 1.0) == 1.0

        p1 = p.bond_price(0.0, 1.0)
        p5 = p.bond_price(0.0, 5.0)
        p10 = p.bond_price(0.0, 10.0)

        assert 0.0 < p10 < p5 < p1 < 1.0

        y1 = p.bond_yield(0.0, 1.0)
        assert math.isclose(y1, -math.log(p1) / 1.0, rel_tol=1e-9)

    def test_g2pp_integrated_variance_V(self) -> None:
        """Integrated variance V(t, T) is zero for T=t and positive for T > t."""
        p = G2PlusPlus()
        assert p.V(1.0, 1.0) == 0.0
        assert p.V(0.0, 2.0) > 0.0

    def test_g2pp_european_bond_options(self) -> None:
        """European bond options satisfy put-call parity."""
        p = G2PlusPlus(a=0.1, b=0.05, sigma=0.015, eta=0.01, rho=-0.3, r0=0.03)
        t, T, S = 0.0, 1.0, 2.0
        K = 0.95

        call = p.european_call_bond_option(t, T, S, K)
        put = p.european_put_bond_option(t, T, S, K)
        P_t_T = p.bond_price(t, T)
        P_t_S = p.bond_price(t, S)

        # Put-Call Parity: C - P = P(t, S) - K * P(t, T)
        assert math.isclose(call - put, P_t_S - K * P_t_T, abs_tol=1e-7)

    def test_g2pp_simulation(self) -> None:
        """Exact path simulation produces correct shapes and returns factor components."""
        p = G2PlusPlus(a=0.2, b=0.1, sigma=0.02, eta=0.015, rho=-0.4, r0=0.03)
        t_grid = np.linspace(0, 1.0, 11)
        rng = np.random.default_rng(42)

        paths = p.simulate(t_grid, n_paths=200, rng=rng)
        assert paths.shape == (200, 11)
        assert np.allclose(paths[:, 0], 0.03)

        r_paths, x_paths, y_paths = p.simulate(t_grid, n_paths=50, rng=rng, return_factors=True)
        assert r_paths.shape == (50, 11)
        assert x_paths.shape == (50, 11)
        assert y_paths.shape == (50, 11)
        assert np.allclose(x_paths[:, 0], 0.0)
        assert np.allclose(y_paths[:, 0], 0.0)
