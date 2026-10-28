"""
Exactness and mathematical consistency tests for SABR Stochastic Volatility Model.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import SABR, SABRModel, SABRProcess


class TestSABRExactness:
    """Mathematical exactness and consistency test suite for SABRProcess."""

    def test_sabr_exactness_level_and_aliases(self) -> None:
        """SABRProcess has MOMENT_PROPAGATION level and is aliased to SABRModel, SABR."""
        sabr = SABRProcess(alpha=0.2, beta=0.6, nu=0.4, rho=-0.25, f0=100.0)
        assert sabr.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert SABR is SABRProcess
        assert SABRModel is SABRProcess
        assert "SABRProcess" in repr(sabr)
        assert "alpha=0.2" in repr(sabr)
        assert "beta=0.6" in repr(sabr)
        assert "nu=0.4" in repr(sabr)
        assert "rho=-0.25" in repr(sabr)

    def test_sabr_triplet_raises_exactness_error(self) -> None:
        """Calling triplet raises ExactnessError with mathematical explanation."""
        sabr = SABRProcess()
        with pytest.raises(ExactnessError, match="MOMENT_PROPAGATION"):
            _ = sabr.triplet

    def test_sabr_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on out-of-domain parameters."""
        with pytest.raises(ValueError, match="Initial volatility alpha"):
            SABRProcess(alpha=-0.1)
        with pytest.raises(ValueError, match="Initial volatility alpha"):
            SABRProcess(alpha=0.0)
        with pytest.raises(ValueError, match="Elasticity exponent beta"):
            SABRProcess(beta=-0.1)
        with pytest.raises(ValueError, match="Elasticity exponent beta"):
            SABRProcess(beta=1.1)
        with pytest.raises(ValueError, match="Vol-of-vol nu"):
            SABRProcess(nu=-0.2)
        with pytest.raises(ValueError, match="Correlation rho"):
            SABRProcess(rho=1.2)
        with pytest.raises(ValueError, match="Correlation rho"):
            SABRProcess(rho=-1.2)
        with pytest.raises(ValueError, match="Forward price f0"):
            SABRProcess(f0=-10.0)
        with pytest.raises(ValueError, match="Forward price f0"):
            SABRProcess(f0=0.0)

    def test_sabr_properties(self) -> None:
        """Structural properties reflect state-dependent stochastic volatility forward."""
        sabr = SABRProcess()
        props = sabr.properties
        assert not props.is_subordinator
        assert not props.has_independent_increments
        assert not props.has_stationary_increments
        assert props.is_martingale
        assert props.has_finite_mean
        assert props.has_finite_variance
        assert props.notes is not None
        assert any("Hagan" in note for note in props.notes)

    def test_sabr_forward_martingale_and_moments(self) -> None:
        """Under no-drift forward dynamics, E[F_t] = F_0 exactly."""
        sabr = SABRProcess(alpha=0.25, beta=0.5, nu=0.3, rho=-0.2, f0=105.0)
        for t in [0.25, 0.5, 1.0, 2.0]:
            assert sabr.conditional_mean(t) == pytest.approx(105.0, rel=1e-12)
            cums = sabr.conditional_cumulants(t, order=1)
            assert len(cums) == 1
            assert cums[1] == pytest.approx(105.0, rel=1e-12)

    def test_sabr_volatility_moments(self) -> None:
        """Volatility moments follow exact geometric Brownian motion formulas."""
        alpha0 = 0.3
        nu = 0.4
        sabr = SABRProcess(alpha=alpha0, beta=0.7, nu=nu, rho=-0.1, f0=100.0)

        t = 1.5
        # E[\alpha_t] = \alpha_0
        assert sabr.volatility_mean(t) == pytest.approx(alpha0, rel=1e-12)
        # Var(\alpha_t) = \alpha_0^2 (exp(\nu^2 t) - 1)
        expected_var = (alpha0**2) * (math.exp(nu**2 * t) - 1.0)
        assert sabr.volatility_variance(t) == pytest.approx(expected_var, rel=1e-10)

        # Log-volatility moments: E[ln \alpha_t] = ln \alpha_0 - 0.5 \nu^2 t
        assert sabr.log_volatility_mean(t) == pytest.approx(
            math.log(alpha0) - 0.5 * (nu**2) * t, rel=1e-12
        )
        assert sabr.log_volatility_variance(t) == pytest.approx((nu**2) * t, rel=1e-12)

    def test_sabr_atm_implied_volatility_consistency(self) -> None:
        """implied_volatility(K=f0) matches atm_volatility within asymptotic tolerances."""
        sabr = SABRProcess(alpha=0.25, beta=0.6, nu=0.35, rho=-0.3, f0=100.0)
        t = 1.0
        atm_direct = sabr.atm_volatility(T=t)
        iv_at_f0 = sabr.implied_volatility(K=100.0, T=t)
        assert iv_at_f0 == pytest.approx(atm_direct, rel=1e-5)

        # For near-ATM strikes, implied vol is smooth and close to ATM
        iv_near_left = sabr.implied_volatility(K=99.999, T=t)
        iv_near_right = sabr.implied_volatility(K=100.001, T=t)
        assert iv_near_left == pytest.approx(atm_direct, rel=1e-4)
        assert iv_near_right == pytest.approx(atm_direct, rel=1e-4)

    def test_sabr_lognormal_limiting_case(self) -> None:
        """When beta=1 and nu=0, SABR reduces to Black-76 with flat implied vol = alpha."""
        alpha0 = 0.22
        sabr = SABRProcess(alpha=alpha0, beta=1.0, nu=0.0, rho=0.0, f0=100.0)
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        vols = sabr.implied_volatility(strikes, T=1.0)
        np.testing.assert_allclose(vols, alpha0, rtol=1e-5)

    def test_sabr_skew_and_smile_behavior(self) -> None:
        """Negative rho produces downward skew; positive rho produces upward skew."""
        t = 1.0
        # Negative correlation -> downward sloping implied volatility (equity skew)
        sabr_neg = SABRProcess(alpha=0.2, beta=0.5, nu=0.4, rho=-0.6, f0=100.0)
        iv_low_k = sabr_neg.implied_volatility(80.0, T=t)
        iv_high_k = sabr_neg.implied_volatility(120.0, T=t)
        assert iv_low_k > iv_high_k

        # Positive correlation -> upward sloping implied volatility
        sabr_pos = SABRProcess(alpha=0.2, beta=0.5, nu=0.4, rho=0.6, f0=100.0)
        iv_low_k_pos = sabr_pos.implied_volatility(80.0, T=t)
        iv_high_k_pos = sabr_pos.implied_volatility(120.0, T=t)
        assert iv_low_k_pos < iv_high_k_pos

        # Symmetric vol-of-vol smile: rho=0, beta=1
        sabr_sym = SABRProcess(alpha=0.2, beta=1.0, nu=0.5, rho=0.0, f0=100.0)
        iv_otm_put = sabr_sym.implied_volatility(80.0, T=t)
        iv_otm_call = sabr_sym.implied_volatility(125.0, T=t)
        iv_atm = sabr_sym.implied_volatility(100.0, T=t)
        assert iv_otm_put > iv_atm
        assert iv_otm_call > iv_atm

    def test_sabr_put_call_parity(self) -> None:
        """Call and Put prices satisfy Black-76 Put-Call Parity: C - P = exp(-rT)*(F0 - K)."""
        sabr = SABRProcess(alpha=0.25, beta=0.5, nu=0.4, rho=-0.3, f0=100.0)
        rate = 0.04
        t = 1.5
        strikes = [70.0, 85.0, 100.0, 115.0, 130.0]

        df = math.exp(-rate * t)
        for k in strikes:
            c = sabr.call_price(K=k, T=t, r=rate)
            p = sabr.put_price(K=k, T=t, r=rate)
            expected_parity = df * (100.0 - k)
            actual_parity = c - p
            assert actual_parity == pytest.approx(expected_parity, rel=1e-7, abs=1e-8)

    def test_sabr_vectorized_option_pricing(self) -> None:
        """call_price and put_price accept array-like strikes."""
        sabr = SABRProcess(alpha=0.2, beta=0.7, nu=0.3, rho=-0.2, f0=100.0)
        strikes = np.array([90.0, 100.0, 110.0])
        calls = sabr.call_price(strikes, T=1.0, r=0.03)
        puts = sabr.put_price(strikes, T=1.0, r=0.03)

        assert calls.shape == (3,)
        assert puts.shape == (3,)
        assert calls[0] > calls[1] > calls[2]
        assert puts[0] < puts[1] < puts[2]

    def test_sabr_simulation_paths_and_schemes(self) -> None:
        """Vectorized simulation generates correct dimensions and positive forward/volatility."""
        rng = np.random.default_rng(1234)
        sabr = SABRProcess(alpha=0.25, beta=0.5, nu=0.35, rho=-0.3, f0=100.0)

        # Single path forward simulation
        paths = sabr.simulate(n_steps=50, time_horizon=1.0, n_paths=20, rng=rng)
        assert paths.shape == (20, 51)
        assert np.all(paths[:, 0] == 100.0)
        assert np.all(paths >= 0.0)

        # Joint forward and volatility simulation
        f_paths, a_paths = sabr.simulate_joint(
            n_steps=50, time_horizon=1.0, n_paths=20, rng=rng, method="euler"
        )
        assert f_paths.shape == (20, 51)
        assert a_paths.shape == (20, 51)
        assert np.all(f_paths[:, 0] == 100.0)
        assert np.all(a_paths[:, 0] == 0.25)
        assert np.all(f_paths >= 0.0)
        assert np.all(a_paths > 0.0)

    def test_sabr_simulation_martingale_property(self) -> None:
        """Monte Carlo mean of terminal forward F_T converges to F_0."""
        rng = np.random.default_rng(2026)
        f0 = 100.0
        sabr = SABRProcess(alpha=0.2, beta=0.5, nu=0.3, rho=-0.2, f0=f0)

        paths = sabr.simulate(
            n_steps=100, time_horizon=1.0, n_paths=30_000, rng=rng, method="log_euler"
        )
        terminal_mean = float(np.mean(paths[:, -1]))
        # Forward is a martingale, so E[F_T] = F_0. Within 1% standard error tolerance.
        assert terminal_mean == pytest.approx(f0, rel=0.015)
