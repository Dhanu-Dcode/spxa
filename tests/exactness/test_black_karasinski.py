"""
Exactness and mathematical consistency tests for Black-Karasinski (1991) short-rate model.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import (
    BKModel,
    BKProcess,
    BlackKarasinski,
    BlackKarasinskiProcess,
)


class TestBlackKarasinskiExactness:
    """Mathematical exactness and consistency test suite for BlackKarasinskiProcess."""

    def test_bk_exactness_level_and_aliases(self) -> None:
        """BlackKarasinskiProcess has MOMENT_PROPAGATION level and is aliased to BKProcess, BKModel, BlackKarasinski."""
        bk = BlackKarasinskiProcess(a=0.15, sigma=0.20, b=-3.5, r0=0.03)
        assert bk.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert BlackKarasinski is BlackKarasinskiProcess
        assert BKProcess is BlackKarasinskiProcess
        assert BKModel is BlackKarasinskiProcess
        assert "BlackKarasinskiProcess" in repr(bk)
        assert "a=0.15" in repr(bk)
        assert "sigma=0.2" in repr(bk)

    def test_bk_triplet_raises_exactness_error(self) -> None:
        """Calling triplet raises ExactnessError with mathematical explanation."""
        bk = BlackKarasinskiProcess()
        with pytest.raises(ExactnessError, match="MOMENT_PROPAGATION"):
            _ = bk.triplet

    def test_bk_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on out-of-domain inputs."""
        with pytest.raises(ValueError, match="Mean-reversion speed 'a'"):
            BlackKarasinskiProcess(a=-0.1)
        with pytest.raises(ValueError, match="Mean-reversion speed 'a'"):
            BlackKarasinskiProcess(a=0.0)
        with pytest.raises(ValueError, match="Volatility 'sigma'"):
            BlackKarasinskiProcess(sigma=-0.05)
        with pytest.raises(ValueError, match="Volatility 'sigma'"):
            BlackKarasinskiProcess(sigma=0.0)
        with pytest.raises(ValueError, match="Initial short rate 'r0'"):
            BlackKarasinskiProcess(r0=-0.01)
        with pytest.raises(ValueError, match="Initial short rate 'r0'"):
            BlackKarasinskiProcess(r0=0.0)

    def test_bk_properties(self) -> None:
        """Structural properties reflect state-dependent mean-reverting diffusion."""
        bk = BlackKarasinskiProcess()
        props = bk.properties
        assert not props.is_subordinator
        assert not props.has_independent_increments
        assert not props.has_stationary_increments
        assert not props.is_martingale
        assert props.has_finite_mean
        assert props.has_finite_variance
        assert props.notes is not None

    def test_bk_log_rate_moments(self) -> None:
        """Log-rate x_t follows exact Ornstein-Uhlenbeck moments."""
        a = 0.20
        sigma = 0.25
        b = -3.4
        r0 = 0.04
        x0 = math.log(r0)
        bk = BlackKarasinskiProcess(a=a, sigma=sigma, b=b, r0=r0)

        for t in [0.5, 1.0, 2.0]:
            expected_mean = b + (x0 - b) * math.exp(-a * t)
            expected_var = (sigma ** 2 / (2.0 * a)) * (1.0 - math.exp(-2.0 * a * t))

            assert bk.log_rate_mean(t) == pytest.approx(expected_mean, rel=1e-12)
            assert bk.log_rate_variance(t) == pytest.approx(expected_var, rel=1e-12)
            assert bk.log_rate_std(t) == pytest.approx(math.sqrt(expected_var), rel=1e-12)

    def test_bk_short_rate_lognormal_moments(self) -> None:
        """Short rate r_t follows exact lognormal moments based on log-rate x_t."""
        a = 0.15
        sigma = 0.18
        b = -3.2
        r0 = 0.035
        bk = BlackKarasinskiProcess(a=a, sigma=sigma, b=b, r0=r0)

        for t in [0.5, 1.5, 3.0]:
            mu_x = bk.log_rate_mean(t)
            var_x = bk.log_rate_variance(t)

            expected_r_mean = math.exp(mu_x + 0.5 * var_x)
            expected_r_var = math.exp(2.0 * mu_x + var_x) * (math.exp(var_x) - 1.0)

            assert bk.conditional_mean(t) == pytest.approx(expected_r_mean, rel=1e-12)
            assert bk.conditional_variance(t) == pytest.approx(expected_r_var, rel=1e-12)
            assert bk.conditional_std(t) == pytest.approx(math.sqrt(expected_r_var), rel=1e-12)

            cums = bk.conditional_cumulants(t, order=2)
            assert cums[1] == pytest.approx(expected_r_mean, rel=1e-12)
            assert cums[2] == pytest.approx(expected_r_var, rel=1e-12)

    def test_bk_stationary_moments(self) -> None:
        """Asymptotic stationary distribution of short rate."""
        a = 0.25
        sigma = 0.20
        b = -3.5
        bk = BlackKarasinskiProcess(a=a, sigma=sigma, b=b, r0=0.03)

        var_inf_x = sigma ** 2 / (2.0 * a)
        expected_r_inf = math.exp(b + sigma ** 2 / (4.0 * a))
        expected_var_inf = math.exp(2.0 * b + var_inf_x) * (math.exp(var_inf_x) - 1.0)

        assert bk.stationary_log_mean == pytest.approx(b, rel=1e-12)
        assert bk.stationary_log_variance == pytest.approx(var_inf_x, rel=1e-12)
        assert bk.stationary_median == pytest.approx(math.exp(b), rel=1e-12)
        assert bk.stationary_mean == pytest.approx(expected_r_inf, rel=1e-12)
        assert bk.stationary_variance == pytest.approx(expected_var_inf, rel=1e-12)

    def test_bk_zero_coupon_bond_pricing_monotonicity(self) -> None:
        """Zero-coupon bond prices are monotonically decreasing in maturity."""
        rng = np.random.default_rng(42)
        bk = BlackKarasinskiProcess(a=0.10, sigma=0.15, b=-3.5, r0=0.03)
        maturities = [0.5, 1.0, 2.0, 5.0]
        prices = bk.bond_price(maturities, n_paths=10_000, rng=rng)

        assert len(prices) == 4
        # 1 > P(0.5) > P(1.0) > P(2.0) > P(5.0) > 0
        assert 1.0 > prices[0] > prices[1] > prices[2] > prices[3] > 0.0

        # Yield to maturity is positive
        yields = bk.yield_to_maturity(maturities, n_paths=10_000, rng=np.random.default_rng(42))
        assert np.all(yields > 0.0)

    def test_bk_simulation_paths_and_positivity(self) -> None:
        """Simulated short rates are strictly positive across all paths."""
        rng = np.random.default_rng(1234)
        bk = BlackKarasinskiProcess(a=0.15, sigma=0.25, b=-3.5, r0=0.03)

        paths = bk.simulate(n_steps=30, time_horizon=1.0, n_paths=20, rng=rng)
        assert paths.shape == (20, 31)
        assert np.all(paths[:, 0] == 0.03)
        assert np.all(paths > 0.0)

        r_p, x_p = bk.simulate_joint(n_steps=30, time_horizon=1.0, n_paths=20, rng=rng)
        assert r_p.shape == (20, 31)
        assert x_p.shape == (20, 31)
        np.testing.assert_allclose(np.exp(x_p), r_p, rtol=1e-12)

    def test_bk_simulation_moments_convergence(self) -> None:
        """Monte Carlo terminal short rates converge to theoretical lognormal mean."""
        rng = np.random.default_rng(2026)
        bk = BlackKarasinskiProcess(a=0.15, sigma=0.20, b=-3.5, r0=0.03)

        T = 1.0
        paths = bk.simulate(n_steps=25, time_horizon=T, n_paths=30_000, rng=rng)
        mc_mean = float(np.mean(paths[:, -1]))
        theo_mean = bk.conditional_mean(T)

        assert mc_mean == pytest.approx(theo_mean, rel=0.015)
