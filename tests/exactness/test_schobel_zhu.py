"""
Exactness and mathematical consistency tests for Schöbel-Zhu (1999) stochastic volatility model.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import (
    SZModel,
    SchobelZhu,
    SchobelZhuModel,
    SchobelZhuProcess,
)


class TestSchobelZhuExactness:
    """Mathematical exactness and consistency test suite for SchobelZhuProcess."""

    def test_sz_exactness_level_and_aliases(self) -> None:
        """SchobelZhuProcess has MOMENT_PROPAGATION level and is aliased to SchobelZhu, SZModel, SchobelZhuModel."""
        sz = SchobelZhuProcess(sigma0=0.2, kappa=1.5, theta=0.2, tau=0.3, rho=-0.5, s0=100.0)
        assert sz.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert SchobelZhu is SchobelZhuProcess
        assert SchobelZhuModel is SchobelZhuProcess
        assert SZModel is SchobelZhuProcess
        assert "SchobelZhuProcess" in repr(sz)
        assert "kappa=1.5" in repr(sz)
        assert "tau=0.3" in repr(sz)

    def test_sz_triplet_raises_exactness_error(self) -> None:
        """Calling triplet raises ExactnessError with mathematical explanation."""
        sz = SchobelZhuProcess()
        with pytest.raises(ExactnessError, match="MOMENT_PROPAGATION"):
            _ = sz.triplet

    def test_sz_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on out-of-domain inputs."""
        with pytest.raises(ValueError, match="Mean-reversion speed 'kappa'"):
            SchobelZhuProcess(kappa=-1.5)
        with pytest.raises(ValueError, match="Mean-reversion speed 'kappa'"):
            SchobelZhuProcess(kappa=0.0)
        with pytest.raises(ValueError, match="Vol-of-vol 'tau'"):
            SchobelZhuProcess(tau=-0.2)
        with pytest.raises(ValueError, match="Vol-of-vol 'tau'"):
            SchobelZhuProcess(tau=0.0)
        with pytest.raises(ValueError, match="Correlation 'rho'"):
            SchobelZhuProcess(rho=1.5)
        with pytest.raises(ValueError, match="Correlation 'rho'"):
            SchobelZhuProcess(rho=-1.5)
        with pytest.raises(ValueError, match="Initial asset price 's0'"):
            SchobelZhuProcess(s0=-10.0)
        with pytest.raises(ValueError, match="Initial asset price 's0'"):
            SchobelZhuProcess(s0=0.0)

    def test_sz_properties(self) -> None:
        """Structural properties reflect state-dependent stochastic volatility diffusion."""
        sz = SchobelZhuProcess(rate=0.03, dividend=0.03)
        props = sz.properties
        assert not props.is_subordinator
        assert not props.has_independent_increments
        assert not props.has_stationary_increments
        assert props.is_martingale  # r == q implies discounted price is a martingale
        assert props.has_finite_mean
        assert props.has_finite_variance
        assert props.notes is not None

    def test_sz_volatility_moments(self) -> None:
        """Volatility sigma_t follows exact Ornstein-Uhlenbeck Gaussian moments."""
        kappa = 1.8
        theta = 0.22
        tau = 0.35
        sigma0 = 0.15
        sz = SchobelZhuProcess(sigma0=sigma0, kappa=kappa, theta=theta, tau=tau)

        for t in [0.5, 1.0, 2.0]:
            expected_mean = theta + (sigma0 - theta) * math.exp(-kappa * t)
            expected_var = (tau**2 / (2.0 * kappa)) * (1.0 - math.exp(-2.0 * kappa * t))

            assert sz.volatility_mean(t) == pytest.approx(expected_mean, rel=1e-12)
            assert sz.volatility_variance(t) == pytest.approx(expected_var, rel=1e-12)

    def test_sz_variance_moments(self) -> None:
        """Instantaneous variance v_t = sigma_t^2 moments follow squared Gaussian law."""
        kappa = 2.0
        theta = 0.20
        tau = 0.30
        sigma0 = 0.25
        sz = SchobelZhuProcess(sigma0=sigma0, kappa=kappa, theta=theta, tau=tau)

        for t in [0.5, 1.5]:
            m = sz.volatility_mean(t)
            v = sz.volatility_variance(t)

            expected_v_mean = v + m**2
            expected_v_var = 2.0 * (v**2) + 4.0 * (m**2) * v

            assert sz.variance_mean(t) == pytest.approx(expected_v_mean, rel=1e-12)
            assert sz.variance_variance(t) == pytest.approx(expected_v_var, rel=1e-12)

        # Stationary variance
        var_inf_sig = tau**2 / (2.0 * kappa)
        assert sz.stationary_volatility_mean == pytest.approx(theta, rel=1e-12)
        assert sz.stationary_volatility_var == pytest.approx(var_inf_sig, rel=1e-12)
        assert sz.stationary_variance_mean == pytest.approx(theta**2 + var_inf_sig, rel=1e-12)

    def test_sz_characteristic_function_properties(self) -> None:
        """Characteristic function satisfies phi(0)=1 and |phi|<=1 for real u."""
        sz = SchobelZhuProcess(sigma0=0.2, kappa=1.5, theta=0.2, tau=0.3, rho=-0.4, rate=0.04)
        assert sz.char_func(0.0, t=1.0) == pytest.approx(1.0 + 0.0j, abs=1e-12)

        u_grid = np.array([-3.0, -1.0, 0.5, 2.0, 5.0])
        cf_vals = sz.char_func(u_grid, t=1.0)
        assert np.all(np.abs(cf_vals) <= 1.0 + 1e-10)

    def test_sz_option_pricing_put_call_parity(self) -> None:
        """European call and put option prices satisfy exact Put-Call Parity."""
        sz = SchobelZhuProcess(
            sigma0=0.20,
            kappa=1.5,
            theta=0.20,
            tau=0.30,
            rho=-0.5,
            s0=100.0,
            rate=0.04,
            dividend=0.01,
        )
        t = 1.0
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]

        df_r = math.exp(-0.04 * t)
        df_q = math.exp(-0.01 * t)

        for k in strikes:
            call = float(sz.call_price(k, T=t))
            put = float(sz.put_price(k, T=t))
            expected_parity = 100.0 * df_q - k * df_r
            actual_parity = call - put
            assert actual_parity == pytest.approx(expected_parity, rel=1e-6, abs=1e-7)

    def test_sz_vectorized_option_pricing(self) -> None:
        """call_price and put_price accept array-like strikes."""
        sz = SchobelZhuProcess(s0=100.0, rate=0.03)
        strikes = np.array([90.0, 100.0, 110.0])
        calls = sz.call_price(strikes, T=1.0)
        puts = sz.put_price(strikes, T=1.0)

        assert calls.shape == (3,)
        assert puts.shape == (3,)
        assert calls[0] > calls[1] > calls[2]
        assert puts[0] < puts[1] < puts[2]

    def test_sz_simulation_paths_and_martingale(self) -> None:
        """Monte Carlo paths start at (s0, sigma0) and average terminal price satisfies martingale."""
        rng = np.random.default_rng(2026)
        s0 = 100.0
        rate = 0.05
        div = 0.02
        T = 1.0
        sz = SchobelZhuProcess(
            sigma0=0.20, kappa=1.5, theta=0.20, tau=0.30, rho=-0.4, s0=s0, rate=rate, dividend=div
        )

        s_paths, sig_paths = sz.simulate_joint(n_steps=25, time_horizon=T, n_paths=20_000, rng=rng)
        assert s_paths.shape == (20_000, 26)
        assert sig_paths.shape == (20_000, 26)
        assert np.all(s_paths[:, 0] == s0)
        assert np.all(sig_paths[:, 0] == 0.20)
        assert np.all(s_paths > 0.0)

        mc_terminal = float(np.mean(s_paths[:, -1]))
        theo_terminal = s0 * math.exp((rate - div) * T)
        assert mc_terminal == pytest.approx(theo_terminal, rel=0.015)
