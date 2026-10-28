"""
Exactness and mathematical consistency tests for Barndorff-Nielsen & Shephard (BNS) model.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import (
    BNS,
    BNSModel,
    BNSProcess,
    BarndorffNielsenShephard,
)


class TestBNSExactness:
    """Mathematical exactness and consistency test suite for BNS model."""

    def test_bns_exactness_level_and_aliases(self) -> None:
        """BNSModel is MOMENT_PROPAGATION and aliased properly."""
        bns = BNSModel(
            lambda_param=1.5,
            alpha=2.0,
            beta_bdlp=25.0,
            rho=-0.5,
            beta=-0.5,
            sigma0_sq=0.04,
            s0=100.0,
        )
        assert bns.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert BNS is BNSModel
        assert BNSProcess is BNSModel
        assert BarndorffNielsenShephard is BNSModel
        assert "BNSModel" in repr(bns)
        assert "lambda_param=1.5" in repr(bns)
        assert "s0=100.0" in repr(bns)

    def test_bns_triplet_raises_exactness_error(self) -> None:
        """Calling triplet raises ExactnessError due to stochastic volatility dynamics."""
        bns = BNSModel()
        with pytest.raises(ExactnessError, match="triplet"):
            _ = bns.triplet

    def test_bns_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on invalid inputs."""
        with pytest.raises(ValueError, match="lambda_param"):
            BNSModel(lambda_param=-1.0)
        with pytest.raises(ValueError, match="alpha"):
            BNSModel(alpha=-2.0)
        with pytest.raises(ValueError, match="beta_bdlp"):
            BNSModel(beta_bdlp=-10.0)
        with pytest.raises(ValueError, match="rho"):
            BNSModel(rho=0.5)
        with pytest.raises(ValueError, match="sigma0_sq"):
            BNSModel(sigma0_sq=-0.04)
        with pytest.raises(ValueError, match="s0"):
            BNSModel(s0=-100.0)

    def test_bns_variance_moments(self) -> None:
        """Stationary, conditional, and integrated variance moments."""
        lam = 1.8
        alpha = 2.5
        beta_bdlp = 20.0
        v0 = 0.05
        bns = BNSModel(lambda_param=lam, alpha=alpha, beta_bdlp=beta_bdlp, sigma0_sq=v0)

        # Stationary moments
        stat_mean = alpha / beta_bdlp
        stat_var = alpha / (beta_bdlp**2)
        assert bns.stationary_variance_mean == pytest.approx(stat_mean, rel=1e-12)
        assert bns.stationary_variance_variance == pytest.approx(stat_var, rel=1e-12)
        assert bns.stationary_variance_std == pytest.approx(math.sqrt(stat_var), rel=1e-12)

        # Conditional moments
        for t in [0.5, 1.0, 2.0]:
            exp_lt = math.exp(-lam * t)
            cond_mean = v0 * exp_lt + stat_mean * (1.0 - exp_lt)
            cond_var = stat_var * (1.0 - math.exp(-2.0 * lam * t))

            assert bns.variance_mean(t) == pytest.approx(cond_mean, rel=1e-12)
            assert bns.variance_variance(t) == pytest.approx(cond_var, rel=1e-12)

            # Integrated variance mean
            factor = (1.0 - exp_lt) / lam
            expected_int_var = v0 * factor + stat_mean * (t - factor)
            assert bns.integrated_variance_mean(t) == pytest.approx(expected_int_var, rel=1e-12)

    def test_bns_risk_neutral_martingale_condition(self) -> None:
        """Discounted asset price is a martingale: phi(-i; t) == exp((r - q)*t)."""
        bns = BNSModel(
            lambda_param=1.5,
            alpha=2.0,
            beta_bdlp=25.0,
            rho=-0.5,
            beta=-0.5,
            rate=0.04,
            dividend=0.01,
        )
        for t in [0.25, 0.5, 1.0]:
            cf_val = bns.char_func(-1j, t=t)
            expected_val = math.exp((0.04 - 0.01) * t)
            assert complex(cf_val).real == pytest.approx(expected_val, rel=1e-4)
            assert abs(complex(cf_val).imag) < 1e-4

    def test_bns_characteristic_function_properties(self) -> None:
        """Characteristic function phi(0)=1, modulus <= 1, Hermitian symmetry."""
        bns = BNSModel(lambda_param=1.5, alpha=2.0, beta_bdlp=25.0, rho=-0.5, beta=-0.5, rate=0.05)
        t = 1.0

        # At u = 0
        assert bns.char_func(0.0, t=t) == pytest.approx(1.0 + 0.0j, abs=1e-12)

        # Modulus <= 1
        u_vals = np.array([-2.0, -0.5, 0.5, 2.0])
        cf_vals = bns.char_func(u_vals, t=t)
        assert np.all(np.abs(cf_vals) <= 1.0 + 1e-12)

        # Hermitian symmetry
        u = 1.5
        cf_pos = bns.char_func(u, t=t)
        cf_neg = bns.char_func(-u, t=t)
        assert complex(cf_neg) == pytest.approx(complex(np.conj(cf_pos)), rel=1e-10)

    def test_bns_european_option_pricing_and_put_call_parity(self) -> None:
        """Put-Call parity holds exactly: C - P = S0 * exp(-q*T) - K * exp(-r*T)."""
        s0 = 100.0
        rate = 0.05
        div = 0.01
        bns = BNSModel(
            lambda_param=1.5,
            alpha=2.0,
            beta_bdlp=25.0,
            rho=-0.5,
            beta=-0.5,
            sigma0_sq=0.04,
            s0=s0,
            rate=rate,
            dividend=div,
        )

        tau = 1.0
        disc_r = math.exp(-rate * tau)
        disc_q = math.exp(-div * tau)

        strikes = [80.0, 100.0, 120.0]
        call_prices = []

        for K in strikes:
            call = bns.call_price(strike=K, time_to_maturity=tau)
            put = bns.put_price(strike=K, time_to_maturity=tau)
            call_prices.append(call)

            assert call > 0.0
            assert put > 0.0

            parity_lhs = call - put
            parity_rhs = s0 * disc_q - K * disc_r
            assert parity_lhs == pytest.approx(parity_rhs, rel=1e-10)

        # Monotonicity in strike
        assert call_prices[0] > call_prices[1] > call_prices[2]

        # Implied vol is positive
        iv = bns.implied_volatility(strike=100.0, time_to_maturity=tau)
        assert 0.10 < iv < 0.40

    def test_bns_options_vs_monte_carlo(self) -> None:
        """Fourier inversion call price matches Monte Carlo discounted expectation."""
        s0 = 100.0
        K = 100.0
        rate = 0.05
        T = 0.5

        bns = BNSModel(
            lambda_param=1.5,
            alpha=2.0,
            beta_bdlp=25.0,
            rho=-0.5,
            beta=-0.5,
            sigma0_sq=0.04,
            s0=s0,
            rate=rate,
        )

        ana_call = bns.call_price(strike=K, time_to_maturity=T)

        # Monte Carlo
        dt = 0.002
        n_steps = int(T / dt)
        s_paths = bns.simulate(
            n_steps=n_steps, time_horizon=T, n_paths=40_000, rng=np.random.default_rng(2026)
        )
        s_T = s_paths[:, -1]
        mc_call = math.exp(-rate * T) * float(np.mean(np.maximum(s_T - K, 0.0)))

        assert mc_call == pytest.approx(ana_call, rel=0.03)

    def test_bns_joint_simulation_paths_and_mean(self) -> None:
        """Joint simulation returns asset and variance paths with matching moments."""
        bns = BNSModel(
            lambda_param=1.5,
            alpha=2.0,
            beta_bdlp=25.0,
            rho=-0.5,
            beta=-0.5,
            sigma0_sq=0.04,
            s0=100.0,
            rate=0.05,
        )

        t = 1.0
        s_paths, v_paths = bns.simulate_joint(
            n_steps=25, time_horizon=t, n_paths=25_000, rng=np.random.default_rng(2026)
        )
        assert s_paths.shape == (25_000, 26)
        assert v_paths.shape == (25_000, 26)
        assert np.all(s_paths[:, 0] == 100.0)
        assert np.all(v_paths[:, 0] == 0.04)
        assert np.all(s_paths > 0.0)
        assert np.all(v_paths >= 0.0)

        mc_var_mean = float(np.mean(v_paths[:, -1]))
        theo_var_mean = bns.variance_mean(t)
        assert mc_var_mean == pytest.approx(theo_var_mean, rel=0.03)
