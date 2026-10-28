"""
Exactness and mathematical consistency tests for Schwartz Two-Factor commodity model.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import (
    Schwartz2F,
    Schwartz97,
    SchwartzTwoFactor,
    SchwartzTwoFactorModel,
)


class TestSchwartzTwoFactorExactness:
    """Mathematical exactness and consistency test suite for Schwartz Two-Factor model."""

    def test_s2f_exactness_level_and_aliases(self) -> None:
        """SchwartzTwoFactor is MOMENT_PROPAGATION and aliased properly."""
        s2f = SchwartzTwoFactor(
            s0=20.0, delta0=0.05, kappa=1.2, alpha=0.08, sigma1=0.3, sigma2=0.2, rho=0.6, rate=0.05
        )
        assert s2f.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert Schwartz97 is SchwartzTwoFactor
        assert Schwartz2F is SchwartzTwoFactor
        assert SchwartzTwoFactorModel is SchwartzTwoFactor
        assert "SchwartzTwoFactor" in repr(s2f)
        assert "s0=20.0" in repr(s2f)
        assert "kappa=1.2" in repr(s2f)

    def test_s2f_triplet_raises_exactness_error(self) -> None:
        """Calling triplet raises ExactnessError due to coupled diffusion drift."""
        s2f = SchwartzTwoFactor()
        with pytest.raises(ExactnessError, match="triplet"):
            _ = s2f.triplet

    def test_s2f_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on invalid inputs."""
        with pytest.raises(ValueError, match="s0"):
            SchwartzTwoFactor(s0=-1.0)
        with pytest.raises(ValueError, match="kappa"):
            SchwartzTwoFactor(kappa=-0.5)
        with pytest.raises(ValueError, match="sigma1"):
            SchwartzTwoFactor(sigma1=-0.1)
        with pytest.raises(ValueError, match="sigma2"):
            SchwartzTwoFactor(sigma2=-0.1)
        with pytest.raises(ValueError, match="rho"):
            SchwartzTwoFactor(rho=1.5)

    def test_s2f_convenience_yield_moments(self) -> None:
        """Convenience yield conditional and stationary moments."""
        kappa = 1.5
        alpha = 0.08
        sigma2 = 0.20
        delta0 = 0.04
        s2f = SchwartzTwoFactor(kappa=kappa, alpha=alpha, sigma2=sigma2, delta0=delta0)

        # Stationary variance
        stat_var = (sigma2**2) / (2.0 * kappa)
        assert s2f.stationary_yield_variance == pytest.approx(stat_var, rel=1e-12)
        assert s2f.stationary_yield_std == pytest.approx(math.sqrt(stat_var), rel=1e-12)

        # Conditional moments
        for t in [0.5, 1.0, 2.0]:
            exp_kt = math.exp(-kappa * t)
            cond_mean = delta0 * exp_kt + alpha * (1.0 - exp_kt)
            cond_var = stat_var * (1.0 - math.exp(-2.0 * kappa * t))

            assert s2f.yield_mean(t) == pytest.approx(cond_mean, rel=1e-12)
            assert s2f.yield_variance(t) == pytest.approx(cond_var, rel=1e-12)
            assert s2f.yield_std(t) == pytest.approx(math.sqrt(cond_var), rel=1e-12)

    def test_s2f_log_spot_moments_and_futures_identity(self) -> None:
        """Identity exp(E[ln S_t] + 0.5 Var(ln S_t)) == F(S0, delta0, t) holds exactly."""
        s0 = 25.0
        delta0 = 0.04
        kappa = 1.3
        alpha = 0.07
        sigma1 = 0.28
        sigma2 = 0.18
        rho = 0.55
        rate = 0.04

        s2f = SchwartzTwoFactor(
            s0=s0,
            delta0=delta0,
            kappa=kappa,
            alpha=alpha,
            sigma1=sigma1,
            sigma2=sigma2,
            rho=rho,
            rate=rate,
        )

        for t in [0.25, 0.5, 1.0, 2.0]:
            mean_x = s2f.log_spot_mean(t)
            var_x = s2f.log_spot_variance(t)
            assert var_x > 0.0

            fwd_from_moments = math.exp(mean_x + 0.5 * var_x)
            fwd_direct = s2f.futures_price(t)
            assert fwd_from_moments == pytest.approx(fwd_direct, rel=1e-11)

    def test_s2f_futures_pricing_properties(self) -> None:
        """Futures price at tau=0 equals spot, vectorization works."""
        s2f = SchwartzTwoFactor(
            s0=20.0, delta0=0.05, kappa=1.2, alpha=0.08, sigma1=0.3, sigma2=0.2, rho=0.6, rate=0.05
        )

        # tau = 0
        assert s2f.futures_price(0.0) == pytest.approx(20.0, abs=1e-12)

        # Vectorization
        taus = np.array([0.0, 0.5, 1.0, 2.0])
        fwds = s2f.futures_curve(taus)
        assert len(fwds) == 4
        assert fwds[0] == pytest.approx(20.0, abs=1e-12)
        assert np.all(fwds > 0.0)

    def test_s2f_european_option_put_call_parity_exact(self) -> None:
        """Put-Call parity holds exactly: C - P = exp(-r*T) * (F(T) - K)."""
        s2f = SchwartzTwoFactor(
            s0=20.0, delta0=0.05, kappa=1.2, alpha=0.08, sigma1=0.3, sigma2=0.2, rho=0.6, rate=0.05
        )
        tau = 1.0
        fwd = s2f.futures_price(tau)
        disc = math.exp(-0.05 * tau)

        for strike in [15.0, 20.0, 25.0]:
            call = s2f.call_price(strike=strike, time_to_maturity=tau)
            put = s2f.put_price(strike=strike, time_to_maturity=tau)
            assert call > 0.0
            assert put > 0.0
            expected_diff = disc * (fwd - strike)
            assert (call - put) == pytest.approx(expected_diff, rel=1e-12)

    def test_s2f_options_and_futures_vs_monte_carlo(self) -> None:
        """Analytical futures and option prices match Monte Carlo simulation."""
        s0 = 20.0
        delta0 = 0.05
        kappa = 1.5
        alpha = 0.08
        sigma1 = 0.30
        sigma2 = 0.20
        rho = 0.60
        rate = 0.05
        tau = 1.0
        strike = 20.0

        s2f = SchwartzTwoFactor(
            s0=s0,
            delta0=delta0,
            kappa=kappa,
            alpha=alpha,
            sigma1=sigma1,
            sigma2=sigma2,
            rho=rho,
            rate=rate,
        )

        ana_fwd = s2f.futures_price(tau)
        ana_call = s2f.call_price(strike=strike, time_to_maturity=tau)
        ana_put = s2f.put_price(strike=strike, time_to_maturity=tau)

        # Monte Carlo
        dt = 0.002
        n_steps = int(tau / dt)
        s_paths = s2f.simulate(
            n_steps=n_steps, time_horizon=tau, n_paths=40_000, rng=np.random.default_rng(2026)
        )
        s_T = s_paths[:, -1]

        mc_fwd = float(np.mean(s_T))
        disc = math.exp(-rate * tau)
        mc_call = disc * float(np.mean(np.maximum(s_T - strike, 0.0)))
        mc_put = disc * float(np.mean(np.maximum(strike - s_T, 0.0)))

        assert mc_fwd == pytest.approx(ana_fwd, rel=0.01)
        assert mc_call == pytest.approx(ana_call, rel=0.02)
        assert mc_put == pytest.approx(ana_put, rel=0.02)

    def test_s2f_characteristic_function(self) -> None:
        """Characteristic function phi(0)=1, modulus <= 1, Hermitian symmetry."""
        s2f = SchwartzTwoFactor(
            s0=20.0, delta0=0.05, kappa=1.2, alpha=0.08, sigma1=0.3, sigma2=0.2, rho=0.6, rate=0.05
        )
        t = 1.0

        # At u = 0
        assert s2f.char_func(0.0, t=t) == pytest.approx(1.0 + 0.0j, abs=1e-12)

        # Modulus <= 1 for real u
        u_vals = np.array([-2.0, -0.5, 0.5, 2.0])
        cf_vals = s2f.char_func(u_vals, t=t)
        assert np.all(np.abs(cf_vals) <= 1.0 + 1e-12)

        # Hermitian symmetry
        u = 1.5
        cf_pos = s2f.char_func(u, t=t)
        cf_neg = s2f.char_func(-u, t=t)
        assert complex(cf_neg) == pytest.approx(complex(np.conj(cf_pos)), rel=1e-12)

    def test_s2f_joint_simulation_paths_and_mean(self) -> None:
        """Joint simulation returns spot and yield paths with matching moments."""
        s0 = 20.0
        delta0 = 0.05
        kappa = 1.2
        alpha = 0.08
        sigma1 = 0.30
        sigma2 = 0.20
        rho = 0.60
        rate = 0.05
        t = 1.0

        s2f = SchwartzTwoFactor(
            s0=s0,
            delta0=delta0,
            kappa=kappa,
            alpha=alpha,
            sigma1=sigma1,
            sigma2=sigma2,
            rho=rho,
            rate=rate,
        )

        s_paths, d_paths = s2f.simulate_joint(
            n_steps=25, time_horizon=t, n_paths=25_000, rng=np.random.default_rng(2026)
        )
        assert s_paths.shape == (25_000, 26)
        assert d_paths.shape == (25_000, 26)
        assert np.all(s_paths[:, 0] == s0)
        assert np.all(d_paths[:, 0] == delta0)
        assert np.all(s_paths > 0.0)

        # Yield moment convergence
        mc_yield_mean = float(np.mean(d_paths[:, -1]))
        theo_yield_mean = s2f.yield_mean(t)
        assert mc_yield_mean == pytest.approx(theo_yield_mean, rel=0.02)
