"""
Exactness and mathematical consistency tests for Bates (1996) stochastic volatility with jumps model.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import Bates96, BatesModel, BatesProcess, HestonProcess, SVJ


class TestBatesExactness:
    """Mathematical exactness and consistency test suite for BatesProcess."""

    def test_bates_exactness_level_and_aliases(self) -> None:
        """BatesProcess is MOMENT_PROPAGATION and identical to BatesModel, Bates96, and SVJ."""
        bates = BatesProcess(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            sigma_v=0.3,
            rho=-0.6,
            rate=1.0,
            jump_mean=-0.05,
            jump_std=0.1,
            mu=0.05,
            s0=100.0,
        )
        assert bates.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert BatesModel is BatesProcess
        assert Bates96 is BatesProcess
        assert SVJ is BatesProcess
        assert "BatesProcess" in repr(bates)
        assert "sigma_v=0.3" in repr(bates)
        assert "rate=1.0" in repr(bates)

    def test_bates_triplet_raises_exactness_error(self) -> None:
        """Calling triplet on BatesProcess raises ExactnessError with clear explanation."""
        bates = BatesProcess()
        with pytest.raises(ExactnessError, match="MOMENT_PROPAGATION"):
            _ = bates.triplet

    def test_bates_invalid_parameters_raise(self) -> None:
        """BatesProcess constructor validates parameter bounds."""
        with pytest.raises(ValueError, match="v0"):
            BatesProcess(v0=-0.01)
        with pytest.raises(ValueError, match="kappa"):
            BatesProcess(kappa=0.0)
        with pytest.raises(ValueError, match="theta"):
            BatesProcess(theta=-0.04)
        with pytest.raises(ValueError, match="sigma_v"):
            BatesProcess(sigma_v=0.0)
        with pytest.raises(ValueError, match="rho"):
            BatesProcess(rho=-1.5)
        with pytest.raises(ValueError, match="rho"):
            BatesProcess(rho=1.5)
        with pytest.raises(ValueError, match="lambda"):
            BatesProcess(rate=-0.1)
        with pytest.raises(ValueError, match="sigma_j"):
            BatesProcess(jump_std=0.0)
        with pytest.raises(ValueError, match="s0"):
            BatesProcess(s0=-10.0)

    def test_bates_properties(self) -> None:
        """Check structural properties of Bates stochastic volatility jump model."""
        bates = BatesProcess()
        props = bates.properties
        assert not props.is_subordinator
        assert not props.has_independent_increments
        assert not props.has_stationary_increments
        assert props.has_finite_mean
        assert props.has_finite_variance
        assert props.self_similarity_index is None
        assert len(props.notes) >= 2

    def test_bates_cir_variance_properties(self) -> None:
        """Variance process moments match CIR analytical formulas."""
        v0 = 0.04
        kappa = 2.0
        theta = 0.06
        sigma_v = 0.3
        bates = BatesProcess(v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v)

        # Feller ratio: 2 * 2.0 * 0.06 / 0.09 = 0.24 / 0.09 = 2.6667 > 1
        assert bates.feller_ratio == pytest.approx((2.0 * kappa * theta) / (sigma_v**2))
        assert bates.is_feller_satisfied

        # At t=0, mean is v0 and variance is 0
        assert bates.variance_mean(0.0) == pytest.approx(v0)
        assert bates.variance_var(0.0) == pytest.approx(0.0, abs=1e-12)
        assert bates.expected_integrated_variance(0.0) == 0.0

        # As t -> inf, approaches stationary values
        t_large = 100.0
        assert bates.variance_mean(t_large) == pytest.approx(
            bates.stationary_variance_mean, rel=1e-6
        )
        assert bates.variance_var(t_large) == pytest.approx(bates.stationary_variance_var, rel=1e-6)

        # Expected integrated variance formula
        t = 1.5
        theor_int_var = theta * t + (v0 - theta) * (1.0 - math.exp(-kappa * t)) / kappa
        assert bates.expected_integrated_variance(t) == pytest.approx(theor_int_var, rel=1e-12)

    def test_bates_characteristic_function_properties(self) -> None:
        """Characteristic function satisfies probability theory and martingale requirements."""
        mu = 0.05
        rate = 1.2
        mu_j = -0.06
        sigma_j = 0.12
        bates = BatesProcess(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            sigma_v=0.3,
            rho=-0.6,
            rate=rate,
            jump_mean=mu_j,
            jump_std=sigma_j,
            mu=mu,
            s0=100.0,
        )

        t = 0.75
        # 1. phi(0, t) == 1
        cf_0 = bates.char_func(0.0, t=t)
        assert cf_0 == pytest.approx(1.0 + 0j, abs=1e-12)

        # 2. Modulus |phi(u, t)| <= 1 for real u
        u_vals = np.linspace(-10.0, 10.0, 21)
        cfs = bates.char_func(u_vals, t=t)
        assert np.all(np.abs(cfs) <= 1.0 + 1e-10)

        # 3. Martingale property: E[exp(X_t)] = phi(-i, t) == exp(mu * t)
        cf_mart = bates.char_func(-1j, t=t)
        expected_mart = math.exp(mu * t)
        assert complex(cf_mart).real == pytest.approx(expected_mart, rel=1e-9)
        assert abs(complex(cf_mart).imag) < 1e-9

        # 4. Asset characteristic function: E[exp(i u ln S_t)] = S_0^(i u) * phi(u, t)
        u = 2.5
        phase = np.exp(1j * u * np.log(bates.s0))
        assert bates.asset_char_func(u, t=t) == pytest.approx(
            phase * bates.char_func(u, t=t), rel=1e-12
        )

    def test_bates_reduces_to_heston_when_rate_is_zero(self) -> None:
        """When jump arrival rate is zero, Bates is identical to Heston."""
        v0 = 0.04
        kappa = 2.0
        theta = 0.04
        sigma_v = 0.3
        rho = -0.6
        mu = 0.04
        s0 = 100.0

        bates = BatesProcess(
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            rate=0.0,
            mu=mu,
            s0=s0,
        )
        heston = HestonProcess(
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            mu=mu,
            s0=s0,
        )

        T = 0.5
        u_test = np.array([-2.0, -0.5, 0.0, 1.0, 3.5])
        assert np.allclose(bates.char_func(u_test, t=T), heston.char_func(u_test, t=T), atol=1e-12)

        # Option prices match Heston exactly
        strikes = [90.0, 100.0, 110.0]
        c_bates = bates.call_price(strikes, T=T, r=0.05, q=0.01)
        c_heston = heston.call_price(strikes, T=T, r=0.05, q=0.01)
        assert np.allclose(c_bates, c_heston, atol=1e-8)

    def test_bates_cumulants_and_moments(self) -> None:
        """Cumulants match analytical expectation and conditional variance."""
        mu = 0.05
        rate = 1.5
        mu_j = -0.04
        sigma_j = 0.10
        bates = BatesProcess(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            sigma_v=0.3,
            rho=-0.6,
            rate=rate,
            jump_mean=mu_j,
            jump_std=sigma_j,
            mu=mu,
            s0=100.0,
        )

        T = 1.0
        k = bates.conditional_cumulants(t=T, order=4)
        mean_theor = bates.conditional_mean(t=T)
        var_theor = bates.conditional_variance(t=T)

        assert k[1] == pytest.approx(mean_theor, rel=1e-4)
        assert k[2] == pytest.approx(var_theor, rel=1e-4)
        assert not np.isnan(k[3])
        assert not np.isnan(k[4])

    def test_bates_option_pricing_and_put_call_parity(self) -> None:
        """Call and Put prices satisfy strict Put-Call parity and monotonic properties."""
        S0 = 100.0
        r = 0.05
        q = 0.02
        T = 0.75

        bates = BatesProcess(
            v0=0.04,
            kappa=2.5,
            theta=0.04,
            sigma_v=0.35,
            rho=-0.7,
            rate=1.2,
            jump_mean=-0.08,
            jump_std=0.15,
            mu=r - q,
            s0=S0,
        )

        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        calls = bates.call_price(strikes, T=T, r=r, q=q)
        puts = bates.put_price(strikes, T=T, r=r, q=q)

        # Monotonicity
        assert np.all(np.diff(calls) < 0)  # Calls decrease with strike
        assert np.all(np.diff(puts) > 0)  # Puts increase with strike

        # Put-Call parity: C - P = S0 * exp(-q * T) - K * exp(-r * T)
        theor_diff = S0 * np.exp(-q * T) - strikes * np.exp(-r * T)
        assert np.allclose(calls - puts, theor_diff, atol=1e-10)

    def test_bates_call_pricing_vs_monte_carlo(self) -> None:
        """Gil-Pelaez option price agrees with Monte Carlo simulation within sampling error."""
        S0 = 100.0
        K = 100.0
        r = 0.05
        q = 0.0
        T = 0.5

        bates = BatesProcess(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            sigma_v=0.3,
            rho=-0.5,
            rate=1.0,
            jump_mean=-0.05,
            jump_std=0.10,
            mu=r - q,
            s0=S0,
        )

        analytical_call = float(bates.call_price(K, T=T, r=r, q=q))

        # Vectorized Monte Carlo simulation
        n_paths = 100_000
        n_steps = 50
        rng = np.random.default_rng(42)
        paths = bates.simulate(n_steps=n_steps, time_horizon=T, n_paths=n_paths, rng=rng)
        s_T = S0 * np.exp(paths[:, -1])
        mc_call = float(np.exp(-r * T) * np.mean(np.maximum(s_T - K, 0.0)))
        mc_se = float(np.exp(-r * T) * np.std(np.maximum(s_T - K, 0.0)) / math.sqrt(n_paths))

        assert abs(analytical_call - mc_call) < 3.0 * mc_se

    def test_bates_implied_vol_smile(self) -> None:
        """Implied volatility smile produces valid non-negative implied volatilities."""
        S0 = 100.0
        r = 0.04
        T = 0.5
        bates = BatesProcess(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            sigma_v=0.3,
            rho=-0.6,
            rate=1.5,
            jump_mean=-0.08,
            jump_std=0.12,
            mu=r,
            s0=S0,
        )

        strikes = [85.0, 95.0, 100.0, 105.0, 115.0]
        ivs = bates.implied_vol_smile(strikes, T=T, r=r)
        assert len(ivs) == 5
        assert np.all(ivs > 0.10)
        assert np.all(ivs < 1.0)
        # With negative rho and negative jump mean, low strikes should have higher vol (skew)
        assert ivs[0] > ivs[-1]

    def test_bates_simulation_paths_and_joint_simulation(self) -> None:
        """Simulation produces correct shapes, initial values, and non-negative variance."""
        bates = BatesProcess(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            sigma_v=0.3,
            rho=-0.5,
            rate=1.0,
            jump_mean=-0.05,
            jump_std=0.10,
            s0=100.0,
        )

        n_paths = 20
        n_steps = 30
        rng = np.random.default_rng(123)

        # 1. simulate returns shape (n_paths, n_steps + 1)
        x_paths = bates.simulate(n_steps=n_steps, time_horizon=0.5, n_paths=n_paths, rng=rng)
        assert x_paths.shape == (n_paths, n_steps + 1)
        assert np.all(x_paths[:, 0] == 0.0)

        # 2. simulate_joint returns (paths_x, paths_v)
        px, pv = bates.simulate_joint(n_steps=n_steps, time_horizon=0.5, n_paths=n_paths, rng=rng)
        assert px.shape == (n_paths, n_steps + 1)
        assert pv.shape == (n_paths, n_steps + 1)
        assert np.all(px[:, 0] == 0.0)
        assert np.all(pv[:, 0] == bates.v0)
        assert np.all(pv >= 0.0)  # Full truncation guarantees non-negative variance
