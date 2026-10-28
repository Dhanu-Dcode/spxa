"""
Exactness and mathematical consistency tests for Jump-Extended CIR (JCIR) process.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import (
    JCIR,
    JCIRModel,
    JCIRProcess,
    JumpCIRProcess,
)


class TestJCIRExactness:
    """Mathematical exactness and consistency test suite for JCIR process."""

    def test_jcir_exactness_level_and_aliases(self) -> None:
        """JCIR is MOMENT_PROPAGATION and aliased properly."""
        jcir = JCIRProcess(kappa=1.2, theta=0.05, sigma=0.15, lambda_j=0.8, mu_j=0.03, r0=0.04)
        assert jcir.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert JCIR is JCIRProcess
        assert JumpCIRProcess is JCIRProcess
        assert JCIRModel is JCIRProcess
        assert "JCIRProcess" in repr(jcir)
        assert "kappa=1.2" in repr(jcir)
        assert "r0=0.04" in repr(jcir)

    def test_jcir_triplet_raises_exactness_error(self) -> None:
        """Calling triplet raises ExactnessError due to state-dependent diffusion and jumps."""
        jcir = JCIRProcess()
        with pytest.raises(ExactnessError, match="triplet"):
            _ = jcir.triplet

    def test_jcir_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on invalid inputs."""
        with pytest.raises(ValueError, match="kappa"):
            JCIRProcess(kappa=-1.0)
        with pytest.raises(ValueError, match="theta"):
            JCIRProcess(theta=-0.05)
        with pytest.raises(ValueError, match="sigma"):
            JCIRProcess(sigma=-0.1)
        with pytest.raises(ValueError, match="lambda_j"):
            JCIRProcess(lambda_j=-0.5)
        with pytest.raises(ValueError, match="mu_j"):
            JCIRProcess(mu_j=-0.01)
        with pytest.raises(ValueError, match="r0"):
            JCIRProcess(r0=-0.01)

    def test_jcir_feller_properties(self) -> None:
        """Feller ratio 2*kappa*theta / sigma^2."""
        jcir_satisfied = JCIRProcess(kappa=2.0, theta=0.05, sigma=0.1)
        assert jcir_satisfied.feller_ratio == pytest.approx(2.0 * 2.0 * 0.05 / (0.1**2))
        assert jcir_satisfied.feller_satisfied is True

        jcir_violated = JCIRProcess(kappa=1.0, theta=0.02, sigma=0.3)
        assert jcir_violated.feller_ratio < 1.0
        assert jcir_violated.feller_satisfied is False

    def test_jcir_stationary_and_conditional_moments(self) -> None:
        """Stationary and conditional moments match exact formulas."""
        kappa = 1.2
        theta = 0.05
        sigma = 0.15
        lambda_j = 0.8
        mu_j = 0.03
        r0 = 0.02
        jcir = JCIRProcess(
            kappa=kappa, theta=theta, sigma=sigma, lambda_j=lambda_j, mu_j=mu_j, r0=r0
        )

        # Stationary moments
        expected_stat_mean = theta + (lambda_j * mu_j) / kappa
        expected_stat_var = ((sigma**2) * expected_stat_mean + 2.0 * lambda_j * (mu_j**2)) / (
            2.0 * kappa
        )

        assert jcir.stationary_mean == pytest.approx(expected_stat_mean, rel=1e-12)
        assert jcir.stationary_variance == pytest.approx(expected_stat_var, rel=1e-12)
        assert jcir.stationary_std == pytest.approx(math.sqrt(expected_stat_var), rel=1e-12)

        # Conditional moments at t
        for t in [0.5, 1.0, 2.0]:
            exp_kt = math.exp(-kappa * t)
            exp_2kt = math.exp(-2.0 * kappa * t)
            cond_mean = r0 * exp_kt + expected_stat_mean * (1.0 - exp_kt)
            cond_var = (sigma**2 * (r0 - expected_stat_mean) / kappa) * (
                exp_kt - exp_2kt
            ) + expected_stat_var * (1.0 - exp_2kt)

            assert jcir.conditional_mean(t) == pytest.approx(cond_mean, rel=1e-12)
            assert jcir.conditional_variance(t) == pytest.approx(cond_var, rel=1e-12)
            assert jcir.conditional_std(t) == pytest.approx(math.sqrt(cond_var), rel=1e-12)

    def test_jcir_reduces_to_cir_when_lambda_is_zero(self) -> None:
        """When lambda_j = 0, JCIR moments and bond prices reduce to baseline CIR."""
        kappa = 1.5
        theta = 0.04
        sigma = 0.12
        r0 = 0.03
        jcir_zero_jump = JCIRProcess(
            kappa=kappa, theta=theta, sigma=sigma, lambda_j=0.0, mu_j=0.02, r0=r0
        )

        assert jcir_zero_jump.stationary_mean == pytest.approx(theta, rel=1e-12)
        assert jcir_zero_jump.stationary_variance == pytest.approx(
            (sigma**2 * theta) / (2.0 * kappa), rel=1e-12
        )

        t = 1.0
        exp_kt = math.exp(-kappa * t)
        cir_cond_var = r0 * (sigma**2 / kappa) * (exp_kt - math.exp(-2.0 * kappa * t)) + theta * (
            sigma**2 / (2.0 * kappa)
        ) * ((1.0 - exp_kt) ** 2)
        assert jcir_zero_jump.conditional_variance(t) == pytest.approx(cir_cond_var, rel=1e-12)

    def test_jcir_zero_coupon_bond_pricing_properties(self) -> None:
        """Bond price P(0, 0)=1, decreases monotonically with maturity, forward rate consistency."""
        jcir = JCIRProcess(kappa=1.2, theta=0.05, sigma=0.15, lambda_j=0.8, mu_j=0.03, r0=0.04)

        # At tau = 0
        assert jcir.bond_price(0.0) == pytest.approx(1.0, abs=1e-12)
        assert jcir.forward_rate(0.0) == pytest.approx(0.04, abs=1e-12)

        # Monotonicity across maturities
        maturities = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
        prices = jcir.bond_price(maturities)
        assert np.all(prices > 0.0)
        assert np.all(np.diff(prices) < 0.0)  # strictly decreasing

        # Yields are positive
        yields = jcir.yield_curve(maturities)
        assert np.all(yields > 0.0)

        # Forward rate matches numerical derivative of -ln P(tau)
        tau = 2.0
        h = 1e-6
        p_plus = jcir.bond_price(tau + h)
        p_minus = jcir.bond_price(tau - h)
        num_fwd = -(math.log(p_plus) - math.log(p_minus)) / (2.0 * h)
        ana_fwd = jcir.forward_rate(tau)
        assert ana_fwd == pytest.approx(num_fwd, rel=1e-5)

    def test_jcir_bond_pricing_vs_monte_carlo(self) -> None:
        """Analytical bond price matches Monte Carlo discounted expectation."""
        kappa = 1.2
        theta = 0.05
        sigma = 0.15
        lambda_j = 0.8
        mu_j = 0.03
        r0 = 0.04
        tau = 1.0

        jcir = JCIRProcess(
            kappa=kappa, theta=theta, sigma=sigma, lambda_j=lambda_j, mu_j=mu_j, r0=r0
        )
        ana_price = jcir.bond_price(tau)

        # Monte Carlo discount expectation
        dt = 0.005
        n_steps = int(tau / dt)
        paths = jcir.simulate(
            n_steps=n_steps, time_horizon=tau, n_paths=40_000, rng=np.random.default_rng(2026)
        )
        int_r = np.sum(paths[:, :-1], axis=1) * dt
        mc_price = float(np.mean(np.exp(-int_r)))

        assert mc_price == pytest.approx(ana_price, rel=0.01)

    def test_jcir_characteristic_function(self) -> None:
        """Characteristic function phi(0)=1, modulus <= 1, Hermitian symmetry."""
        jcir = JCIRProcess(kappa=1.2, theta=0.05, sigma=0.15, lambda_j=0.8, mu_j=0.03, r0=0.04)
        t = 1.0

        # At u = 0
        assert jcir.char_func(0.0, t=t) == pytest.approx(1.0 + 0.0j, abs=1e-12)

        # Modulus <= 1
        u_vals = np.array([-2.0, -0.5, 0.5, 2.0])
        cf_vals = jcir.char_func(u_vals, t=t)
        assert np.all(np.abs(cf_vals) <= 1.0 + 1e-12)

        # Hermitian symmetry: phi(-u) = conj(phi(u))
        u = 1.2
        cf_pos = jcir.char_func(u, t=t)
        cf_neg = jcir.char_func(-u, t=t)
        assert complex(cf_neg) == pytest.approx(complex(np.conj(cf_pos)), rel=1e-10)

    def test_jcir_simulation_paths_and_mean(self) -> None:
        """Simulated paths are non-negative and converge to theoretical conditional moments."""
        kappa = 1.2
        theta = 0.05
        sigma = 0.15
        lambda_j = 0.8
        mu_j = 0.03
        r0 = 0.04
        t = 1.0
        jcir = JCIRProcess(
            kappa=kappa, theta=theta, sigma=sigma, lambda_j=lambda_j, mu_j=mu_j, r0=r0
        )

        paths = jcir.simulate(
            n_steps=25, time_horizon=t, n_paths=25_000, rng=np.random.default_rng(2026)
        )
        assert paths.shape == (25_000, 26)
        assert np.all(paths[:, 0] == r0)
        assert np.all(paths >= 0.0)

        mc_mean = float(np.mean(paths[:, -1]))
        theo_mean = jcir.conditional_mean(t)
        assert mc_mean == pytest.approx(theo_mean, rel=0.02)
