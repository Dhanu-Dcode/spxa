"""
Tests for RegimeSwitchingGBM (RS-GBM) exactness, moments, Fourier option pricing, and simulation.
"""

import math
import numpy as np
import pytest

from spxa import (
    ExactnessError,
    ExactnessLevel,
    MarkovSwitchingGBM,
    RegimeSwitchingBlackScholes,
    RegimeSwitchingGBM,
    RSGBM,
)


class TestRegimeSwitchingGBMExactness:
    """Mathematical and exactness verification for RegimeSwitchingGBM."""

    def test_rs_gbm_exactness_level_and_aliases(self) -> None:
        """Process exactness level is MOMENT_PROPAGATION and aliases instantiate correctly."""
        proc = RegimeSwitchingGBM()
        assert proc.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert RSGBM is RegimeSwitchingGBM
        assert RegimeSwitchingBlackScholes is RegimeSwitchingGBM
        assert MarkovSwitchingGBM is RegimeSwitchingGBM

        p2 = RSGBM(mu=[0.05, -0.02], sigma=[0.12, 0.28])
        assert p2.n_regimes == 2
        assert p2.exactness == ExactnessLevel.MOMENT_PROPAGATION

    def test_rs_gbm_triplet_raises_exactness_error(self) -> None:
        """Calling triplet() must raise ExactnessError."""
        proc = RegimeSwitchingGBM()
        with pytest.raises(ExactnessError) as exc_info:
            proc.triplet()
        assert exc_info.value.method == "triplet"
        assert exc_info.value.required == "EXACT"
        assert exc_info.value.actual == "MOMENT_PROPAGATION"

    def test_rs_gbm_invalid_parameters_raise(self) -> None:
        """Invalid model parameters raise ValueError."""
        with pytest.raises(ValueError, match="match"):
            RegimeSwitchingGBM(mu=[0.05, 0.02], sigma=[0.2])
        with pytest.raises(ValueError, match="strictly positive"):
            RegimeSwitchingGBM(mu=[0.05, 0.02], sigma=[0.2, -0.1])
        with pytest.raises(ValueError, match="s0"):
            RegimeSwitchingGBM(s0=-10.0)
        with pytest.raises(ValueError, match="initial_regime"):
            RegimeSwitchingGBM(initial_regime=5)
        with pytest.raises(ValueError, match="zero"):
            RegimeSwitchingGBM(transition_rates=[[0.5, 0.5], [0.5, 0.5]])

    def test_rs_gbm_ctmc_transition_and_stationary(self) -> None:
        """Markov chain transition matrix and stationary distribution properties."""
        q = np.array([[-0.5, 0.5], [1.0, -1.0]])
        proc = RegimeSwitchingGBM(mu=[0.08, -0.04], sigma=[0.15, 0.35], transition_rates=q)

        # At t = 0, P(0) = Identity
        p0 = proc.transition_probabilities(t=0.0)
        np.testing.assert_allclose(p0, np.eye(2), atol=1e-12)

        # For any t > 0, rows sum to 1
        pt = proc.transition_probabilities(t=1.5)
        np.testing.assert_allclose(np.sum(pt, axis=1), [1.0, 1.0], atol=1e-12)

        # Stationary distribution pi * Q = 0
        pi = proc.stationary_distribution
        assert np.isclose(np.sum(pi), 1.0)
        np.testing.assert_allclose(pi @ q, [0.0, 0.0], atol=1e-10)
        # For lambda_01 = 0.5, lambda_10 = 1.0: pi = [2/3, 1/3]
        np.testing.assert_allclose(pi, [2.0 / 3.0, 1.0 / 3.0], atol=1e-10)

    def test_rs_gbm_matrix_exponential_moments(self) -> None:
        """Matrix exponential moments match theoretical formulas and Monte Carlo."""
        proc = RegimeSwitchingGBM(
            mu=[0.10, -0.05],
            sigma=[0.15, 0.35],
            transition_rates=[[-0.5, 0.5], [1.0, -1.0]],
            s0=100.0,
            initial_regime=0,
        )

        # At t = 0
        assert proc.conditional_mean(t=0.0) == pytest.approx(100.0)
        assert proc.conditional_second_moment(t=0.0) == pytest.approx(10000.0)
        assert proc.conditional_variance(t=0.0) == pytest.approx(0.0)

        # At horizon T = 1.0
        m1 = proc.conditional_mean(t=1.0)
        m2 = proc.conditional_second_moment(t=1.0)
        var = proc.conditional_variance(t=1.0)
        std = proc.conditional_std(t=1.0)

        assert m2 > m1**2
        assert var == pytest.approx(m2 - m1**2)
        assert std == pytest.approx(math.sqrt(var))

        # Check Monte Carlo convergence
        rng = np.random.default_rng(123)
        paths = proc.simulate(n_steps=200, time_horizon=1.0, n_paths=40000, rng=rng)
        s_t = paths[:, -1]

        mc_m1 = float(np.mean(s_t))
        mc_var = float(np.var(s_t))

        assert mc_m1 == pytest.approx(m1, rel=0.01)
        assert mc_var == pytest.approx(var, rel=0.05)

    def test_rs_gbm_characteristic_function_properties(self) -> None:
        """Exact continuous characteristic function satisfies analytical properties."""
        proc = RegimeSwitchingGBM(
            mu=[0.08, -0.04],
            sigma=[0.18, 0.32],
            s0=100.0,
            initial_regime=0,
        )

        # phi(0) = 1
        phi_0 = proc.char_func(0.0, t=1.0)
        assert phi_0 == pytest.approx(1.0 + 0.0j, abs=1e-12)

        # Modulus <= 1
        u_vals = np.array([0.5, 1.0, 2.0, 5.0])
        phi_vals = proc.char_func(u_vals, t=1.0)
        assert np.all(np.abs(phi_vals) <= 1.0 + 1e-12)

        # Conjugate symmetry: phi(-u) = conj(phi(u))
        phi_pos = proc.char_func(1.5, t=1.0)
        phi_neg = proc.char_func(-1.5, t=1.0)
        assert phi_neg == pytest.approx(np.conj(phi_pos), abs=1e-12)

    def test_rs_gbm_option_pricing_and_put_call_parity(self) -> None:
        """European option pricing via Gil-Pelaez inversion satisfies put-call parity."""
        proc = RegimeSwitchingGBM(
            mu=[0.05, 0.05],  # Risk-neutral drift r = 0.05
            sigma=[0.15, 0.30],
            s0=100.0,
            initial_regime=0,
        )

        tau = 0.75
        r = 0.05
        strike = 100.0

        call = proc.call_price(strike=strike, time_to_maturity=tau, discount_rate=r)
        put = proc.put_price(strike=strike, time_to_maturity=tau, discount_rate=r)

        # Put-Call parity: C - P = exp(-r*T) * (F - K)
        forward = proc.conditional_mean(t=tau)
        parity_rhs = math.exp(-r * tau) * (forward - strike)
        assert call - put == pytest.approx(parity_rhs, rel=1e-4)

        # Compare call price with Monte Carlo
        rng = np.random.default_rng(42)
        paths = proc.simulate(n_steps=150, time_horizon=tau, n_paths=40000, rng=rng)
        s_t = paths[:, -1]
        mc_call = float(math.exp(-r * tau) * np.mean(np.maximum(s_t - strike, 0.0)))
        assert call == pytest.approx(mc_call, rel=0.03)

    def test_rs_gbm_simulation_joint_and_positivity(self) -> None:
        """Simulated paths remain strictly positive and joint regimes match valid states."""
        proc = RegimeSwitchingGBM(s0=50.0, initial_regime=1)
        rng = np.random.default_rng(88)

        s_paths, reg_paths = proc.simulate_joint(n_steps=30, time_horizon=0.5, n_paths=20, rng=rng)

        assert s_paths.shape == (20, 31)
        assert reg_paths.shape == (20, 31)
        assert np.all(s_paths[:, 0] == 50.0)
        assert np.all(reg_paths[:, 0] == 1)
        assert np.all(s_paths > 0.0)
        assert np.all((reg_paths == 0) | (reg_paths == 1))
