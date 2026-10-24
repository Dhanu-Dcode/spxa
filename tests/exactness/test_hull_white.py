"""
Exactness and mathematical consistency tests for Hull-White (1990) one-factor short-rate model.
"""

import math
import numpy as np
import pytest

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.zoo.beyond import ExtendedVasicek, HW1F, HullWhite, HullWhiteProcess


class TestHullWhiteExactness:
    """Mathematical exactness and consistency test suite for HullWhiteProcess."""

    def test_hw_exactness_level_and_aliases(self) -> None:
        """HullWhiteProcess is MOMENT_PROPAGATION and identical to HullWhite, HW1F, ExtendedVasicek."""
        hw = HullWhiteProcess(a=0.15, sigma=0.025, b=0.035, r0=0.03)
        assert hw.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert HullWhite is HullWhiteProcess
        assert HW1F is HullWhiteProcess
        assert ExtendedVasicek is HullWhiteProcess
        assert "HullWhiteProcess" in repr(hw)
        assert "a=0.15" in repr(hw)
        assert "sigma=0.025" in repr(hw)

    def test_hw_triplet_raises_exactness_error(self) -> None:
        """Calling triplet raises ExactnessError with mathematical explanation."""
        hw = HullWhiteProcess()
        with pytest.raises(ExactnessError, match="MOMENT_PROPAGATION"):
            _ = hw.triplet

    def test_hw_invalid_parameters_raise(self) -> None:
        """Constructor raises ValueError on non-positive mean-reversion speed or volatility."""
        with pytest.raises(ValueError, match="speed 'a'"):
            HullWhiteProcess(a=-0.05)
        with pytest.raises(ValueError, match="speed 'a'"):
            HullWhiteProcess(a=0.0)
        with pytest.raises(ValueError, match="Volatility 'sigma'"):
            HullWhiteProcess(sigma=0.0)
        with pytest.raises(ValueError, match="Volatility 'sigma'"):
            HullWhiteProcess(sigma=-0.01)

    def test_hw_properties(self) -> None:
        """Structural properties reflect state-dependent mean-reverting diffusion."""
        hw = HullWhiteProcess()
        props = hw.properties
        assert not props.is_subordinator
        assert not props.has_independent_increments
        assert not props.has_stationary_increments
        assert props.has_finite_mean
        assert props.has_finite_variance
        assert props.self_similarity_index is None
        assert len(props.notes) >= 2

    def test_hw_conditional_mean_and_variance(self) -> None:
        """Conditional moments match Gaussian affine formulas."""
        a = 0.2
        sigma = 0.03
        b = 0.05
        r0 = 0.02
        hw = HullWhiteProcess(a=a, sigma=sigma, b=b, r0=r0)

        # At t=0: mean is r0, variance is 0
        assert hw.conditional_mean(0.0) == pytest.approx(r0)
        assert hw.conditional_variance(0.0) == pytest.approx(0.0, abs=1e-12)
        assert hw.conditional_std(0.0) == pytest.approx(0.0, abs=1e-12)

        # At t > 0
        t = 2.0
        theor_mean = b + (r0 - b) * math.exp(-a * t)
        theor_var = (sigma**2 / (2.0 * a)) * (1.0 - math.exp(-2.0 * a * t))
        assert hw.conditional_mean(t) == pytest.approx(theor_mean, rel=1e-12)
        assert hw.conditional_variance(t) == pytest.approx(theor_var, rel=1e-12)
        assert hw.conditional_std(t) == pytest.approx(math.sqrt(theor_var), rel=1e-12)

        # As t -> inf: approaches stationary values
        assert hw.conditional_mean(100.0) == pytest.approx(hw.stationary_mean, rel=1e-6)
        assert hw.conditional_variance(100.0) == pytest.approx(hw.stationary_variance, rel=1e-6)

    def test_hw_integrated_moments_and_autocovariance(self) -> None:
        """Integrated short rate moments and autocovariance match theory."""
        a = 0.15
        sigma = 0.02
        b = 0.04
        r0 = 0.03
        hw = HullWhiteProcess(a=a, sigma=sigma, b=b, r0=r0)

        t = 1.5
        theor_int_mean = b * t + (r0 - b) * (1.0 - math.exp(-a * t)) / a
        term1 = t
        term2 = -2.0 * (1.0 - math.exp(-a * t)) / a
        term3 = (1.0 - math.exp(-2.0 * a * t)) / (2.0 * a)
        theor_int_var = (sigma**2 / (a**2)) * (term1 + term2 + term3)

        assert hw.integrated_mean(t) == pytest.approx(theor_int_mean, rel=1e-12)
        assert hw.integrated_variance(t) == pytest.approx(theor_int_var, rel=1e-12)

        # Autocovariance
        t1, t2 = 1.0, 2.5
        theor_cov = (sigma**2 / (2.0 * a)) * (
            math.exp(-a * abs(t1 - t2)) - math.exp(-a * (t1 + t2))
        )
        assert hw.autocovariance(t1, t2) == pytest.approx(theor_cov, rel=1e-12)
        assert hw.autocovariance(t1, t2) == hw.autocovariance(t2, t1)

    def test_hw_characteristic_function_and_cumulants(self) -> None:
        """Characteristic function and cumulants satisfy Gaussian distribution requirements."""
        hw = HullWhiteProcess(a=0.1, sigma=0.02, b=0.04, r0=0.03)
        t = 1.0

        # phi(0) == 1
        assert hw.char_func(0.0, t=t) == pytest.approx(1.0 + 0j)

        # Cumulants
        k = hw.conditional_cumulants(t=t, order=4)
        assert k[1] == pytest.approx(hw.conditional_mean(t))
        assert k[2] == pytest.approx(hw.conditional_variance(t))
        assert k[3] == 0.0
        assert k[4] == 0.0

    def test_hw_zero_coupon_bond_pricing_and_yield_curve(self) -> None:
        """Zero-coupon bond pricing satisfies boundary conditions and yield relations."""
        a = 0.1
        sigma = 0.015
        b = 0.05
        r0 = 0.03
        hw = HullWhiteProcess(a=a, sigma=sigma, b=b, r0=r0)

        # P(t, t) == 1
        assert hw.bond_price(0.0) == pytest.approx(1.0)
        assert hw.bond_price(1.0, t=1.0) == pytest.approx(1.0)

        # P(0, T) strictly decreases with maturity
        maturities = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        prices = hw.bond_price(maturities)
        assert np.all(np.diff(prices) < 0)

        # Yield to maturity: y(t, T) = -ln P(t, T) / (T - t)
        yields = hw.yield_to_maturity(maturities)
        for mat, p, y in zip(maturities, prices, yields):
            assert y == pytest.approx(-math.log(p) / mat, rel=1e-12)

        # Instantaneous forward rate matches analytical derivative
        fwd_0 = hw.instantaneous_forward_rate(0.0)
        assert fwd_0 == pytest.approx(r0, rel=1e-12)

    def test_hw_bond_options_put_call_parity(self) -> None:
        """European bond options satisfy exact Put-Call parity."""
        a = 0.1
        sigma = 0.02
        b = 0.04
        r0 = 0.03
        hw = HullWhiteProcess(a=a, sigma=sigma, b=b, r0=r0)

        T = 1.0
        S = 4.0
        strikes = [0.80, 0.85, 0.90, 0.95]

        p_0_S = hw.bond_price(S)
        p_0_T = hw.bond_price(T)

        for K in strikes:
            call = hw.bond_call_price(T=T, S=S, K=K)
            put = hw.bond_put_price(T=T, S=S, K=K)

            # Put-Call Parity: C - P = P(0, S) - K * P(0, T)
            assert (call - put) == pytest.approx(p_0_S - K * p_0_T, abs=1e-12)

    def test_hw_caplet_and_floorlet_parity(self) -> None:
        """Caplet and Floorlet prices satisfy caplet-floorlet parity."""
        a = 0.12
        sigma = 0.018
        b = 0.035
        r0 = 0.025
        hw = HullWhiteProcess(a=a, sigma=sigma, b=b, r0=r0)

        T = 1.0
        tau = 0.5
        S = T + tau
        R_K = 0.03

        caplet = hw.caplet_price(T=T, strike_rate=R_K, tau=tau)
        floorlet = hw.floorlet_price(T=T, strike_rate=R_K, tau=tau)

        # Caplet - Floorlet = Swaplet = P(0, T) - (1 + tau * R_K) * P(0, S)
        # using standard Libor forward swaplet formula
        p_T = hw.bond_price(T)
        p_S = hw.bond_price(S)
        swaplet = p_T - (1.0 + tau * R_K) * p_S
        assert (caplet - floorlet) == pytest.approx(swaplet, abs=1e-12)

    def test_hw_bond_option_pricing_vs_monte_carlo(self) -> None:
        """Jamshidian bond call price agrees with exact Monte Carlo simulation."""
        a = 0.1
        sigma = 0.02
        b = 0.05
        r0 = 0.03
        T = 1.0
        S = 3.0
        K = 0.85
        hw = HullWhiteProcess(a=a, sigma=sigma, b=b, r0=r0)

        analytical_call = hw.bond_call_price(T=T, S=S, K=K)

        # Monte Carlo simulation
        n_paths = 100_000
        n_steps = 50
        dt = T / n_steps
        rng = np.random.default_rng(42)

        paths = hw.simulate(
            n_steps=n_steps, time_horizon=T, n_paths=n_paths, rng=rng, method="exact"
        )
        final_r = paths[:, -1]

        # Discount factor exp(-int_0^T r_s ds) via trapezoid integration
        int_r = np.trapezoid(paths, dx=dt, axis=1)
        discount = np.exp(-int_r)

        # Bond price at T: P(T, S; r_T)
        p_T_S = hw.bond_price(S, r=final_r, t=T)
        payoff = np.maximum(p_T_S - K, 0.0)

        mc_call = float(np.mean(discount * payoff))
        mc_se = float(np.std(discount * payoff) / math.sqrt(n_paths))

        assert abs(analytical_call - mc_call) < 3.0 * mc_se

    def test_hw_simulation_paths_shapes_and_schemes(self) -> None:
        """Simulation produces correct shapes, starting values, and supports exact and euler."""
        hw = HullWhiteProcess(a=0.1, sigma=0.02, b=0.03, r0=0.025)
        n_paths = 15
        n_steps = 25
        rng = np.random.default_rng(99)

        for meth in ["exact", "euler"]:
            paths = hw.simulate(
                n_steps=n_steps, time_horizon=1.0, n_paths=n_paths, rng=rng, method=meth
            )
            assert paths.shape == (n_paths, n_steps + 1)
            assert np.all(paths[:, 0] == hw.r0)
