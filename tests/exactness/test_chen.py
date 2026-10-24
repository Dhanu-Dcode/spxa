"""
Tests for ChenModel (Chen 1996 three-factor short rate model) exactness, moments, and simulation.
"""

import math
import numpy as np
import pytest

from spxa import (
    ChenModel,
    ChenProcess,
    ChenShortRate,
    ChenThreeFactor,
    ExactnessError,
    ExactnessLevel,
)


class TestChenExactness:
    """Mathematical and exactness verification for Chen Three-Factor Model."""

    def test_chen_exactness_level_and_aliases(self) -> None:
        """Process exactness level is MOMENT_PROPAGATION and aliases instantiate correctly."""
        model = ChenModel()
        assert model.exactness == ExactnessLevel.MOMENT_PROPAGATION
        assert ChenThreeFactor is ChenModel
        assert ChenShortRate is ChenModel
        assert ChenProcess is ChenModel

        m2 = ChenThreeFactor(k=2.0, theta_bar=0.06, kappa=1.0, alpha=0.05)
        assert m2.k == 2.0
        assert m2.theta_bar == 0.06
        assert m2.exactness == ExactnessLevel.MOMENT_PROPAGATION

    def test_chen_triplet_raises_exactness_error(self) -> None:
        """Calling triplet() must raise ExactnessError."""
        model = ChenModel()
        with pytest.raises(ExactnessError) as exc_info:
            model.triplet()
        assert exc_info.value.method == "triplet"
        assert exc_info.value.required == "EXACT"
        assert exc_info.value.actual == "MOMENT_PROPAGATION"

    def test_chen_invalid_parameters_raise(self) -> None:
        """Invalid model parameters raise ValueError."""
        with pytest.raises(ValueError, match="k"):
            ChenModel(k=-1.0)
        with pytest.raises(ValueError, match="theta_bar"):
            ChenModel(theta_bar=-0.05)
        with pytest.raises(ValueError, match="kappa"):
            ChenModel(kappa=-0.8)
        with pytest.raises(ValueError, match="alpha"):
            ChenModel(alpha=-0.04)
        with pytest.raises(ValueError, match="zeta_bar"):
            ChenModel(zeta_bar=-0.02)
        with pytest.raises(ValueError, match="lambda_param"):
            ChenModel(lambda_param=-1.0)
        with pytest.raises(ValueError, match="beta"):
            ChenModel(beta=-0.05)
        with pytest.raises(ValueError, match="r0"):
            ChenModel(r0=-0.04)

    def test_chen_feller_conditions_and_stationary_means(self) -> None:
        """Feller conditions and stationary expectations."""
        # Case where 2*kappa*theta_bar >= alpha^2 and 2*lambda*zeta_bar >= beta^2
        # 2 * 0.8 * 0.05 = 0.08 >= 0.04^2 = 0.0016 (True)
        # 2 * 1.0 * 0.02 = 0.04 >= 0.05^2 = 0.0025 (True)
        model = ChenModel(
            k=1.5, theta_bar=0.05, kappa=0.8, alpha=0.04,
            zeta_bar=0.02, lambda_param=1.0, beta=0.05,
        )
        assert model.feller_theta is True
        assert model.feller_zeta is True
        assert model.stationary_mean_short_rate == pytest.approx(0.05)
        assert model.stationary_mean_central_tendency == pytest.approx(0.05)
        assert model.stationary_mean_volatility == pytest.approx(0.02)

    def test_chen_exact_conditional_means(self) -> None:
        """Exact conditional expectations of r_t, theta_t, and zeta_t."""
        model = ChenModel(
            k=1.5, theta_bar=0.05, kappa=0.8, alpha=0.04,
            zeta_bar=0.02, lambda_param=1.0, beta=0.05,
            r0=0.03, theta0=0.06, zeta0=0.015,
        )

        # At t = 0
        assert model.conditional_mean(t=0.0) == pytest.approx(0.03)
        assert model.conditional_mean_central_tendency(t=0.0) == pytest.approx(0.06)
        assert model.conditional_mean_volatility(t=0.0) == pytest.approx(0.015)

        # At horizon t = 1.0
        e_th = model.conditional_mean_central_tendency(t=1.0)
        expected_th = 0.05 + (0.06 - 0.05) * math.exp(-0.8 * 1.0)
        assert e_th == pytest.approx(expected_th)

        e_zeta = model.conditional_mean_volatility(t=1.0)
        expected_zeta = 0.02 + (0.015 - 0.02) * math.exp(-1.0 * 1.0)
        assert e_zeta == pytest.approx(expected_zeta)

        e_r = model.conditional_mean(t=1.0)
        # Theoretical formula
        base = math.exp(-1.5) * 0.03 + 0.05 * (1.0 - math.exp(-1.5))
        corr = (1.5 * (0.06 - 0.05) / (0.8 - 1.5)) * (math.exp(-1.5) - math.exp(-0.8))
        assert e_r == pytest.approx(base + corr)

        # As t -> infty, mean approaches theta_bar
        assert model.conditional_mean(t=50.0) == pytest.approx(0.05, abs=1e-5)

    def test_chen_matrix_exponential_moments_and_mc(self) -> None:
        """Matrix exponential moment system matches Monte Carlo."""
        model = ChenModel(
            k=1.5, theta_bar=0.05, kappa=0.8, alpha=0.04,
            zeta_bar=0.02, lambda_param=1.0, beta=0.05,
            r0=0.04, theta0=0.05, zeta0=0.02,
        )

        moments = model.conditional_moments_system(t=1.0)
        mean_r = model.conditional_mean(t=1.0)
        sec_mom = model.conditional_second_moment(t=1.0)
        var_r = model.conditional_variance(t=1.0)
        std_r = model.conditional_std(t=1.0)

        assert moments["r"] == pytest.approx(mean_r)
        assert sec_mom == pytest.approx(moments["r2"])
        assert var_r == pytest.approx(sec_mom - mean_r ** 2)
        assert std_r == pytest.approx(math.sqrt(var_r))

        # Check Monte Carlo convergence
        rng = np.random.default_rng(42)
        r_paths = model.simulate(n_steps=150, time_horizon=1.0, n_paths=30000, rng=rng)
        r_term = r_paths[:, -1]

        assert float(np.mean(r_term)) == pytest.approx(mean_r, rel=0.02)
        assert float(np.var(r_term)) == pytest.approx(var_r, rel=0.15)

    def test_chen_bond_pricing_and_yield_curve(self) -> None:
        """Bond prices decrease with maturity and yields remain positive."""
        model = ChenModel(r0=0.04, theta_bar=0.05, zeta_bar=0.02)
        rng = np.random.default_rng(101)

        p1 = model.bond_price(maturity=0.5, n_steps=50, n_paths=5000, rng=rng)
        p2 = model.bond_price(maturity=1.0, n_steps=50, n_paths=5000, rng=rng)

        assert 0.0 < p2 < p1 <= 1.0

        maturities = [0.25, 0.5, 1.0]
        yields = model.yield_curve(maturities, n_steps=40, n_paths=4000, rng=rng)
        assert len(yields) == 3
        assert np.all(yields > 0.0)

    def test_chen_simulation_factors_and_positivity(self) -> None:
        """Coupled factors simulation produces non-negative paths for all 3 state variables."""
        model = ChenModel(r0=0.03, theta0=0.04, zeta0=0.015)
        rng = np.random.default_rng(77)

        r_p, th_p, z_p = model.simulate_factors(
            n_steps=20, time_horizon=0.5, n_paths=15, rng=rng
        )

        assert r_p.shape == (15, 21)
        assert th_p.shape == (15, 21)
        assert z_p.shape == (15, 21)
        assert np.all(r_p[:, 0] == 0.03)
        assert np.all(th_p[:, 0] == 0.04)
        assert np.all(z_p[:, 0] == 0.015)
        assert np.all(r_p >= 0.0)
        assert np.all(th_p >= 0.0)
        assert np.all(z_p >= 0.0)
