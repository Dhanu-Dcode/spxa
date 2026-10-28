"""Exactness tests for Merton's Optimal Portfolio & Consumption Model."""

import numpy as np
import pytest

from spxa.control.merton import (
    MertonControl,
    MertonPortfolio,
    MertonPortfolioControl,
    OptimalConsumptionInvestment,
)


class TestMertonControlExactness:
    def test_merton_aliases_and_initialization(self):
        model = MertonPortfolioControl(mu=0.08, sigma=0.20, r=0.03, gamma=2.0)
        assert MertonControl is MertonPortfolioControl
        assert MertonPortfolio is MertonPortfolioControl
        assert OptimalConsumptionInvestment is MertonPortfolioControl

        # Merton ratio: (0.08 - 0.03) / (2.0 * 0.04) = 0.05 / 0.08 = 0.625
        assert model.optimal_risky_weight == pytest.approx(0.625, rel=1e-7)

    def test_merton_validation_raises(self):
        with pytest.raises(ValueError, match="Asset volatility sigma must be positive"):
            MertonPortfolioControl(sigma=-0.1)
        with pytest.raises(ValueError, match="Risk aversion gamma must be positive"):
            MertonPortfolioControl(gamma=-2.0)
        with pytest.raises(ValueError, match="Logarithmic utility gamma=1.0"):
            MertonPortfolioControl(gamma=1.0)
        with pytest.raises(ValueError, match="Discount rate rho must be positive"):
            MertonPortfolioControl(rho=-0.05)
        with pytest.raises(ValueError, match="Bequest parameter epsilon"):
            MertonPortfolioControl(epsilon_bequest=-1.0)

    def test_merton_consumption_propensity_and_terminal_limit(self):
        model = MertonPortfolioControl(
            mu=0.07, sigma=0.15, r=0.02, gamma=3.0, rho=0.04, epsilon_bequest=2.0, t_terminal=3.0
        )
        # At t = T, kappa(T) = 1 / epsilon^{1/gamma} = 1 / 2^{1/3}
        expected_kappa_t = 1.0 / (2.0 ** (1.0 / 3.0))
        assert model.consumption_propensity(3.0) == pytest.approx(expected_kappa_t, rel=1e-7)

        # Infinite horizon propensity
        assert model.infinite_horizon_consumption_propensity == pytest.approx(model.nu, rel=1e-7)

    def test_merton_hjb_equation_satisfaction(self):
        # Verify that the closed-form V(t, W) exactly satisfies the HJB PDE:
        # \partial_t V - rho V + sup_{pi, c} { [r W + pi(mu - r) - c] \partial_W V + 0.5 * pi^2 sigma^2 \partial_{WW} V + c^{1-gamma}/(1-gamma) } = 0
        model = MertonPortfolioControl(
            mu=0.09, sigma=0.20, r=0.03, gamma=2.5, rho=0.05, epsilon_bequest=1.0, t_terminal=4.0
        )
        w = 50.0
        t = 1.5

        # Analytical value function and derivatives
        h_val = 1e-4
        v_curr = model.value_function(t, w)
        v_t_plus = model.value_function(t + h_val, w)
        v_t_minus = model.value_function(t - h_val, w)
        dt_v = (v_t_plus - v_t_minus) / (2.0 * h_val)

        v_w_plus = model.value_function(t, w + h_val)
        v_w_minus = model.value_function(t, w - h_val)
        dw_v = (v_w_plus - v_w_minus) / (2.0 * h_val)
        dww_v = (v_w_plus - 2.0 * v_curr + v_w_minus) / (h_val**2)

        # Optimal controls
        u_opt = model.optimal_control(t, w)
        pi_star = u_opt[0]
        c_star = u_opt[1]

        # Hamilton-Jacobi-Bellman operator
        hjb = (
            dt_v
            - model.rho * v_curr
            + (model.r * w + pi_star * (model.mu - model.r) - c_star) * dw_v
            + 0.5 * (pi_star**2) * (model.sigma**2) * dww_v
            + (c_star ** (1.0 - model.gamma)) / (1.0 - model.gamma)
        )
        # Should evaluate to zero (HJB PDE residual)
        assert abs(hjb) < 1e-4

    def test_merton_simulation_convergence_to_value_function(self):
        model = MertonPortfolioControl(
            mu=0.08, sigma=0.20, r=0.02, gamma=2.0, rho=0.06, epsilon_bequest=1.0, t_terminal=2.0
        )
        w0 = 100.0
        traj = model.simulate(x0=w0, num_steps=60, num_paths=3000, seed=42)

        assert traj.states.shape == (3000, 61)
        assert traj.controls.shape == (3000, 60, 2)
        assert np.all(traj.states > 0.0)
        assert np.all(traj.controls > 0.0)

        # Monte Carlo mean utility vs analytical value function
        v0 = model.value_function(0.0, w0)
        assert traj.mean_cost == pytest.approx(v0, rel=0.05)
