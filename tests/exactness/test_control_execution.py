"""Exactness tests for Almgren-Chriss Optimal Order Execution Model."""

import math
import numpy as np
import pytest

from spxa.control.execution import (
    AlmgrenChriss,
    AlmgrenChrissExecution,
    AlmgrenChrissModel,
    OptimalExecution,
)


class TestAlmgrenChrissExactness:
    def test_almgren_chriss_aliases_and_initialization(self):
        exec_model = AlmgrenChrissExecution(
            total_shares=50000.0,
            t_terminal=1.0,
            sigma=0.30,
            eta=2.5e-4,
            gamma_impact=2.5e-5,
            risk_aversion=1e-4,
        )
        assert AlmgrenChriss is AlmgrenChrissExecution
        assert AlmgrenChrissModel is AlmgrenChrissExecution
        assert OptimalExecution is AlmgrenChrissExecution

        # Urgency kappa = sqrt(1e-4 * 0.09 / 2.5e-4) = sqrt(9e-6 / 2.5e-4) = sqrt(0.036) ~ 0.189736
        expected_kappa = math.sqrt(1e-4 * 0.09 / 2.5e-4)
        assert exec_model.urgency_parameter == pytest.approx(expected_kappa, rel=1e-6)

    def test_almgren_chriss_validation_raises(self):
        with pytest.raises(ValueError, match="Total shares to liquidate must be non-zero"):
            AlmgrenChrissExecution(total_shares=0.0)
        with pytest.raises(ValueError, match="Price volatility sigma must be positive"):
            AlmgrenChrissExecution(sigma=-0.2)
        with pytest.raises(ValueError, match="Temporary impact parameter eta must be positive"):
            AlmgrenChrissExecution(eta=-1e-4)
        with pytest.raises(
            ValueError, match="Permanent impact parameter gamma must be non-negative"
        ):
            AlmgrenChrissExecution(gamma_impact=-1e-5)
        with pytest.raises(ValueError, match="Risk aversion lambda must be non-negative"):
            AlmgrenChrissExecution(risk_aversion=-1e-3)

    def test_almgren_chriss_twap_limit(self):
        # When risk aversion is 0, execution is purely linear TWAP
        twap = AlmgrenChrissExecution(
            total_shares=10000.0,
            t_terminal=2.0,
            sigma=0.20,
            eta=1e-4,
            gamma_impact=1e-5,
            risk_aversion=0.0,
        )
        assert twap.is_twap is True

        # Inventory at t=0, 1, 2
        assert twap.optimal_inventory(0.0) == pytest.approx(10000.0, rel=1e-7)
        assert twap.optimal_inventory(1.0) == pytest.approx(5000.0, rel=1e-7)
        assert twap.optimal_inventory(2.0) == pytest.approx(0.0, abs=1e-9)

        # Constant velocity = -10000 / 2 = -5000 shares/year
        assert twap.optimal_trading_speed(0.0) == pytest.approx(-5000.0, rel=1e-7)
        assert twap.optimal_trading_speed(1.0) == pytest.approx(-5000.0, rel=1e-7)

        # Expected shortfall for TWAP: 0.5 * gamma * Q0^2 + eta * Q0^2 / T
        expected_shortfall = 0.5 * 1e-5 * (10000.0**2) + 1e-4 * (10000.0**2) / 2.0
        assert twap.expected_shortfall() == pytest.approx(expected_shortfall, rel=1e-7)

        # Variance for TWAP: sigma^2 * Q0^2 * T / 3
        expected_var = (0.2**2) * (10000.0**2) * 2.0 / 3.0
        assert twap.variance_shortfall() == pytest.approx(expected_var, rel=1e-7)

    def test_almgren_chriss_boundary_conditions(self):
        exec_model = AlmgrenChrissExecution(
            total_shares=20000.0,
            t_terminal=1.0,
            sigma=0.25,
            eta=5e-4,
            gamma_impact=5e-5,
            risk_aversion=5e-3,
        )
        # q(0) = Q0 and q(T) = 0
        assert exec_model.optimal_inventory(0.0) == pytest.approx(20000.0, rel=1e-7)
        assert exec_model.optimal_inventory(1.0) == pytest.approx(0.0, abs=1e-7)

    def test_almgren_chriss_efficient_frontier_monotonicity(self):
        exec_model = AlmgrenChrissExecution(total_shares=10000.0, t_terminal=1.0)
        stds, means = exec_model.efficient_frontier(n_points=20)

        # As risk aversion increases:
        # Variance/std decreases (more aggressive execution reduces price risk)
        # Expected shortfall increases (higher trading speed increases temporary impact)
        # Therefore, higher std corresponds to lower expected shortfall!
        assert len(stds) == 20
        assert len(means) == 20
        assert np.all(np.diff(stds) < 0.0)
        assert np.all(np.diff(means) > 0.0)

    def test_almgren_chriss_simulation_convergence(self):
        exec_model = AlmgrenChrissExecution(
            total_shares=1000.0,
            t_terminal=0.5,
            sigma=0.20,
            eta=1e-3,
            gamma_impact=1e-4,
            risk_aversion=1e-3,
            s0=50.0,
        )
        traj = exec_model.simulate(num_steps=50, num_paths=2000, seed=42)

        assert traj.states.shape == (2000, 51)
        assert traj.controls.shape == (2000, 50)
        # Terminal inventory should be close to 0
        assert np.allclose(traj.states[:, -1], 0.0, atol=1e-3)

        # Monte Carlo mean shortfall vs analytical expected shortfall
        analytic_mean = exec_model.expected_shortfall()
        assert traj.mean_cost == pytest.approx(analytic_mean, rel=0.08)

        # Monte Carlo std of shortfall vs analytical std of shortfall
        analytic_std = math.sqrt(exec_model.variance_shortfall())
        assert traj.std_cost == pytest.approx(analytic_std, rel=0.08)
