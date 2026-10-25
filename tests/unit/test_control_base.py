"""Unit tests for continuous-time stochastic control base abstractions."""

import numpy as np
import pytest

from spxa.control.base import ControlledTrajectory, ControlProblem


class SimpleDriftControl(ControlProblem):
    """Simple 1D control problem: dX_t = u_t dt + sigma dW_t, minimize int_0^T u^2 dt + X_T^2."""

    def __init__(self, t_terminal: float = 1.0, sigma: float = 0.5):
        super().__init__(t_terminal=t_terminal)
        self.sigma = sigma

    def running_cost(
        self, t: float, x: float | np.ndarray, u: float | np.ndarray
    ) -> float | np.ndarray:
        return u**2

    def terminal_cost(self, x: float | np.ndarray) -> float | np.ndarray:
        return x**2

    def drift(self, t: float, x: float | np.ndarray, u: float | np.ndarray) -> float | np.ndarray:
        return u

    def diffusion(
        self, t: float, x: float | np.ndarray, u: float | np.ndarray
    ) -> float | np.ndarray:
        return self.sigma

    def optimal_control(self, t: float, x: float | np.ndarray) -> float | np.ndarray:
        tau = self.t_terminal - t
        # For dX = u dt, min int u^2 + X_T^2: P(t) = 1 / (1 + T - t)
        p_t = 1.0 / (1.0 + tau)
        return -p_t * x

    def value_function(self, t: float, x: float) -> float:
        tau = self.t_terminal - t
        p_t = 1.0 / (1.0 + tau)
        # s(t) = int_t^T sigma^2 P(s) ds = sigma^2 ln(1 + tau)
        s_t = (self.sigma**2) * np.log(1.0 + tau)
        return float(p_t * (x**2) + s_t)


class TestControlBase:
    def test_control_problem_validation(self):
        with pytest.raises(ValueError, match="Terminal horizon"):
            SimpleDriftControl(t_terminal=-1.0)
        with pytest.raises(ValueError, match="Discount rate"):
            SimpleDriftControl(t_terminal=1.0).discount_rate = -0.5

    def test_controlled_trajectory_properties(self):
        times = np.linspace(0, 1, 11)
        states = np.zeros((10, 11))
        controls = np.zeros((10, 10))
        costs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        traj = ControlledTrajectory(
            times=times,
            states=states,
            controls=controls,
            realized_costs=costs,
        )
        assert traj.mean_cost == pytest.approx(5.5, rel=1e-7)
        assert traj.std_cost == pytest.approx(np.std(costs), rel=1e-7)

    def test_simulation_matches_expected_value(self):
        problem = SimpleDriftControl(t_terminal=1.0, sigma=0.4)
        x0 = 2.0
        traj = problem.simulate(x0=x0, num_steps=50, num_paths=2000, seed=42)

        assert traj.states.shape == (2000, 51)
        assert traj.controls.shape == (2000, 50)
        assert traj.times.shape == (51,)
        assert len(traj.realized_costs) == 2000

        # Mean realized cost should approximate analytical value function at t=0
        v0 = problem.value_function(0.0, x0)
        assert traj.mean_cost == pytest.approx(v0, rel=0.08)
