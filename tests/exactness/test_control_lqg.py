"""Exactness tests for Linear-Quadratic-Gaussian (LQG) Stochastic Optimal Control."""

import numpy as np
import pytest
from scipy import linalg

from spxa.control.lqg import LinearQuadraticGaussian, LQG, LinearQuadraticRegulator


class TestLQGExactness:
    def test_lqg_aliases_and_initialization(self):
        controller = LinearQuadraticGaussian(
            a_matrix=[[0.0, 1.0], [-1.0, -0.5]],
            b_matrix=[[0.0], [1.0]],
            c_matrix=[[0.1], [0.2]],
            q_matrix=[[1.0, 0.0], [0.0, 1.0]],
            r_matrix=[[1.0]],
            t_terminal=2.0,
        )
        assert LQG is LinearQuadraticGaussian
        assert LinearQuadraticRegulator is LinearQuadraticGaussian
        assert controller.dim_state == 2
        assert controller.dim_control == 1

    def test_lqg_validation_raises(self):
        with pytest.raises(ValueError, match="R must be strictly positive definite"):
            LinearQuadraticGaussian(
                a_matrix=[[1.0]],
                b_matrix=[[1.0]],
                c_matrix=[[0.1]],
                q_matrix=[[1.0]],
                r_matrix=[[-1.0]],
            )

        with pytest.raises(ValueError, match="B must have 2 rows matching A"):
            LinearQuadraticGaussian(
                a_matrix=np.eye(2),
                b_matrix=[[1.0]],  # 1 row instead of 2
                c_matrix=np.eye(2),
                q_matrix=np.eye(2),
                r_matrix=[[1.0]],
            )

    def test_lqg_terminal_condition(self):
        # At t = T, Riccati matrix P(T) must equal terminal weight M
        m_mat = np.array([[3.0, 0.5], [0.5, 2.0]])
        controller = LinearQuadraticGaussian(
            a_matrix=np.zeros((2, 2)),
            b_matrix=np.eye(2),
            c_matrix=0.1 * np.eye(2),
            q_matrix=np.eye(2),
            r_matrix=np.eye(2),
            m_terminal=m_mat,
            t_terminal=1.5,
        )
        p_t = controller.riccati_matrix(1.5)
        np.testing.assert_allclose(p_t, m_mat, rtol=1e-5, atol=1e-5)

    def test_lqg_1d_analytical_riccati_solution(self):
        # 1D Integrator: dX = u dt + sigma dW
        # min int_0^T (q x^2 + r u^2) dt + m x_T^2
        # For A=0, B=1, Q=1, R=1, M=1:
        # Differential Riccati eq: dP/dtau = 1 - P^2 with P(0)=1
        # Solution: P(tau) = 1 (constant!) since 1 - 1^2 = 0
        controller = LinearQuadraticGaussian(
            a_matrix=[[0.0]],
            b_matrix=[[1.0]],
            c_matrix=[[0.5]],
            q_matrix=[[1.0]],
            r_matrix=[[1.0]],
            m_terminal=[[1.0]],
            t_terminal=3.0,
        )
        # For all t, P(t) == 1.0!
        for t_test in [0.0, 0.5, 1.5, 2.5, 3.0]:
            p_val = controller.riccati_matrix(t_test)[0, 0]
            assert p_val == pytest.approx(1.0, rel=1e-4)

        # Steady-state ARE should also equal 1.0
        p_are = controller.steady_state_riccati()[0, 0]
        assert p_are == pytest.approx(1.0, rel=1e-7)

    def test_lqg_steady_state_algebraic_riccati(self):
        a = np.array([[-1.0, 0.5], [0.0, -2.0]])
        b = np.array([[1.0], [1.0]])
        q = np.array([[2.0, 0.0], [0.0, 3.0]])
        r = np.array([[0.5]])

        controller = LinearQuadraticGaussian(
            a_matrix=a,
            b_matrix=b,
            c_matrix=0.2 * np.eye(2),
            q_matrix=q,
            r_matrix=r,
            t_terminal=5.0,
        )
        p_are = controller.steady_state_riccati()
        # Verify ARE residual: P A + A^T P - P B R^-1 B^T P + Q == 0
        res = p_are @ a + a.T @ p_are - p_are @ b @ linalg.inv(r) @ b.T @ p_are + q
        np.testing.assert_allclose(res, 0.0, atol=1e-9)

    def test_lqg_simulation_convergence_to_value_function(self):
        # 1D LQG problem
        controller = LinearQuadraticGaussian(
            a_matrix=[[-0.5]],
            b_matrix=[[1.0]],
            c_matrix=[[0.3]],
            q_matrix=[[1.0]],
            r_matrix=[[0.5]],
            m_terminal=[[2.0]],
            t_terminal=1.0,
        )
        x0 = 1.5
        traj = controller.simulate(x0=x0, num_steps=50, num_paths=3000, seed=42)
        v0 = controller.value_function(0.0, x0)

        # Monte Carlo mean realized cost vs analytical value function
        assert traj.mean_cost == pytest.approx(v0, rel=0.06)
