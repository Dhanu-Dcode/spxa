"""Exactness tests for Finite-Difference HJB Viscosity Solver."""

import numpy as np
import pytest

from spxa.control.hjb import (
    FiniteDifferenceHJB,
    HamiltonJacobiBellman,
    HJBSolver,
    ViscosityHJBSolver,
)
from spxa.control.lqg import LinearQuadraticGaussian


class TestHJBExactness:
    def test_hjb_aliases_and_initialization(self):
        solver = FiniteDifferenceHJB(
            x_min=-1.0,
            x_max=1.0,
            n_space=51,
            t_terminal=0.5,
            n_time=50,
        )
        assert HJBSolver is FiniteDifferenceHJB
        assert HamiltonJacobiBellman is FiniteDifferenceHJB
        assert ViscosityHJBSolver is FiniteDifferenceHJB
        assert solver.n_space == 51
        assert solver.n_time == 50

    def test_hjb_validation_raises(self):
        with pytest.raises(ValueError, match="x_max .* must be strictly greater"):
            FiniteDifferenceHJB(x_min=2.0, x_max=1.0)
        with pytest.raises(ValueError, match="n_space must be at least 10"):
            FiniteDifferenceHJB(n_space=5)
        with pytest.raises(ValueError, match="t_terminal must be positive"):
            FiniteDifferenceHJB(t_terminal=-1.0)
        with pytest.raises(ValueError, match="mode must be 'minimize' or 'maximize'"):
            FiniteDifferenceHJB(mode="invalid")
        with pytest.raises(ValueError, match="boundary must be 'neumann' or 'dirichlet'"):
            FiniteDifferenceHJB(boundary="robin")

    def test_hjb_not_solved_raises(self):
        solver = FiniteDifferenceHJB()
        with pytest.raises(RuntimeError, match="HJB solver must be executed"):
            solver.value(0.0, 0.0)
        with pytest.raises(RuntimeError, match="HJB solver must be executed"):
            solver.optimal_control(0.0, 0.0)

    def test_hjb_convergence_to_analytical_lqg(self):
        # 1D Deterministic/Low-noise Linear Quadratic Regulator:
        # dx = u dt, min int_0^T (q x^2 + r u^2) dt + m x_T^2
        # where q=1, r=1, m=1, T=0.2.
        # Analytical solution has P(tau) = 1.0 (constant!) and V(t, x) = P(t) x^2 = x^2!
        # Optimal control u*(x) = - R^-1 B^T P x = -x!
        lqg = LinearQuadraticGaussian(
            a_matrix=[[0.0]],
            b_matrix=[[1.0]],
            c_matrix=[[0.01]],
            q_matrix=[[1.0]],
            r_matrix=[[1.0]],
            m_terminal=[[1.0]],
            t_terminal=0.2,
        )

        solver = FiniteDifferenceHJB(
            x_min=-1.0,
            x_max=1.0,
            n_space=101,
            t_terminal=0.2,
            n_time=200,
            control_candidates=np.linspace(-1.5, 1.5, 301),
            mode="minimize",
        )
        solver.solve_problem(lqg)

        assert solver.v_grid.shape == (201, 101)
        assert solver.policy_grid.shape == (201, 101)

        # Check terminal value at t = T matches x^2
        for x_test in [-0.5, 0.0, 0.5]:
            v_term = solver.value(0.2, x_test)
            assert v_term == pytest.approx(x_test**2, abs=1e-3)

        # Check t = 0 value function: V(0, x) should approximate x^2 closely
        for x_test in [-0.4, 0.0, 0.4]:
            v_val = solver.value(0.0, x_test)
            v_analytic = lqg.value_function(0.0, x_test)
            assert v_val == pytest.approx(v_analytic, abs=0.03)

        # Check optimal control at x = 0.5: u* should be close to -0.5
        u_opt = solver.optimal_control(0.1, 0.5)
        assert u_opt == pytest.approx(-0.5, abs=0.05)
