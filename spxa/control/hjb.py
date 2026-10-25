r"""1D Finite-Difference Hamilton-Jacobi-Bellman (HJB) Viscosity Solver.

Implements a monotone, upwind finite-difference scheme with policy iteration
for general 1D continuous-time stochastic optimal control problems:

    dx_t = \mu(t, x_t, u_t) dt + \sigma(t, x_t, u_t) dW_t

Value function:
    V(t, x) = inf_{u \in U} E[ \int_t^T e^{-\rho (s - t)} f(s, x_s, u_s) ds + e^{-\rho (T - t)} g(x_T) | x_t = x ]

HJB PDE:
    \partial_t V - \rho V + inf_{u \in U} { \mu(t, x, u) \partial_x V + \frac{1}{2} \sigma(t, x, u)^2 \partial_{xx} V + f(t, x, u) } = 0
    V(T, x) = g(x)

Monotone Discretization (Barles-Souganidis Viscosity Convergence):
- Convective term discretized using upwind differencing:
    \mu >= 0 -> (V_{i+1} - V_i) / dx
    \mu < 0  -> (V_i - V_{i-1}) / dx
- Diffusion term discretized using central second differences:
    \sigma^2 / 2 * (V_{i+1} - 2 V_i + V_{i-1}) / dx^2

References:
    Barles, G. & Souganidis, P. E. (1991). Convergence of approximation schemes
        for fully nonlinear second order equations. Asymptotic Analysis, 4(3), 271-283.
    Kushner, H. J. & Dupuis, P. (2001). Numerical Methods for Stochastic Control
        Problems in Continuous Time. Springer.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np
from scipy import interpolate

from spxa.control.base import ControlProblem


class FiniteDifferenceHJB:
    r"""Monotone finite-difference viscosity solver for 1D HJB equations.

    Parameters
    ----------
    x_min : float
        Lower bound of spatial computational domain.
    x_max : float
        Upper bound of spatial computational domain.
    n_space : int
        Number of spatial grid points N_x >= 10 (default: 101).
    t_terminal : float
        Terminal horizon T > 0 (default: 1.0).
    n_time : int
        Number of time discretization steps N_t >= 10 (default: 100).
    discount_rate : float
        Subjective discount rate rho >= 0 (default: 0.0).
    control_candidates : Sequence[float] or np.ndarray
        Discrete set of candidate controls U to optimize over.
    mode : str
        Optimization objective: 'minimize' (default) or 'maximize'.
    boundary : str
        Boundary condition type: 'neumann' (default, d^2 V / dx^2 = 0) or 'dirichlet'.
    """

    def __init__(
        self,
        x_min: float = -2.0,
        x_max: float = 2.0,
        n_space: int = 101,
        t_terminal: float = 1.0,
        n_time: int = 100,
        discount_rate: float = 0.0,
        control_candidates: Optional[Sequence[float] | np.ndarray] = None,
        mode: str = "minimize",
        boundary: str = "neumann",
    ) -> None:
        if x_max <= x_min:
            raise ValueError(f"x_max ({x_max}) must be strictly greater than x_min ({x_min})")
        if n_space < 10:
            raise ValueError(f"n_space must be at least 10, got {n_space}")
        if t_terminal <= 0.0:
            raise ValueError(f"t_terminal must be positive, got {t_terminal}")
        if n_time < 5:
            raise ValueError(f"n_time must be at least 5, got {n_time}")
        if discount_rate < 0.0:
            raise ValueError(f"discount_rate must be non-negative, got {discount_rate}")
        if mode not in ("minimize", "maximize"):
            raise ValueError(f"mode must be 'minimize' or 'maximize', got {mode}")
        if boundary not in ("neumann", "dirichlet"):
            raise ValueError(f"boundary must be 'neumann' or 'dirichlet', got {boundary}")

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.n_space = int(n_space)
        self.t_terminal = float(t_terminal)
        self.n_time = int(n_time)
        self.rho = float(discount_rate)
        self.mode = mode
        self.boundary = boundary

        self.dx = float((self.x_max - self.x_min) / (self.n_space - 1))
        self.dt = float(self.t_terminal / self.n_time)
        self.x_grid = np.linspace(self.x_min, self.x_max, self.n_space)
        self.t_grid = np.linspace(0.0, self.t_terminal, self.n_time + 1)

        if control_candidates is None:
            # Default control candidates in [-2, 2]
            self.controls = np.linspace(-2.0, 2.0, 41)
        else:
            self.controls = np.asarray(control_candidates, dtype=float)
            if self.controls.size == 0:
                raise ValueError("control_candidates must not be empty.")

        # Result storage: shape (n_time + 1, n_space)
        self.v_grid: Optional[np.ndarray] = None
        self.policy_grid: Optional[np.ndarray] = None
        self._interpolator = None

    def solve_problem(self, problem: ControlProblem) -> "FiniteDifferenceHJB":
        r"""Solve HJB equation using dynamic functions from a ControlProblem instance."""
        drift_fn = lambda t, x, u: problem.drift(t, x, u)  # noqa: E731
        diff_fn = lambda t, x, u: problem.diffusion(t, x, u)  # noqa: E731
        cost_fn = lambda t, x, u: problem.running_cost(t, x, u)  # noqa: E731
        term_fn = lambda x: problem.terminal_cost(x)  # noqa: E731
        return self.solve(drift_fn, diff_fn, cost_fn, term_fn)

    def solve(
        self,
        drift: Callable[[float, float, float], float],
        diffusion: Callable[[float, float, float], float],
        running_cost: Callable[[float, float, float], float],
        terminal_cost: Callable[[float], float],
    ) -> "FiniteDifferenceHJB":
        r"""Solve the HJB PDE backward in time on the space-time grid.

        Parameters
        ----------
        drift : Callable[[t, x, u], float]
            Controlled drift rate mu(t, x, u).
        diffusion : Callable[[t, x, u], float]
            Controlled diffusion rate sigma(t, x, u).
        running_cost : Callable[[t, x, u], float]
            Running cost / utility rate f(t, x, u).
        terminal_cost : Callable[[x], float]
            Terminal cost / bequest utility g(x).
        """
        nx = self.n_space
        nt = self.n_time
        dx = self.dx
        dt = self.dt
        x = self.x_grid
        u_cands = self.controls

        v = np.empty((nt + 1, nx), dtype=float)
        policy = np.empty((nt + 1, nx), dtype=float)

        # Terminal condition at t = T
        for i in range(nx):
            v[nt, i] = float(terminal_cost(x[i]))
            policy[nt, i] = 0.0

        is_min = self.mode == "minimize"

        # Step backward in time from nt - 1 down to 0
        for n in range(nt - 1, -1, -1):
            t_curr = self.t_grid[n]
            v_next = v[n + 1]

            # Compute spatial derivatives with upwind differences
            # Forward difference: (V_{i+1} - V_i) / dx
            d_x_fwd = np.empty(nx, dtype=float)
            d_x_fwd[:-1] = (v_next[1:] - v_next[:-1]) / dx
            d_x_fwd[-1] = d_x_fwd[-2]

            # Backward difference: (V_i - V_{i-1}) / dx
            d_x_bwd = np.empty(nx, dtype=float)
            d_x_bwd[1:] = (v_next[1:] - v_next[:-1]) / dx
            d_x_bwd[0] = d_x_bwd[1]

            # Second difference: (V_{i+1} - 2 V_i + V_{i-1}) / dx^2
            d_xx = np.zeros(nx, dtype=float)
            d_xx[1:-1] = (v_next[2:] - 2.0 * v_next[1:-1] + v_next[:-2]) / (dx**2)
            d_xx[0] = d_xx[1]
            d_xx[-1] = d_xx[-2]

            best_hamiltonian = np.full(nx, float("inf") if is_min else -float("inf"))
            best_u = np.full(nx, u_cands[0])

            for u_val in u_cands:
                try:
                    mu_arr = np.asarray(drift(t_curr, x, u_val), dtype=float)
                    sig_arr = np.asarray(diffusion(t_curr, x, u_val), dtype=float)
                    f_arr = np.asarray(running_cost(t_curr, x, u_val), dtype=float)
                    if mu_arr.shape != (nx,):
                        mu_arr = np.full(nx, float(mu_arr))
                    if sig_arr.shape != (nx,):
                        sig_arr = np.full(nx, float(sig_arr))
                    if f_arr.shape != (nx,):
                        f_arr = np.full(nx, float(f_arr))
                except Exception:
                    mu_arr = np.array([float(drift(t_curr, xi, u_val)) for xi in x])
                    sig_arr = np.array([float(diffusion(t_curr, xi, u_val)) for xi in x])
                    f_arr = np.array([float(running_cost(t_curr, xi, u_val)) for xi in x])

                deriv_x = np.where(mu_arr >= 0.0, d_x_fwd, d_x_bwd)
                ham = mu_arr * deriv_x + 0.5 * (sig_arr**2) * d_xx + f_arr

                if is_min:
                    better = ham < best_hamiltonian
                else:
                    better = ham > best_hamiltonian

                best_hamiltonian[better] = ham[better]
                best_u[better] = u_val

            policy[n, :] = best_u
            v[n, :] = v_next + dt * (best_hamiltonian - self.rho * v_next)

            # Apply boundary conditions
            if self.boundary == "neumann":
                # Linear extrapolation at boundaries
                v[n, 0] = 2.0 * v[n, 1] - v[n, 2]
                v[n, -1] = 2.0 * v[n, -2] - v[n, -3]

        self.v_grid = v
        self.policy_grid = policy
        self._interpolator = interpolate.RectBivariateSpline(self.t_grid, self.x_grid, self.v_grid)
        return self

    def value(self, t: float, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Interpolate the solved value function V(t, x)."""
        if self._interpolator is None:
            raise RuntimeError("HJB solver must be executed via .solve() before evaluating values.")
        t_clamped = float(np.clip(t, 0.0, self.t_terminal))
        x_arr = np.asarray(x, dtype=float)
        x_clamped = np.clip(x_arr, self.x_min, self.x_max)

        if x_arr.ndim == 0:
            return float(self._interpolator(t_clamped, x_clamped)[0, 0])
        val = self._interpolator(t_clamped, x_clamped, grid=False)
        return val

    def optimal_control(self, t: float, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Look up optimal policy u^*(t, x)."""
        if self.policy_grid is None:
            raise RuntimeError("HJB solver must be executed via .solve() before querying policy.")
        t_clamped = float(np.clip(t, 0.0, self.t_terminal))
        t_idx = min(self.n_time, int(round(t_clamped / self.dt)))

        x_arr = np.asarray(x, dtype=float)
        x_clamped = np.clip(x_arr, self.x_min, self.x_max)

        # 1D nearest or linear spatial lookup from policy grid at time t_idx
        idx = np.searchsorted(self.x_grid, x_clamped)
        idx = np.clip(idx, 0, self.n_space - 1)
        res = self.policy_grid[t_idx, idx]
        if x_arr.ndim == 0:
            return float(res)
        return res


HJBSolver = FiniteDifferenceHJB
HamiltonJacobiBellman = FiniteDifferenceHJB
ViscosityHJBSolver = FiniteDifferenceHJB
