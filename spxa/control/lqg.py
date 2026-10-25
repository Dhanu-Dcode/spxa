r"""Linear-Quadratic-Gaussian (LQG) continuous-time stochastic optimal control.

Solves the multi-dimensional continuous-time stochastic linear-quadratic regulator:

    d X_t = (A X_t + B u_t) dt + C dW_t,   X_0 = x_0

Cost functional:
    J(u) = E[ \int_0^T (X_t^T Q X_t + u_t^T R u_t) dt + X_T^T M X_T ]

By the Hamilton-Jacobi-Bellman (HJB) separation principle, the optimal feedback policy is:
    u^*(t, x) = - K(t) x = - R^{-1} B^T P(t) x

where P(t) satisfies the continuous-time backward Matrix Differential Riccati Equation:
    -\dot{P}(t) = P(t) A + A^T P(t) - P(t) B R^{-1} B^T P(t) + Q,   P(T) = M

Value function:
    V(t, x) = x^T P(t) x + s(t)
    where s(t) = \int_t^T \text{Tr}(C C^T P(\tau)) d\tau

References:
    Athans, M. (1971). The role and use of the stochastic Linear-Quadratic-Gaussian
        problem in control theory. IEEE Transactions on Automatic Control, 16(6), 529-552.
    Fleming, W. H. & Rishel, R. W. (1975). Deterministic and Stochastic Optimal Control.
        Springer-Verlag.
    Yong, J. & Zhou, X. Y. (1999). Stochastic Controls: Hamiltonian Systems and HJB
        Equations. Springer.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate, linalg

from spxa.control.base import ControlProblem


class LinearQuadraticGaussian(ControlProblem):
    r"""Linear-Quadratic-Gaussian (LQG) continuous-time stochastic regulator.

    Parameters
    ----------
    a_matrix : ArrayLike
        State drift matrix A of shape (d, d).
    b_matrix : ArrayLike
        Control input matrix B of shape (d, m).
    c_matrix : ArrayLike
        Noise volatility matrix C of shape (d, p).
    q_matrix : ArrayLike
        State running cost weight Q of shape (d, d), symmetric positive semidefinite.
    r_matrix : ArrayLike
        Control running cost weight R of shape (m, m), symmetric positive definite.
    m_terminal : ArrayLike, optional
        Terminal state cost weight M of shape (d, d). If None, defaults to Q.
    t_terminal : float
        Terminal time horizon T > 0 (default: 1.0).
    n_riccati_grid : int
        Number of time interpolation points for Riccati ODE solver (default: 200).
    """

    def __init__(
        self,
        a_matrix: ArrayLike,
        b_matrix: ArrayLike,
        c_matrix: ArrayLike,
        q_matrix: ArrayLike,
        r_matrix: ArrayLike,
        m_terminal: Optional[ArrayLike] = None,
        t_terminal: float = 1.0,
        n_riccati_grid: int = 200,
    ) -> None:
        super().__init__(t_terminal=t_terminal)

        self.A = np.asarray(a_matrix, dtype=float)
        self.B = np.asarray(b_matrix, dtype=float)
        self.C = np.asarray(c_matrix, dtype=float)
        self.Q = np.asarray(q_matrix, dtype=float)
        self.R = np.asarray(r_matrix, dtype=float)

        if self.A.ndim == 1:
            self.A = self.A.reshape(1, 1)
        if self.B.ndim == 1:
            self.B = self.B.reshape(-1, 1)
        if self.C.ndim == 1:
            self.C = self.C.reshape(-1, 1)
        if self.Q.ndim == 1:
            self.Q = self.Q.reshape(1, 1)
        if self.R.ndim == 1:
            self.R = self.R.reshape(1, 1)

        d = self.A.shape[0]
        m = self.B.shape[1]

        if self.A.shape != (d, d):
            raise ValueError(f"A must be square of shape ({d}, {d}), got {self.A.shape}")
        if self.B.shape[0] != d:
            raise ValueError(f"B must have {d} rows matching A, got {self.B.shape}")
        if self.C.shape[0] != d:
            raise ValueError(f"C must have {d} rows matching A, got {self.C.shape}")
        if self.Q.shape != (d, d):
            raise ValueError(f"Q must have shape ({d}, {d}), got {self.Q.shape}")
        if self.R.shape != (m, m):
            raise ValueError(f"R must have shape ({m}, {m}), got {self.R.shape}")

        # Check positive definiteness of R
        eig_r = np.linalg.eigvalsh(self.R)
        if np.any(eig_r <= 0):
            raise ValueError(f"R must be strictly positive definite, got eigenvalues {eig_r}")

        if m_terminal is None:
            self.M = self.Q.copy()
        else:
            self.M = np.asarray(m_terminal, dtype=float)
            if self.M.ndim == 1:
                self.M = self.M.reshape(1, 1)
            if self.M.shape != (d, d):
                raise ValueError(f"M must have shape ({d}, {d}), got {self.M.shape}")

        self.dim_state = d
        self.dim_control = m
        self.dim_noise = self.C.shape[1]
        self.n_riccati_grid = max(50, int(n_riccati_grid))

        # Solve continuous backward matrix Riccati ODE
        self._solve_differential_riccati()

    def _solve_differential_riccati(self) -> None:
        r"""Integrate backward Riccati differential equation from t = T to t = 0:

        - dP/dt = P A + A^T P - P B R^{-1} B^T P + Q,   P(T) = M
        """
        d = self.dim_state
        r_inv = linalg.inv(self.R)
        b_r_inv_bt = self.B @ r_inv @ self.B.T
        c_ct = self.C @ self.C.T

        t_grid = np.linspace(0.0, self.t_terminal, self.n_riccati_grid)
        self._t_grid = t_grid

        # Vectorized system for symmetric matrix P(t) of size d*(d+1)/2
        def riccati_rhs(tau: float, p_flat: np.ndarray) -> np.ndarray:
            # tau = T - t (forward in time from tau = 0 to T)
            p_mat = p_flat.reshape((d, d))
            # dP/dtau = P A + A^T P - P B R^-1 B^T P + Q
            dp_dtau = p_mat @ self.A + self.A.T @ p_mat - p_mat @ b_r_inv_bt @ p_mat + self.Q
            # Enforce symmetry
            dp_dtau = 0.5 * (dp_dtau + dp_dtau.T)
            return dp_dtau.ravel()

        sol = integrate.solve_ivp(
            riccati_rhs,
            t_span=(0.0, self.t_terminal),
            y0=self.M.ravel(),
            t_eval=self.t_terminal - t_grid[::-1],  # evaluated at increasing tau
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )

        # Reverse back so that index 0 corresponds to t = 0 and -1 to t = T
        p_tau_rev = sol.y[:, ::-1]
        self._p_trajectory = np.empty((self.n_riccati_grid, d, d), dtype=float)
        self._gain_trajectory = np.empty((self.n_riccati_grid, self.dim_control, d), dtype=float)

        for i in range(self.n_riccati_grid):
            p_mat = p_tau_rev[:, i].reshape((d, d))
            p_mat = 0.5 * (p_mat + p_mat.T)
            self._p_trajectory[i] = p_mat
            # Gain K(t) = R^{-1} B^T P(t)
            self._gain_trajectory[i] = r_inv @ self.B.T @ p_mat

        # Compute scalar trace term s(t) = int_t^T Tr(C C^T P(tau)) dtau
        traces = np.array(
            [np.trace(c_ct @ self._p_trajectory[i]) for i in range(self.n_riccati_grid)]
        )
        # Backward cumulative integral
        s_vals = np.zeros(self.n_riccati_grid, dtype=float)
        dt_riccati = self.t_terminal / (self.n_riccati_grid - 1)
        for i in range(self.n_riccati_grid - 2, -1, -1):
            s_vals[i] = s_vals[i + 1] + 0.5 * (traces[i] + traces[i + 1]) * dt_riccati
        self._s_trajectory = s_vals

    def riccati_matrix(self, t: float) -> np.ndarray:
        r"""Return Riccati matrix P(t) interpolated at time t."""
        t_clamped = np.clip(t, 0.0, self.t_terminal)
        idx = np.searchsorted(self._t_grid, t_clamped)
        if idx == 0:
            return self._p_trajectory[0]
        if idx >= self.n_riccati_grid:
            return self._p_trajectory[-1]
        # Linear interpolation
        t0, t1 = self._t_grid[idx - 1], self._t_grid[idx]
        w = (t_clamped - t0) / (t1 - t0)
        return (1.0 - w) * self._p_trajectory[idx - 1] + w * self._p_trajectory[idx]

    def feedback_gain(self, t: float) -> np.ndarray:
        r"""Optimal feedback gain matrix K(t) = R^{-1} B^T P(t)."""
        p_mat = self.riccati_matrix(t)
        return linalg.inv(self.R) @ self.B.T @ p_mat

    def steady_state_riccati(self) -> np.ndarray:
        r"""Solve infinite-horizon continuous Algebraic Riccati Equation (ARE):

        P A + A^T P - P B R^{-1} B^T P + Q = 0.
        """
        return linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)

    def steady_state_gain(self) -> np.ndarray:
        r"""Steady-state feedback gain matrix K_infty = R^{-1} B^T P_infty."""
        p_inf = self.steady_state_riccati()
        return linalg.inv(self.R) @ self.B.T @ p_inf

    def optimal_control(self, t: float, x: np.ndarray | float) -> np.ndarray:
        r"""Optimal feedback control u^*(t, x) = - K(t) x."""
        k_mat = self.feedback_gain(t)
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            if x_arr.shape[0] == self.dim_state:
                # Single state vector of shape (d,)
                return -k_mat @ x_arr
            # Array of 1D scalar states of shape (n_paths,)
            return -k_mat[0, 0] * x_arr
        # Array of batch states (n_paths, d)
        return -(x_arr @ k_mat.T)

    def value_function(self, t: float, x: np.ndarray | float) -> float | np.ndarray:
        r"""Analytical value function V(t, x) = x^T P(t) x + s(t)."""
        p_mat = self.riccati_matrix(t)
        t_clamped = np.clip(t, 0.0, self.t_terminal)
        idx = min(
            self.n_riccati_grid - 1,
            int(round(t_clamped / self.t_terminal * (self.n_riccati_grid - 1))),
        )
        s_val = self._s_trajectory[idx]

        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
            quad = p_mat[0, 0] * (float(x) ** 2)
            return float(quad + s_val)

        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1 and x_arr.shape[0] == self.dim_state:
            quad = float(x_arr.T @ p_mat @ x_arr)
            return float(quad + s_val)
        if x_arr.ndim == 1:
            # 1D batch array (n_paths,)
            quad = p_mat[0, 0] * (x_arr**2)
            return quad + s_val
        # Batch states (n_paths, d)
        quad = np.sum((x_arr @ p_mat) * x_arr, axis=-1)
        return quad + s_val

    def running_cost(
        self, t: float, x: np.ndarray | float, u: np.ndarray | float
    ) -> float | np.ndarray:
        r"""Quadratic running cost f(x, u) = x^T Q x + u^T R u."""
        if (np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)) and (
            np.isscalar(u) or (isinstance(u, np.ndarray) and u.ndim == 0)
        ):
            return float(self.Q[0, 0] * (float(x) ** 2) + self.R[0, 0] * (float(u) ** 2))

        x_arr = np.asarray(x, dtype=float)
        u_arr = np.asarray(u, dtype=float)

        if x_arr.ndim == 1 and x_arr.shape[0] == self.dim_state:
            cost_x = float(x_arr.T @ self.Q @ x_arr)
            cost_u = float(u_arr.T @ self.R @ u_arr)
            return float(cost_x + cost_u)
        if x_arr.ndim == 1:
            return self.Q[0, 0] * (x_arr**2) + self.R[0, 0] * (u_arr**2)

        cost_x = np.sum((x_arr @ self.Q) * x_arr, axis=-1)
        cost_u = np.sum((u_arr @ self.R) * u_arr, axis=-1)
        return cost_x + cost_u

    def terminal_cost(self, x: np.ndarray | float) -> float | np.ndarray:
        r"""Terminal cost g(x) = x^T M x."""
        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
            return float(self.M[0, 0] * (float(x) ** 2))

        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1 and x_arr.shape[0] == self.dim_state:
            return float(x_arr.T @ self.M @ x_arr)
        if x_arr.ndim == 1:
            return self.M[0, 0] * (x_arr**2)
        return np.sum((x_arr @ self.M) * x_arr, axis=-1)

    def drift(self, t: float, x: np.ndarray | float, u: np.ndarray | float) -> float | np.ndarray:
        r"""Controlled drift mu(x, u) = A x + B u."""
        if (np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)) and (
            np.isscalar(u) or (isinstance(u, np.ndarray) and u.ndim == 0)
        ):
            return float(self.A[0, 0] * float(x) + self.B[0, 0] * float(u))

        x_arr = np.asarray(x, dtype=float)
        u_arr = np.asarray(u, dtype=float)
        if x_arr.ndim == 1 and x_arr.shape[0] == self.dim_state:
            return self.A @ x_arr + self.B @ u_arr
        if x_arr.ndim == 1:
            return self.A[0, 0] * x_arr + self.B[0, 0] * u_arr
        return x_arr @ self.A.T + u_arr @ self.B.T
        if x_arr.ndim == 1:
            return self.A[0, 0] * x_arr + self.B[0, 0] * u_arr
        return x_arr @ self.A.T + u_arr @ self.B.T

    def diffusion(
        self, t: float, x: np.ndarray | float, u: np.ndarray | float
    ) -> float | np.ndarray:
        r"""Diffusion matrix C."""
        if self.dim_state == 1 and self.dim_noise == 1:
            return float(self.C[0, 0])
        return self.C


LQG = LinearQuadraticGaussian
LinearQuadraticRegulator = LinearQuadraticGaussian
