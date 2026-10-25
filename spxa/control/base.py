r"""Continuous-time stochastic optimal control base abstractions.

Defines the mathematical interface for stochastic optimal control problems
governed by Ito diffusions and Hamilton-Jacobi-Bellman (HJB) dynamic programming:

    dX_t = mu(t, X_t, u_t) dt + sigma(t, X_t, u_t) dW_t,   X_0 = x_0

Objective:
    V(t, x) = sup_{u in U} E[ \int_t^T f(s, X_s, u_s) ds + g(X_T) | X_t = x ]

HJB Equation:
    -\partial_t V(t, x) = sup_{u in U} { mu(t, x, u) \partial_x V(t, x)
                                        + 0.5 * sigma^2(t, x, u) \partial_{xx} V(t, x)
                                        + f(t, x, u) }
    with terminal condition V(T, x) = g(x).

References:
    Fleming, W. H. & Rishel, R. W. (1975). Deterministic and Stochastic Optimal Control.
        Springer-Verlag.
    Yong, J. & Zhou, X. Y. (1999). Stochastic Controls: Hamiltonian Systems and HJB
        Equations. Springer.
    Pham, H. (2009). Continuous-time Stochastic Control and Optimization with
        Financial Applications. Springer.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ControlledTrajectory:
    """Container for simulated trajectories under a stochastic control policy.

    Attributes
    ----------
    times : np.ndarray
        Array of discrete time points of shape (n_steps + 1,).
    states : np.ndarray
        State paths of shape (n_paths, n_steps + 1) or (n_paths, n_steps + 1, dim).
    controls : np.ndarray
        Applied control actions of shape (n_paths, n_steps) or (n_paths, n_steps, control_dim).
    realized_costs : np.ndarray
        Realized cumulative objective value / payoff for each path of shape (n_paths,).
    """

    times: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    realized_costs: np.ndarray

    @property
    def mean_cost(self) -> float:
        """Empirical mean of realized cost/payoff across all paths."""
        return float(np.mean(self.realized_costs))

    @property
    def std_cost(self) -> float:
        """Empirical standard deviation of realized cost/payoff across all paths."""
        return float(np.std(self.realized_costs))


class ControlProblem(abc.ABC):
    """Abstract base class for continuous-time stochastic control problems.

    Parameters
    ----------
    t_terminal : float
        Terminal time horizon T > 0.
    discount_rate : float
        Subjective discount rate rho >= 0 (default: 0.0).
    """

    def __init__(self, t_terminal: float = 1.0, discount_rate: float = 0.0) -> None:
        if t_terminal <= 0.0:
            raise ValueError(f"Terminal horizon t_terminal must be positive, got {t_terminal}")
        self.t_terminal = float(t_terminal)
        self.discount_rate = discount_rate

    @property
    def discount_rate(self) -> float:
        """Subjective discount rate rho >= 0."""
        return self._discount_rate

    @discount_rate.setter
    def discount_rate(self, val: float) -> None:
        if val < 0.0:
            raise ValueError(f"Discount rate must be non-negative, got {val}")
        self._discount_rate = float(val)

    @abc.abstractmethod
    def running_cost(
        self, t: float, x: np.ndarray | float, u: np.ndarray | float
    ) -> float | np.ndarray:
        """Instantaneous running cost or utility rate f(t, x, u)."""

    @abc.abstractmethod
    def terminal_cost(self, x: np.ndarray | float) -> float | np.ndarray:
        """Terminal cost or bequest value g(x) evaluated at t = T."""

    @abc.abstractmethod
    def drift(self, t: float, x: np.ndarray | float, u: np.ndarray | float) -> float | np.ndarray:
        """Controlled state drift mu(t, x, u)."""

    @abc.abstractmethod
    def diffusion(
        self, t: float, x: np.ndarray | float, u: np.ndarray | float
    ) -> float | np.ndarray:
        """Controlled state diffusion coefficient sigma(t, x, u)."""

    @abc.abstractmethod
    def optimal_control(self, t: float, x: np.ndarray | float) -> float | np.ndarray:
        """Closed-form or numerical optimal feedback control u*(t, x)."""

    @abc.abstractmethod
    def value_function(self, t: float, x: np.ndarray | float) -> float | np.ndarray:
        """Exact or analytical value function V(t, x)."""

    def simulate(
        self,
        x0: float | np.ndarray,
        num_steps: int = 100,
        num_paths: int = 1000,
        seed: Optional[int] = None,
    ) -> ControlledTrajectory:
        """Simulate controlled paths under the optimal feedback policy u*(t, x).

        Parameters
        ----------
        x0 : float or np.ndarray
            Initial state at t = 0.
        num_steps : int
            Number of discrete simulation steps.
        num_paths : int
            Number of simulated sample paths.
        seed : int, optional
            RNG seed.

        Returns
        -------
        ControlledTrajectory
            Simulated states, controls, and realized cumulative payoffs.
        """
        if num_steps <= 0 or num_paths <= 0:
            raise ValueError("num_steps and num_paths must be strictly positive.")

        rng = np.random.default_rng(seed)
        dt = self.t_terminal / num_steps
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0.0, self.t_terminal, num_steps + 1)

        is_scalar = np.isscalar(x0)
        x0_arr = np.asarray(x0, dtype=float)

        if is_scalar:
            states = np.empty((num_paths, num_steps + 1), dtype=float)
            controls = np.empty((num_paths, num_steps), dtype=float)
            states[:, 0] = float(x0_arr)
        else:
            dim = x0_arr.shape[0]
            states = np.empty((num_paths, num_steps + 1, dim), dtype=float)
            controls = np.empty((num_paths, num_steps, dim), dtype=float)
            states[:, 0] = x0_arr

        running_costs = np.zeros(num_paths, dtype=float)

        for step in range(num_steps):
            t_curr = times[step]
            x_curr = states[:, step]

            u_curr = self.optimal_control(t_curr, x_curr)
            if is_scalar:
                controls[:, step] = u_curr
                f_val = self.running_cost(t_curr, x_curr, u_curr)
                running_costs += f_val * dt * np.exp(-self.discount_rate * t_curr)

                mu_val = self.drift(t_curr, x_curr, u_curr)
                sig_val = self.diffusion(t_curr, x_curr, u_curr)
                z = rng.standard_normal(num_paths)
                states[:, step + 1] = x_curr + mu_val * dt + sig_val * sqrt_dt * z
            else:
                controls[:, step] = u_curr
                f_val = self.running_cost(t_curr, x_curr, u_curr)
                running_costs += f_val * dt * np.exp(-self.discount_rate * t_curr)

                mu_val = self.drift(t_curr, x_curr, u_curr)
                sig_val = self.diffusion(t_curr, x_curr, u_curr)
                z = rng.standard_normal((num_paths, dim))
                states[:, step + 1] = x_curr + mu_val * dt + sig_val * sqrt_dt * z

        term_val = self.terminal_cost(states[:, -1])
        total_costs = running_costs + term_val * np.exp(-self.discount_rate * self.t_terminal)

        return ControlledTrajectory(
            times=times,
            states=states,
            controls=controls,
            realized_costs=total_costs,
        )
