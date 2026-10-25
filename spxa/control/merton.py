r"""Merton's continuous-time optimal consumption and portfolio selection problem.

Solves the classic Merton (1969, 1971) optimal lifetime asset allocation and
consumption problem under Constant Relative Risk Aversion (CRRA) power utility:

    dW_t = [ r * W_t + pi_t * (mu - r) - c_t ] dt + pi_t * sigma * dW_t

Objective:
    V(t, W) = sup_{pi, c} E[ \int_t^T e^{-\rho (s - t)} \frac{c_s^{1-\gamma}}{1-\gamma} ds
                           + \epsilon e^{-\rho (T - t)} \frac{W_T^{1-\gamma}}{1-\gamma} | W_t = W ]

Exact Closed-Form Solution:
- Optimal Risky Asset Weight (Merton Ratio):
    alpha^* = \frac{pi_t^*}{W_t} = \frac{mu - r}{\gamma * sigma^2}   (constant over wealth and time)
- Optimal Consumption Propensity:
    c_t^* = kappa(t) * W_t
    where kappa(t) = \frac{\nu}{1 + (\epsilon^{1/\gamma} \nu - 1) e^{-\nu (T - t)}}
    and \nu = \frac{\rho - (1 - \gamma)(r + \frac{(mu - r)^2}{2 \gamma sigma^2})}{\gamma}
- Value Function:
    V(t, W) = h(t)^\gamma \frac{W^{1-\gamma}}{1-\gamma},   h(t) = \frac{1}{kappa(t)}

References:
    Merton, R. C. (1969). Lifetime portfolio selection under uncertainty: The
        continuous-time case. Review of Economics and Statistics, 51(3), 247-257.
    Merton, R. C. (1971). Optimum consumption and portfolio rules in a
        continuous-time model. Journal of Economic Theory, 3(4), 373-413.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from spxa.control.base import ControlledTrajectory, ControlProblem


class MertonPortfolioControl(ControlProblem):
    r"""Merton's continuous-time optimal consumption and investment model.

    Parameters
    ----------
    mu : float
        Expected rate of return on the risky asset (must satisfy mu > r).
    sigma : float
        Volatility of the risky asset (must be positive).
    r : float
        Risk-free interest rate (default: 0.03).
    gamma : float
        Relative risk aversion coefficient gamma > 0, gamma != 1 (default: 2.0).
    rho : float
        Subjective discount rate rho > 0 (default: 0.05).
    epsilon_bequest : float
        Terminal wealth bequest utility weight epsilon >= 0 (default: 1.0).
    t_terminal : float
        Planning horizon T > 0 (default: 5.0).
    """

    def __init__(
        self,
        mu: float = 0.08,
        sigma: float = 0.20,
        r: float = 0.03,
        gamma: float = 2.0,
        rho: float = 0.05,
        epsilon_bequest: float = 1.0,
        t_terminal: float = 5.0,
    ) -> None:
        if sigma <= 0.0:
            raise ValueError(f"Asset volatility sigma must be positive, got {sigma}")
        if gamma <= 0.0:
            raise ValueError(f"Risk aversion gamma must be positive, got {gamma}")
        if math.isclose(gamma, 1.0, rel_tol=1e-7):
            raise ValueError(
                "Logarithmic utility gamma=1.0 has separate degenerate form; use gamma close to 1."
            )
        if rho <= 0.0:
            raise ValueError(f"Discount rate rho must be positive, got {rho}")
        if epsilon_bequest < 0.0:
            raise ValueError(
                f"Bequest parameter epsilon must be non-negative, got {epsilon_bequest}"
            )

        super().__init__(t_terminal=t_terminal, discount_rate=rho)

        self.mu = float(mu)
        self.sigma = float(sigma)
        self.r = float(r)
        self.gamma = float(gamma)
        self.rho = float(rho)
        self.epsilon = float(epsilon_bequest)

        # Precompute Merton constant ratio alpha^* = (mu - r) / (gamma * sigma^2)
        self.merton_ratio = float((self.mu - self.r) / (self.gamma * (self.sigma**2)))

        # Precompute Merton growth rate and effective discount rate nu
        # Growth factor from optimal investment: r + 0.5 * (mu - r)^2 / (gamma * sigma^2)
        inv_growth = self.r + 0.5 * ((self.mu - self.r) ** 2) / (self.gamma * (self.sigma**2))
        self.nu = float((self.rho - (1.0 - self.gamma) * inv_growth) / self.gamma)

    @property
    def optimal_risky_weight(self) -> float:
        r"""Optimal fraction of wealth invested in the risky asset: \alpha^* = (mu - r) / (gamma * sigma^2)."""
        return self.merton_ratio

    @property
    def infinite_horizon_consumption_propensity(self) -> float:
        r"""Long-run consumption rate propensity as T -> infty: \nu."""
        return self.nu

    def consumption_propensity(self, t: float) -> float:
        r"""Finite-horizon optimal consumption propensity kappa(t) = c^*(t) / W(t).

        kappa(t) = \frac{\nu}{1 + (\epsilon^{1/\gamma} \nu - 1) e^{-\nu (T - t)}}.
        """
        tau = max(0.0, self.t_terminal - t)
        if tau == 0.0:
            if self.epsilon > 0.0:
                return float(1.0 / (self.epsilon ** (1.0 / self.gamma)))
            return float("inf")

        nu = self.nu
        if abs(nu) < 1e-12:
            denom = tau + (self.epsilon ** (1.0 / self.gamma))
            return float(1.0 / denom)

        e_nu_tau = math.exp(-nu * tau)
        if self.epsilon > 0.0:
            factor = (self.epsilon ** (1.0 / self.gamma)) * nu - 1.0
            denom = 1.0 + factor * e_nu_tau
        else:
            denom = 1.0 - e_nu_tau

        if denom <= 0.0:
            return float("inf")
        return float(nu / denom)

    def optimal_control(self, t: float, x: np.ndarray | float) -> np.ndarray:
        r"""Optimal feedback control vector [pi^*_t, c^*_t] given current wealth W_t = x.

        Returns
        -------
        np.ndarray
            If x is scalar: array([pi_star, c_star]).
            If x is array of shape (n_paths,): array of shape (n_paths, 2).
        """
        w = np.asarray(x, dtype=float)
        kap = self.consumption_propensity(t)

        pi_star = self.merton_ratio * w
        c_star = kap * w

        if w.ndim == 0:
            return np.array([float(pi_star), float(c_star)])
        return np.column_stack((pi_star, c_star))

    def value_function(self, t: float, x: np.ndarray | float) -> float | np.ndarray:
        r"""Exact analytical HJB value function V(t, W) = h(t)^\gamma \frac{W^{1-\gamma}}{1-\gamma}."""
        kap = self.consumption_propensity(t)
        h_t = 1.0 / kap if kap > 0.0 else 0.0

        w = np.asarray(x, dtype=float)
        w_safe = np.maximum(1e-12, w)
        val = (h_t**self.gamma) * (w_safe ** (1.0 - self.gamma)) / (1.0 - self.gamma)

        if w.ndim == 0:
            return float(val)
        return val

    def running_cost(
        self, t: float, x: np.ndarray | float, u: np.ndarray | float
    ) -> float | np.ndarray:
        r"""Instantaneous utility rate U(c) = c^{1-\gamma} / (1-\gamma)."""
        u_arr = np.asarray(u, dtype=float)
        if u_arr.ndim == 1 and u_arr.shape[0] == 2:
            c = max(1e-12, float(u_arr[1]))
            return float((c ** (1.0 - self.gamma)) / (1.0 - self.gamma))
        # Batch (n_paths, 2)
        c = np.maximum(1e-12, u_arr[:, 1])
        return (c ** (1.0 - self.gamma)) / (1.0 - self.gamma)

    def terminal_cost(self, x: np.ndarray | float) -> float | np.ndarray:
        r"""Terminal bequest utility g(W) = epsilon * W^{1-\gamma} / (1-\gamma)."""
        w = np.asarray(x, dtype=float)
        w_safe = np.maximum(1e-12, w)
        val = self.epsilon * (w_safe ** (1.0 - self.gamma)) / (1.0 - self.gamma)
        if w.ndim == 0:
            return float(val)
        return val

    def drift(self, t: float, x: np.ndarray | float, u: np.ndarray | float) -> float | np.ndarray:
        r"""Wealth drift: r * W + pi * (mu - r) - c."""
        w = np.asarray(x, dtype=float)
        u_arr = np.asarray(u, dtype=float)
        if u_arr.ndim == 1 and u_arr.shape[0] == 2:
            pi = u_arr[0]
            c = u_arr[1]
            return float(self.r * w + pi * (self.mu - self.r) - c)
        pi = u_arr[:, 0]
        c = u_arr[:, 1]
        return self.r * w + pi * (self.mu - self.r) - c

    def diffusion(
        self, t: float, x: np.ndarray | float, u: np.ndarray | float
    ) -> float | np.ndarray:
        r"""Wealth diffusion: pi * sigma."""
        u_arr = np.asarray(u, dtype=float)
        if u_arr.ndim == 1 and u_arr.shape[0] == 2:
            return float(u_arr[0] * self.sigma)
        return u_arr[:, 0] * self.sigma

    def simulate(
        self,
        x0: float,
        num_steps: int = 100,
        num_paths: int = 1000,
        seed: Optional[int] = None,
    ) -> ControlledTrajectory:
        r"""Simulate controlled wealth and consumption paths under optimal policy."""
        if x0 <= 0.0:
            raise ValueError(f"Initial wealth x0 must be strictly positive, got {x0}")
        if num_steps <= 0 or num_paths <= 0:
            raise ValueError("num_steps and num_paths must be strictly positive.")

        rng = np.random.default_rng(seed)
        dt = self.t_terminal / num_steps
        sqrt_dt = math.sqrt(dt)
        times = np.linspace(0.0, self.t_terminal, num_steps + 1)

        w_paths = np.empty((num_paths, num_steps + 1), dtype=float)
        controls = np.empty((num_paths, num_steps, 2), dtype=float)
        running_utils = np.zeros(num_paths, dtype=float)

        w_paths[:, 0] = float(x0)
        alpha = self.merton_ratio

        for step in range(num_steps):
            t_curr = times[step]
            w_curr = np.maximum(1e-12, w_paths[:, step])

            kap = self.consumption_propensity(t_curr)
            pi_curr = alpha * w_curr
            c_curr = kap * w_curr

            controls[:, step, 0] = pi_curr
            controls[:, step, 1] = c_curr

            # Running discounted utility
            u_c = (c_curr ** (1.0 - self.gamma)) / (1.0 - self.gamma)
            running_utils += u_c * dt * math.exp(-self.rho * t_curr)

            # Log-Euler wealth step to preserve positivity:
            # d ln W = [ r + alpha*(mu - r) - kappa - 0.5 * alpha^2 * sigma^2 ] dt + alpha * sigma * dW
            drift_log = (
                self.r + alpha * (self.mu - self.r) - kap - 0.5 * (alpha**2) * (self.sigma**2)
            )
            vol_log = alpha * self.sigma

            z = rng.standard_normal(num_paths)
            log_w_next = np.log(w_curr) + drift_log * dt + vol_log * sqrt_dt * z
            w_paths[:, step + 1] = np.exp(log_w_next)

        term_util = self.epsilon * (w_paths[:, -1] ** (1.0 - self.gamma)) / (1.0 - self.gamma)
        total_utility = running_utils + term_util * math.exp(-self.rho * self.t_terminal)

        return ControlledTrajectory(
            times=times,
            states=w_paths,
            controls=controls,
            realized_costs=total_utility,
        )


MertonControl = MertonPortfolioControl
MertonPortfolio = MertonPortfolioControl
OptimalConsumptionInvestment = MertonPortfolioControl
