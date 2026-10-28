r"""Almgren-Chriss (2000) optimal portfolio execution and liquidation problem.

Solves the continuous-time optimal liquidation problem for a block of Q_0 shares
over a finite execution horizon [0, T] subject to permanent and temporary market impact
under a mean-variance risk-aversion criterion:

    d q_t = v_t dt,   q_0 = Q_0,   q_T = 0
    d S_t = gamma * v_t dt + sigma * dW_t
    \tilde{S}_t = S_t + eta * v_t

Objective:
    min_{v_t} E[Total Execution Cost] + lambda * Var[Total Execution Cost]

Exact Closed-Form Solution:
- Urgency Parameter:
    kappa = \sqrt{\frac{\lambda * sigma^2}{\eta}}
- Optimal Inventory Trajectory:
    q^*(t) = Q_0 * \frac{\sinh(\kappa * (T - t))}{\sinh(\kappa * T)}
- Optimal Trading Speed:
    v^*(t) = - Q_0 * \kappa * \frac{\cosh(\kappa * (T - t))}{\sinh(\kappa * T)}
- Expected Shortfall:
    E[S] = 0.5 * gamma * Q_0^2 + 0.5 * eta * Q_0^2 * \kappa * \frac{\cosh(\kappa T)\sinh(\kappa T) + \kappa T}{\sinh^2(\kappa T)}
- Variance of Shortfall:
    Var[S] = sigma^2 * Q_0^2 * \frac{\sinh(2\kappa T) - 2\kappa T}{4 * \kappa * \sinh^2(\kappa T)}

References:
    Almgren, R. & Chriss, N. (2000). Optimal execution of portfolio transactions.
        Journal of Risk, 3(2), 5-39.
    Bertsimas, D. & Lo, A. W. (1998). Optimal control of execution costs.
        Journal of Financial Markets, 1(1), 1-50.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from spxa.control.base import ControlledTrajectory, ControlProblem


class AlmgrenChrissExecution(ControlProblem):
    r"""Almgren-Chriss continuous-time optimal order execution model.

    Parameters
    ----------
    total_shares : float
        Initial position Q_0 to be liquidated (can be positive for selling or negative for buying).
    t_terminal : float
        Execution window length T > 0 (default: 1.0).
    sigma : float
        Price volatility sigma > 0 (default: 0.30).
    eta : float
        Temporary price impact coefficient eta > 0 (default: 2.5e-4).
    gamma_impact : float
        Permanent price impact coefficient gamma >= 0 (default: 2.5e-5).
    risk_aversion : float
        Risk aversion parameter lambda >= 0 (default: 1e-3). If 0, recovers TWAP.
    s0 : float
        Initial asset spot price (default: 100.0).
    """

    def __init__(
        self,
        total_shares: float = 10000.0,
        t_terminal: float = 1.0,
        sigma: float = 0.30,
        eta: float = 2.5e-4,
        gamma_impact: float = 2.5e-5,
        risk_aversion: float = 1e-3,
        s0: float = 100.0,
    ) -> None:
        if total_shares == 0.0:
            raise ValueError("Total shares to liquidate must be non-zero.")
        if sigma <= 0.0:
            raise ValueError(f"Price volatility sigma must be positive, got {sigma}")
        if eta <= 0.0:
            raise ValueError(f"Temporary impact parameter eta must be positive, got {eta}")
        if gamma_impact < 0.0:
            raise ValueError(
                f"Permanent impact parameter gamma must be non-negative, got {gamma_impact}"
            )
        if risk_aversion < 0.0:
            raise ValueError(f"Risk aversion lambda must be non-negative, got {risk_aversion}")
        if s0 <= 0.0:
            raise ValueError(f"Initial price s0 must be positive, got {s0}")

        super().__init__(t_terminal=t_terminal)

        self.q0 = float(total_shares)
        self.sigma = float(sigma)
        self.eta = float(eta)
        self.gamma_impact = float(gamma_impact)
        self.risk_aversion = float(risk_aversion)
        self.s0 = float(s0)

        # Urgency parameter kappa = sqrt(lambda * sigma^2 / eta)
        if self.risk_aversion > 0.0:
            self.kappa = float(math.sqrt(self.risk_aversion * (self.sigma**2) / self.eta))
        else:
            self.kappa = 0.0

    @property
    def urgency_parameter(self) -> float:
        r"""Execution urgency parameter \kappa = \sqrt{\lambda * \sigma^2 / \eta}."""
        return self.kappa

    @property
    def is_twap(self) -> bool:
        r"""Check if model is in the risk-neutral TWAP limit (\lambda == 0)."""
        return self.risk_aversion == 0.0

    def optimal_inventory(self, t: float) -> float:
        r"""Analytical optimal inventory q^*(t) at time t in [0, T].

        q^*(t) = Q_0 * \frac{\sinh(\kappa * (T - t))}{\sinh(\kappa * T)}.
        """
        if t <= 0.0:
            return self.q0
        if t >= self.t_terminal:
            return 0.0

        tau = self.t_terminal - t
        if self.is_twap or self.kappa < 1e-7:
            # Linear TWAP inventory: Q_0 * (1 - t/T)
            return float(self.q0 * (tau / self.t_terminal))

        k_tau = self.kappa * tau
        k_t = self.kappa * self.t_terminal
        # Guard against overflow in sinh
        if k_t > 80.0:
            return float(self.q0 * math.exp(-self.kappa * t))

        return float(self.q0 * math.sinh(k_tau) / math.sinh(k_t))

    def optimal_trading_speed(self, t: float) -> float:
        r"""Analytical optimal trading velocity v^*(t) = dq^*(t)/dt at time t.

        v^*(t) = - Q_0 * \kappa * \frac{\cosh(\kappa * (T - t))}{\sinh(\kappa * T)}.
        """
        if t >= self.t_terminal:
            return 0.0

        tau = self.t_terminal - t
        if self.is_twap or self.kappa < 1e-7:
            # Constant TWAP velocity: -Q_0 / T
            return float(-self.q0 / self.t_terminal)

        k_tau = self.kappa * tau
        k_t = self.kappa * self.t_terminal
        if k_t > 80.0:
            return float(-self.q0 * self.kappa * math.exp(-self.kappa * t))

        return float(-self.q0 * self.kappa * math.cosh(k_tau) / math.sinh(k_t))

    def optimal_control(self, t: float, x: np.ndarray | float) -> float | np.ndarray:
        r"""Optimal feedback control v^*(t, q) given current inventory q.

        v^*(t, q) = - q * \kappa * \coth(\kappa * (T - t)).
        """
        tau = max(1e-6, self.t_terminal - t)
        if self.is_twap or self.kappa < 1e-7:
            coeff = -1.0 / tau
        else:
            k_tau = self.kappa * tau
            if k_tau > 80.0:
                coeff = -self.kappa
            else:
                coeff = -self.kappa * (math.cosh(k_tau) / math.sinh(k_tau))

        q_val = np.asarray(x, dtype=float)
        return coeff * q_val

    def expected_shortfall(self) -> float:
        r"""Analytical expected execution shortfall E[S] = E[Q_0 * S_0 - Total Proceeds].

        E[S] = 0.5 * gamma * Q_0^2 + 0.5 * eta * Q_0^2 * \kappa * \frac{\cosh(\kappa T)\sinh(\kappa T) + \kappa T}{\sinh^2(\kappa T)}.
        """
        q0_sq = self.q0**2
        perm_cost = 0.5 * self.gamma_impact * q0_sq

        if self.is_twap or self.kappa < 1e-7:
            temp_cost = self.eta * q0_sq / self.t_terminal
            return float(perm_cost + temp_cost)

        k_t = self.kappa * self.t_terminal
        if k_t > 80.0:
            temp_cost = 0.5 * self.eta * q0_sq * self.kappa
            return float(perm_cost + temp_cost)

        sinh_kt = math.sinh(k_t)
        cosh_kt = math.cosh(k_t)
        factor = (cosh_kt * sinh_kt + k_t) / (sinh_kt**2)
        temp_cost = 0.5 * self.eta * q0_sq * self.kappa * factor
        return float(perm_cost + temp_cost)

    def variance_shortfall(self) -> float:
        r"""Analytical variance of execution shortfall Var[S] = sigma^2 * \int_0^T q^*(t)^2 dt.

        Var[S] = sigma^2 * Q_0^2 * \frac{\sinh(2\kappa T) - 2\kappa T}{4 * \kappa * \sinh^2(\kappa T)}.
        """
        q0_sq = self.q0**2
        sig2 = self.sigma**2

        if self.is_twap or self.kappa < 1e-7:
            # TWAP variance: sigma^2 * Q_0^2 * T / 3
            return float(sig2 * q0_sq * self.t_terminal / 3.0)

        k_t = self.kappa * self.t_terminal
        if k_t > 40.0:
            return float(sig2 * q0_sq / (2.0 * self.kappa))

        sinh_kt = math.sinh(k_t)
        sinh_2kt = math.sinh(2.0 * k_t)
        factor = (sinh_2kt - 2.0 * k_t) / (4.0 * self.kappa * (sinh_kt**2))
        return float(sig2 * q0_sq * factor)

    def efficient_frontier(self, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute the efficient execution frontier (std_shortfall, expected_shortfall).

        Varies risk aversion lambda across a geometric range from 1e-6 to 1e-1.
        """
        lambdas = np.geomspace(1e-6, 1e-1, n_points)
        stds = np.empty(n_points)
        means = np.empty(n_points)

        orig_lambda = self.risk_aversion
        orig_kappa = self.kappa

        for i, lam in enumerate(lambdas):
            self.risk_aversion = lam
            self.kappa = math.sqrt(lam * (self.sigma**2) / self.eta)
            means[i] = self.expected_shortfall()
            stds[i] = math.sqrt(self.variance_shortfall())

        # Restore original state
        self.risk_aversion = orig_lambda
        self.kappa = orig_kappa

        return stds, means

    def value_function(self, t: float, x: np.ndarray | float) -> float | np.ndarray:
        r"""Value function: expected remaining cost + lambda * remaining variance."""
        tau = max(1e-6, self.t_terminal - t)
        q = np.asarray(x, dtype=float)

        if self.is_twap or self.kappa < 1e-7:
            temp_cost = self.eta * (q**2) / tau
            var_cost = (self.sigma**2) * (q**2) * tau / 3.0
            return temp_cost + self.risk_aversion * var_cost

        k_tau = self.kappa * tau
        if k_tau > 80.0:
            cost = 0.5 * self.eta * (q**2) * self.kappa
            var = (self.sigma**2) * (q**2) / (2.0 * self.kappa)
            return cost + self.risk_aversion * var

        sinh_kt = math.sinh(k_tau)
        cosh_kt = math.cosh(k_tau)
        factor = (cosh_kt * sinh_kt + k_tau) / (sinh_kt**2)
        temp_cost = 0.5 * self.eta * (q**2) * self.kappa * factor
        var_factor = (math.sinh(2.0 * k_tau) - 2.0 * k_tau) / (4.0 * self.kappa * (sinh_kt**2))
        var_cost = (self.sigma**2) * (q**2) * var_factor

        return temp_cost + self.risk_aversion * var_cost

    def running_cost(
        self, t: float, x: np.ndarray | float, u: np.ndarray | float
    ) -> float | np.ndarray:
        r"""Running penalty rate: eta * v^2 + lambda * sigma^2 * q^2."""
        q = np.asarray(x, dtype=float)
        v = np.asarray(u, dtype=float)
        return self.eta * (v**2) + self.risk_aversion * (self.sigma**2) * (q**2)

    def terminal_cost(self, x: np.ndarray | float) -> float | np.ndarray:
        r"""Terminal penalty for unexecuted shares (infinite or large quadratic penalty)."""
        q = np.asarray(x, dtype=float)
        return 1e6 * (q**2)

    def drift(self, t: float, x: np.ndarray | float, u: np.ndarray | float) -> float | np.ndarray:
        r"""Inventory drift dq = v dt."""
        return u

    def diffusion(
        self, t: float, x: np.ndarray | float, u: np.ndarray | float
    ) -> float | np.ndarray:
        r"""Inventory diffusion (deterministic inventory dq)."""
        return 0.0

    def simulate(
        self,
        x0: Optional[float] = None,
        num_steps: int = 100,
        num_paths: int = 1000,
        seed: Optional[int] = None,
    ) -> ControlledTrajectory:
        r"""Simulate liquidation execution paths, price evolution, and realized shortfall."""
        if num_steps <= 0 or num_paths <= 0:
            raise ValueError("num_steps and num_paths must be strictly positive.")

        init_q = self.q0 if x0 is None else float(x0)
        rng = np.random.default_rng(seed)
        dt = self.t_terminal / num_steps
        sqrt_dt = math.sqrt(dt)
        times = np.linspace(0.0, self.t_terminal, num_steps + 1)

        q_paths = np.empty((num_paths, num_steps + 1), dtype=float)
        v_paths = np.empty((num_paths, num_steps), dtype=float)
        s_paths = np.empty((num_paths, num_steps + 1), dtype=float)

        q_paths[:, 0] = init_q
        s_paths[:, 0] = self.s0

        # Cash generated
        cash = np.zeros(num_paths, dtype=float)

        for step in range(num_steps):
            t_curr = times[step]
            q_curr = q_paths[:, step]
            s_curr = s_paths[:, step]

            # Optimal trading speed
            v_curr = self.optimal_control(t_curr, q_curr)
            v_paths[:, step] = v_curr

            # Execution price: S_t + eta * v_t
            s_exec = s_curr + self.eta * v_curr
            # Cash from selling (-v_curr * dt shares)
            cash += -v_curr * dt * s_exec

            # Update inventory
            q_paths[:, step + 1] = q_curr + v_curr * dt

            # Spot price update: permanent impact + Brownian noise
            z = rng.standard_normal(num_paths)
            ds = self.gamma_impact * v_curr * dt + self.sigma * sqrt_dt * z
            s_paths[:, step + 1] = s_curr + ds

        # Execution shortfall = Initial Book Value (Q_0 * S_0) - Realized Cash
        shortfalls = init_q * self.s0 - cash

        return ControlledTrajectory(
            times=times,
            states=q_paths,
            controls=v_paths,
            realized_costs=shortfalls,
        )


AlmgrenChriss = AlmgrenChrissExecution
AlmgrenChrissModel = AlmgrenChrissExecution
OptimalExecution = AlmgrenChrissExecution
