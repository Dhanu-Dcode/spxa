r"""Avellaneda-Stoikov (2008) high-frequency optimal market making model.

Solves the optimal bid-ask quotation and inventory control problem for a market maker
facing order book arrival intensities lambda(delta) = A * exp(-k * delta) and
CARA inventory risk aversion gamma > 0:

    dS_t = sigma * dW_t
    dq_t = dN^b_t - dN^a_t
    dX_t = (S_t + delta^a_t) dN^a_t - (S_t - delta^b_t) dN^b_t

Objective:
    max_{\delta^a, \delta^b} E[ - \exp( - \gamma * (X_T + q_T * S_T) ) ]

Exact Closed-Form Solution:
- Reservation (Indifference) Price:
    R(s, q, t) = s - q * gamma * sigma^2 * (T - t)
- Optimal Half-Spreads:
    delta^a(q, t) = \frac{1}{\gamma} \ln(1 + \frac{\gamma}{k}) - \frac{2q - 1}{2} * gamma * sigma^2 * (T - t)
    delta^b(q, t) = \frac{1}{\gamma} \ln(1 + \frac{\gamma}{k}) + \frac{2q + 1}{2} * gamma * sigma^2 * (T - t)
- Optimal Full Bid-Ask Spread:
    s^*(t) = delta^a(q, t) + delta^b(q, t) = \frac{2}{\gamma} \ln(1 + \frac{\gamma}{k}) + gamma * sigma^2 * (T - t)

References:
    Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book.
        Quantitative Finance, 8(3), 217-224.
    Guéant, O., Tapia, C. A., & Manziadi, Z. (2012). Dealing with the inventory risk:
        a solution to the market making problem. Mathematics and Financial Economics, 6(4), 259-277.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from spxa.control.base import ControlledTrajectory, ControlProblem


class AvellanedaStoikovMarketMaker(ControlProblem):
    r"""Avellaneda-Stoikov high-frequency optimal market maker with inventory risk.

    Parameters
    ----------
    t_terminal : float
        Trading session horizon T > 0 (default: 1.0, e.g. 1 trading day or 1 hour).
    sigma : float
        Asset price volatility sigma > 0 (default: 0.20).
    gamma_risk : float
        Inventory risk aversion coefficient gamma > 0 (default: 0.1).
    intensity_a : float
        Order arrival intensity scale factor A > 0 (default: 140.0).
    intensity_k : float
        Order book liquidity decay parameter k > 0 (default: 1.5).
    max_inventory : int
        Maximum absolute inventory capacity |q| <= Q_max (default: 10).
    s0 : float
        Initial mid-market price (default: 100.0).
    """

    def __init__(
        self,
        t_terminal: float = 1.0,
        sigma: float = 0.20,
        gamma_risk: float = 0.1,
        intensity_a: float = 140.0,
        intensity_k: float = 1.5,
        max_inventory: int = 10,
        s0: float = 100.0,
    ) -> None:
        if sigma <= 0.0:
            raise ValueError(f"Asset volatility sigma must be positive, got {sigma}")
        if gamma_risk <= 0.0:
            raise ValueError(f"Risk aversion gamma must be positive, got {gamma_risk}")
        if intensity_a <= 0.0 or intensity_k <= 0.0:
            raise ValueError(
                f"Intensity parameters A and k must be positive: A={intensity_a}, k={intensity_k}"
            )
        if max_inventory <= 0:
            raise ValueError(
                f"Max inventory capacity must be positive integer, got {max_inventory}"
            )
        if s0 <= 0.0:
            raise ValueError(f"Initial price s0 must be positive, got {s0}")

        super().__init__(t_terminal=t_terminal)

        self.sigma = float(sigma)
        self.gamma_risk = float(gamma_risk)
        self.intensity_a = float(intensity_a)
        self.intensity_k = float(intensity_k)
        self.max_inventory = int(max_inventory)
        self.s0 = float(s0)

        # Baseline spread component: (2 / gamma) * ln(1 + gamma / k)
        self.base_spread = float(
            (2.0 / self.gamma_risk) * math.log(1.0 + self.gamma_risk / self.intensity_k)
        )

    def reservation_price(self, s: float, q: int | float, t: float) -> float:
        r"""Compute the reservation (indifference) price R(s, q, t) = s - q * gamma * sigma^2 * (T - t)."""
        tau = max(0.0, self.t_terminal - t)
        return float(s - q * self.gamma_risk * (self.sigma**2) * tau)

    def optimal_spread(self, t: float) -> float:
        r"""Compute total optimal bid-ask spread s^*(t) = delta^a(q, t) + delta^b(q, t).

        s^*(t) = \frac{2}{\gamma} \ln(1 + \frac{\gamma}{k}) + \gamma * \sigma^2 * (T - t).
        """
        tau = max(0.0, self.t_terminal - t)
        return float(self.base_spread + self.gamma_risk * (self.sigma**2) * tau)

    def optimal_quotes(self, s: float, q: int | float, t: float) -> Tuple[float, float]:
        r"""Compute optimal bid price r^b and ask price r^a.

        Returns
        -------
        tuple of (bid_price, ask_price)
        """
        tau = max(0.0, self.t_terminal - t)
        half_base = 0.5 * self.base_spread
        vol_term = self.gamma_risk * (self.sigma**2) * tau

        delta_a = half_base - 0.5 * (2.0 * q - 1.0) * vol_term
        delta_b = half_base + 0.5 * (2.0 * q + 1.0) * vol_term

        # Check inventory boundary limits
        if q >= self.max_inventory:
            # Reached max positive inventory: refuse to buy more
            bid_price = -float("inf")
        else:
            bid_price = s - delta_b

        if q <= -self.max_inventory:
            # Reached max short inventory: refuse to sell more
            ask_price = float("inf")
        else:
            ask_price = s + delta_a

        return float(bid_price), float(ask_price)

    def optimal_control(self, t: float, x: np.ndarray | float) -> np.ndarray:
        r"""Optimal feedback control [delta^b, delta^a] given current inventory q."""
        tau = max(0.0, self.t_terminal - t)
        half_base = 0.5 * self.base_spread
        vol_term = self.gamma_risk * (self.sigma**2) * tau

        q_val = np.asarray(x, dtype=float)
        delta_a = half_base - 0.5 * (2.0 * q_val - 1.0) * vol_term
        delta_b = half_base + 0.5 * (2.0 * q_val + 1.0) * vol_term

        if q_val.ndim == 0:
            return np.array([float(delta_b), float(delta_a)])
        return np.column_stack((delta_b, delta_a))

    def fill_probabilities(self, delta_b: float, delta_a: float, dt: float) -> Tuple[float, float]:
        r"""Compute order execution fill probabilities over a short interval dt: 1 - exp(-lambda * dt)."""
        lambda_b = self.intensity_a * math.exp(-self.intensity_k * max(0.0, delta_b))
        lambda_a = self.intensity_a * math.exp(-self.intensity_k * max(0.0, delta_a))

        p_b = 1.0 - math.exp(-lambda_b * dt)
        p_a = 1.0 - math.exp(-lambda_a * dt)
        return float(p_b), float(p_a)

    def value_function(self, t: float, x: np.ndarray | float) -> float | np.ndarray:
        r"""Analytical certainty equivalent value function for inventory state q."""
        tau = max(0.0, self.t_terminal - t)
        q = np.asarray(x, dtype=float)
        # Expected PnL approximation minus inventory risk penalty
        risk_penalty = 0.5 * self.gamma_risk * (self.sigma**2) * (q**2) * tau
        spread_capture = (
            self.intensity_a * math.exp(-self.intensity_k * 0.5 * self.base_spread) * tau
        )
        val = spread_capture - risk_penalty
        if q.ndim == 0:
            return float(val)
        return val

    def running_cost(
        self, t: float, x: np.ndarray | float, u: np.ndarray | float
    ) -> float | np.ndarray:
        r"""Running penalty for inventory holding risk: 0.5 * gamma * sigma^2 * q^2."""
        q = np.asarray(x, dtype=float)
        return 0.5 * self.gamma_risk * (self.sigma**2) * (q**2)

    def terminal_cost(self, x: np.ndarray | float) -> float | np.ndarray:
        r"""Terminal inventory liquidation penalty."""
        q = np.asarray(x, dtype=float)
        return 0.5 * self.gamma_risk * (self.sigma**2) * (q**2)

    def drift(self, t: float, x: np.ndarray | float, u: np.ndarray | float) -> float | np.ndarray:
        r"""Expected inventory drift E[dq] = (lambda^b - lambda^a) dt."""
        u_arr = np.asarray(u, dtype=float)
        if u_arr.ndim == 1 and u_arr.shape[0] == 2:
            db, da = u_arr[0], u_arr[1]
            lb = self.intensity_a * math.exp(-self.intensity_k * db)
            la = self.intensity_a * math.exp(-self.intensity_k * da)
            return float(lb - la)
        db = u_arr[:, 0]
        da = u_arr[:, 1]
        lb = self.intensity_a * np.exp(-self.intensity_k * db)
        la = self.intensity_a * np.exp(-self.intensity_k * da)
        return lb - la

    def diffusion(
        self, t: float, x: np.ndarray | float, u: np.ndarray | float
    ) -> float | np.ndarray:
        r"""Inventory variance intensity sqrt(lambda^b + lambda^a)."""
        u_arr = np.asarray(u, dtype=float)
        if u_arr.ndim == 1 and u_arr.shape[0] == 2:
            db, da = u_arr[0], u_arr[1]
            lb = self.intensity_a * math.exp(-self.intensity_k * db)
            la = self.intensity_a * math.exp(-self.intensity_k * da)
            return float(math.sqrt(lb + la))
        db = u_arr[:, 0]
        da = u_arr[:, 1]
        lb = self.intensity_a * np.exp(-self.intensity_k * db)
        la = self.intensity_a * np.exp(-self.intensity_k * da)
        return np.sqrt(lb + la)

    def simulate(
        self,
        x0: int = 0,
        num_steps: int = 200,
        num_paths: int = 1000,
        seed: Optional[int] = None,
    ) -> ControlledTrajectory:
        r"""Simulate limit order book market making, price walks, inventory, and realized PnL."""
        if num_steps <= 0 or num_paths <= 0:
            raise ValueError("num_steps and num_paths must be strictly positive.")

        rng = np.random.default_rng(seed)
        dt = self.t_terminal / num_steps
        sqrt_dt = math.sqrt(dt)
        times = np.linspace(0.0, self.t_terminal, num_steps + 1)

        q_paths = np.empty((num_paths, num_steps + 1), dtype=float)
        s_paths = np.empty((num_paths, num_steps + 1), dtype=float)
        controls = np.empty((num_paths, num_steps, 2), dtype=float)
        cash = np.zeros(num_paths, dtype=float)

        q_paths[:, 0] = float(x0)
        s_paths[:, 0] = self.s0

        for step in range(num_steps):
            t_curr = times[step]
            q_curr = q_paths[:, step]
            s_curr = s_paths[:, step]

            # Mid-price Brownian update
            z = rng.standard_normal(num_paths)
            s_next = s_curr + self.sigma * sqrt_dt * z
            s_paths[:, step + 1] = s_next

            # Optimal quotes
            ctrl = self.optimal_control(t_curr, q_curr)
            delta_b = ctrl[:, 0]
            delta_a = ctrl[:, 1]
            controls[:, step] = ctrl

            # Arrival intensities
            lam_b = self.intensity_a * np.exp(-self.intensity_k * np.maximum(0.0, delta_b))
            lam_a = self.intensity_a * np.exp(-self.intensity_k * np.maximum(0.0, delta_a))

            # Fill probabilities
            prob_b = 1.0 - np.exp(-lam_b * dt)
            prob_a = 1.0 - np.exp(-lam_a * dt)

            # Random fills
            rand_b = rng.random(num_paths)
            rand_a = rng.random(num_paths)

            fill_b = (rand_b < prob_b) & (q_curr < self.max_inventory)
            fill_a = (rand_a < prob_a) & (q_curr > -self.max_inventory)

            # Cash and inventory adjustments
            # Buy fill: cash decreases by bid price = s_curr - delta_b, inventory +1
            # Sell fill: cash increases by ask price = s_curr + delta_a, inventory -1
            cash[fill_b] -= s_curr[fill_b] - delta_b[fill_b]
            cash[fill_a] += s_curr[fill_a] + delta_a[fill_a]

            dq = fill_b.astype(float) - fill_a.astype(float)
            q_paths[:, step + 1] = q_curr + dq

        # Terminal wealth = cash + q_T * S_T
        final_wealth = cash + q_paths[:, -1] * s_paths[:, -1]

        return ControlledTrajectory(
            times=times,
            states=q_paths,
            controls=controls,
            realized_costs=final_wealth,
        )


AvellanedaStoikov = AvellanedaStoikovMarketMaker
OptimalMarketMaking = AvellanedaStoikovMarketMaker
HighFrequencyMarketMaker = AvellanedaStoikovMarketMaker
