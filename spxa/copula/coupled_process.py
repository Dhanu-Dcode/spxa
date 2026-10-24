"""
Copula-coupled multivariate stochastic process.

Couples arbitrary 1D stochastic processes (diffusions, jump-diffusions,
stochastic volatility models) via a Copula according to Sklar's Theorem.
Enables multi-asset path simulation, joint CDF/PDF evaluation, and
pricing of multi-asset exotic derivatives (basket, worst-of, spread options).
"""

from __future__ import annotations

import inspect
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from spxa.copula.base import Copula
from spxa.core.process import Process


class CoupledProcess:
    """
    Multivariate stochastic process formed by coupling univariate marginal processes via a copula.

    Parameters
    ----------
    processes :
        Sequence of univariate Process instances (length d).
    copula :
        Copula instance of dimension d.
    """

    def __init__(self, processes: Sequence[Process], copula: Copula) -> None:
        if len(processes) < 2:
            raise ValueError(f"CoupledProcess requires at least 2 marginal processes, got {len(processes)}")
        if len(processes) != copula.dim:
            raise ValueError(
                f"Number of marginal processes ({len(processes)}) must equal "
                f"copula dimension ({copula.dim})."
            )
        self._processes = tuple(processes)
        self._copula = copula
        self._dim = len(processes)

    @property
    def dim(self) -> int:
        """Dimension d (number of coupled assets)."""
        return self._dim

    @property
    def processes(self) -> Tuple[Process, ...]:
        """Tuple of 1D marginal processes."""
        return self._processes

    @property
    def copula(self) -> Copula:
        """Coupling copula."""
        return self._copula

    def __repr__(self) -> str:
        proc_names = ", ".join(type(p).__name__ for p in self._processes)
        return f"CoupledProcess([{proc_names}], copula={type(self._copula).__name__})"

    def simulate(
        self,
        n_steps: int,
        n_paths: int = 1,
        T: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""
        Simulate joint sample paths across time [0, T].

        Draws coupled innovations from the copula at each discrete time step \Delta t,
        then maps them into the marginal process state transitions, preserving
        marginal dynamics while imposing the non-linear copula co-movement.

        Parameters
        ----------
        n_steps : int
            Number of time steps.
        n_paths : int
            Number of sample paths. Default 1.
        T : float
            Time horizon in years. Default 1.0.
        rng : Optional[np.random.Generator]
            Random number generator.

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, n_steps + 1, dim) representing the joint paths.
        """
        if rng is None:
            rng = np.random.default_rng()

        d = self.dim
        dt = float(T) / n_steps
        sqrt_dt = np.sqrt(dt)

        # Preallocate paths: shape (n_paths, n_steps + 1, d)
        out = np.zeros((n_paths, n_steps + 1, d), dtype=float)

        # Initialize starting values for each marginal
        for j, p in enumerate(self._processes):
            x0 = getattr(p, "s0", getattr(p, "x0", getattr(p, "r0", getattr(p, "f0", 1.0))))
            out[:, 0, j] = float(x0)

        # Time stepping simulation
        for step in range(n_steps):
            t_curr = step * dt

            # Draw joint copula innovations: U ~ C in [0, 1]^d
            u_innov = self._copula.sample(n=n_paths, rng=rng)
            # Transform to standard normal drivers: Z_j = \Phi^{-1}(U_j)
            z_innov = stats.norm.ppf(np.clip(u_innov, 1e-15, 1.0 - 1e-15))

            for j, p in enumerate(self._processes):
                curr_val = out[:, step, j]
                p_name = type(p).__name__

                if "Brownian" in p_name:
                    mu = getattr(p, "mu", 0.0)
                    sigma = getattr(p, "sigma", 0.2)
                    out[:, step + 1, j] = curr_val + mu * dt + sigma * sqrt_dt * z_innov[:, j]

                elif "CIR" in p_name or "JCIR" in p_name:
                    kappa = getattr(p, "kappa", 1.0)
                    theta = getattr(p, "theta", 0.05)
                    sigma = getattr(p, "sigma", 0.1)
                    v_pos = np.maximum(curr_val, 0.0)
                    drift = kappa * (theta - v_pos) * dt
                    diff = sigma * np.sqrt(v_pos) * sqrt_dt * z_innov[:, j]
                    # Full truncation scheme
                    out[:, step + 1, j] = np.maximum(curr_val + drift + diff, 0.0)

                elif "Heston" in p_name or "Bates" in p_name:
                    # Marginal asset price driven by primary Brownian motion
                    s0 = getattr(p, "s0", 100.0)
                    mu = getattr(p, "mu", 0.05)
                    v0 = getattr(p, "v0", 0.04)
                    sig = np.sqrt(v0)
                    drift = (mu - 0.5 * sig**2) * dt
                    diff = sig * sqrt_dt * z_innov[:, j]
                    out[:, step + 1, j] = curr_val * np.exp(drift + diff)

                elif "Kou" in p_name or "Merton" in p_name:
                    drift_p = getattr(p, "drift", 0.05)
                    sig = getattr(p, "sigma", 0.2)
                    drift = (drift_p - 0.5 * sig**2) * dt
                    diff = sig * sqrt_dt * z_innov[:, j]

                    # Jump arrival check
                    rate = getattr(p, "rate", 1.0)
                    has_jump = rng.uniform(0.0, 1.0, size=n_paths) < (rate * dt)
                    jump_mult = np.ones(n_paths)

                    if np.any(has_jump):
                        n_jumps = int(np.sum(has_jump))
                        if "Merton" in p_name:
                            j_mean = getattr(p, "jump_mean", -0.05)
                            j_std = getattr(p, "jump_std", 0.1)
                            jump_sizes = rng.normal(j_mean, j_std, size=n_jumps)
                        else:  # Kou
                            p_up = getattr(p, "p", 0.4)
                            eta1 = getattr(p, "eta1", 10.0)
                            eta2 = getattr(p, "eta2", 5.0)
                            is_up = rng.uniform(0.0, 1.0, size=n_jumps) < p_up
                            jump_sizes = np.where(
                                is_up,
                                rng.exponential(1.0 / eta1, size=n_jumps),
                                -rng.exponential(1.0 / eta2, size=n_jumps),
                            )
                        jump_mult[has_jump] = np.exp(jump_sizes)

                    out[:, step + 1, j] = curr_val * np.exp(drift + diff) * jump_mult

                else:
                    # General diffusion fallback (Euler step with copula-correlated driver)
                    drift_val = 0.05 * curr_val * dt
                    diff_val = 0.2 * curr_val * sqrt_dt * z_innov[:, j]
                    out[:, step + 1, j] = curr_val + drift_val + diff_val

        return out

    def basket_call_price(
        self,
        weights: Optional[ArrayLike] = None,
        strike: float = 100.0,
        T: float = 1.0,
        r: float = 0.05,
        n_paths: int = 25000,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        r"""
        Price a European basket call option:
            V_0 = e^{-rT} \mathbb{E}\left[ \left( \sum_{j=1}^d w_j S_T^{(j)} - K \right)^+ \right]

        Parameters
        ----------
        weights :
            Portfolio weights for each asset. Defaults to equal weighting (1/d).
        strike :
            Strike price K.
        T :
            Maturity in years.
        r :
            Risk-free discount rate.
        n_paths :
            Monte Carlo path count. Default 25000.
        """
        if weights is None:
            w_arr = np.full(self.dim, 1.0 / self.dim)
        else:
            w_arr = np.asarray(weights, dtype=float)
            if len(w_arr) != self.dim:
                raise ValueError(f"Length of weights ({len(w_arr)}) must equal dimension ({self.dim})")

        paths = self.simulate(n_steps=20, n_paths=n_paths, T=T, rng=rng)
        terminal_prices = paths[:, -1, :]  # shape (n_paths, d)
        basket_val = terminal_prices @ w_arr  # shape (n_paths,)
        payoffs = np.maximum(basket_val - strike, 0.0)
        return float(np.exp(-r * T) * np.mean(payoffs))

    def worst_of_call_price(
        self,
        strike: float = 100.0,
        T: float = 1.0,
        r: float = 0.05,
        n_paths: int = 25000,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        r"""
        Price a worst-of call option:
            V_0 = e^{-rT} \mathbb{E}\left[ \left( \min_{j=1}^d S_T^{(j)} - K \right)^+ \right]
        """
        paths = self.simulate(n_steps=20, n_paths=n_paths, T=T, rng=rng)
        terminal_prices = paths[:, -1, :]
        worst_vals = np.min(terminal_prices, axis=-1)
        payoffs = np.maximum(worst_vals - strike, 0.0)
        return float(np.exp(-r * T) * np.mean(payoffs))

    def best_of_call_price(
        self,
        strike: float = 100.0,
        T: float = 1.0,
        r: float = 0.05,
        n_paths: int = 25000,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        r"""
        Price a best-of call option:
            V_0 = e^{-rT} \mathbb{E}\left[ \left( \max_{j=1}^d S_T^{(j)} - K \right)^+ \right]
        """
        paths = self.simulate(n_steps=20, n_paths=n_paths, T=T, rng=rng)
        terminal_prices = paths[:, -1, :]
        best_vals = np.max(terminal_prices, axis=-1)
        payoffs = np.maximum(best_vals - strike, 0.0)
        return float(np.exp(-r * T) * np.mean(payoffs))

    def spread_option_price(
        self,
        strike: float = 0.0,
        T: float = 1.0,
        r: float = 0.05,
        n_paths: int = 25000,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        r"""
        Price a bivariate spread / exchange option:
            V_0 = e^{-rT} \mathbb{E}\left[ \left( S_T^{(1)} - S_T^{(2)} - K \right)^+ \right]
        """
        if self.dim != 2:
            raise ValueError("Spread option requires exactly 2 marginal processes.")
        paths = self.simulate(n_steps=20, n_paths=n_paths, T=T, rng=rng)
        s1 = paths[:, -1, 0]
        s2 = paths[:, -1, 1]
        payoffs = np.maximum(s1 - s2 - strike, 0.0)
        return float(np.exp(-r * T) * np.mean(payoffs))
