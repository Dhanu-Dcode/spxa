r"""
Two-Factor Hull–White (G2++) short-rate diffusion model.

The G2++ model represents the instantaneous short rate as a sum of two correlated
Gaussian factors plus a deterministic shift function:
    r(t) = x(t) + y(t) + φ(t)

where x(t) and y(t) satisfy the coupled Ornstein–Uhlenbeck SDEs:
    dx(t) = -a x(t) dt + σ dW_1(t),   x(0) = 0
    dy(t) = -b y(t) dt + η dW_2(t),   y(0) = 0

with correlated Brownian motions:
    d⟨W_1, W_2⟩_t = ρ dt,   ρ ∈ [-1, 1]

and deterministic function φ(t) calibrated to the initial term structure.
For a flat initial forward curve at level r_0:
    φ(t) = r_0 + 0.5 (σ/a)^2 (1 - e^{-at})^2 + 0.5 (η/b)^2 (1 - e^{-bt})^2
               + ρ (ση / (ab)) (1 - e^{-at}) (1 - e^{-bt})

Analytical Features
-------------------
1. Exact Gaussian Transition Law:
   The joint factor state (x(t), y(t)) is bivariate normal:
       E[x(t)] = 0,   E[y(t)] = 0
       Var(x(t)) = (σ² / (2a)) (1 - e^{-2at})
       Var(y(t)) = (η² / (2b)) (1 - e^{-2bt})
       Cov(x(t), y(t)) = ρ (ση / (a + b)) (1 - e^{-(a+b)t})

2. Closed-Form Zero-Coupon Bond Pricing:
   The price at time t of a zero-coupon bond maturing at T is:
       P(t, T) = exp(-∫_t^T φ(u) du - B(a, T-t) x(t) - B(b, T-t) y(t) + 0.5 V(t, T))
   where B(k, τ) = (1 - e^{-k τ}) / k, and V(t, T) is the variance of the integrated factors:
       V(t, T) = Var(∫_t^T (x(u) + y(u)) du)

3. Closed-Form European Option on Zero-Coupon Bonds:
   European call and put options on zero-coupon bonds have exact Black-like formulas
   under the T-forward measure (Brigo & Mercurio, 2006).

Exactness Level: MOMENT_PROPAGATION (analytical moments and exact transition distributions).

References
----------
Brigo, D. & Mercurio, F. (2006). *Interest Rate Models — Theory and Practice:
With Smile, Inflation and Credit*, 2nd ed., Springer Finance, Chapter 4.

Hull, J. & White, A. (1994). Numerical procedures for implementing term
structure models II: Two-factor models. *Journal of Derivatives*, 2(2), 37–48.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy.stats import norm

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class G2PlusPlusProcess(Process):
    r"""
    Two-Factor Hull–White (G2++) short-rate diffusion model.

    r(t) = x(t) + y(t) + φ(t)
    dx(t) = -a x(t) dt + σ dW_1(t)
    dy(t) = -b y(t) dt + η dW_2(t)
    d⟨W_1, W_2⟩_t = ρ dt

    Parameters
    ----------
    a : float
        Mean-reversion speed for factor 1, a > 0.
    b : float
        Mean-reversion speed for factor 2, b > 0.
    sigma : float
        Volatility parameter for factor 1, σ > 0.
    eta : float
        Volatility parameter for factor 2, η > 0.
    rho : float
        Correlation between W_1 and W_2, ρ ∈ [-1, 1].
    r0 : float, optional
        Baseline initial short rate level (default 0.03).
    """

    def __init__(
        self,
        a: float = 0.1,
        b: float = 0.05,
        sigma: float = 0.015,
        eta: float = 0.01,
        rho: float = -0.3,
        r0: float = 0.03,
    ) -> None:
        if a <= 0:
            raise ValueError(f"Speed 'a' must be strictly positive, got {a}")
        if b <= 0:
            raise ValueError(f"Speed 'b' must be strictly positive, got {b}")
        if sigma <= 0:
            raise ValueError(f"Volatility 'sigma' must be strictly positive, got {sigma}")
        if eta <= 0:
            raise ValueError(f"Volatility 'eta' must be strictly positive, got {eta}")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"Correlation 'rho' must be in [-1, 1], got {rho}")

        self.a = float(a)
        self.b = float(b)
        self.sigma = float(sigma)
        self.eta = float(eta)
        self.rho = float(rho)
        self.r0 = float(r0)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"G2PlusPlus(a={self.a}, b={self.b}, sigma={self.sigma}, eta={self.eta}, rho={self.rho})",
            math_note=(
                f"r(t) = x(t) + y(t) + φ(t); dx=-{self.a}x dt + {self.sigma}dW1; "
                f"dy=-{self.b}y dt + {self.eta}dW2; corr={self.rho}"
            ),
            reference="Brigo & Mercurio (2006), Interest Rate Models, Section 4.2",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Deterministic Shift & Variance Functions
    # ------------------------------------------------------------------

    def phi(self, t: float) -> float:
        r"""
        Deterministic shift function φ(t) matching a flat initial term structure at r0.

        φ(t) = r_0 + 0.5 (σ/a)^2 (1 - e^{-at})^2 + 0.5 (η/b)^2 (1 - e^{-bt})^2
                   + ρ (ση / (ab)) (1 - e^{-at}) (1 - e^{-bt})
        """
        if t <= 0:
            return self.r0
        term_a = (self.sigma / self.a) * (1.0 - math.exp(-self.a * t))
        term_b = (self.eta / self.b) * (1.0 - math.exp(-self.b * t))
        return self.r0 + 0.5 * term_a**2 + 0.5 * term_b**2 + self.rho * term_a * term_b

    def int_phi(self, t: float, T: float) -> float:
        r"""
        Definite integral ∫_t^T φ(u) du for flat initial curve at r0.
        """
        tau = T - t
        if tau <= 0:
            return 0.0

        # ∫ r0 du = r0 * tau
        res = self.r0 * tau

        # ∫ 0.5 (σ/a)^2 (1 - e^{-au})^2 du
        # (1 - e^{-au})^2 = 1 - 2 e^{-au} + e^{-2au}
        # primitive: u + (2/a) e^{-au} - (1/(2a)) e^{-2au}
        def prim_a(u: float) -> float:
            return (
                u
                + (2.0 / self.a) * math.exp(-self.a * u)
                - (0.5 / self.a) * math.exp(-2.0 * self.a * u)
            )

        def prim_b(u: float) -> float:
            return (
                u
                + (2.0 / self.b) * math.exp(-self.b * u)
                - (0.5 / self.b) * math.exp(-2.0 * self.b * u)
            )

        def prim_cross(u: float) -> float:
            # (1 - e^{-au})(1 - e^{-bu}) = 1 - e^{-au} - e^{-bu} + e^{-(a+b)u}
            return (
                u
                + (1.0 / self.a) * math.exp(-self.a * u)
                + (1.0 / self.b) * math.exp(-self.b * u)
                - (1.0 / (self.a + self.b)) * math.exp(-(self.a + self.b) * u)
            )

        coeff_a = 0.5 * (self.sigma / self.a) ** 2
        coeff_b = 0.5 * (self.eta / self.b) ** 2
        coeff_cross = self.rho * (self.sigma * self.eta) / (self.a * self.b)

        res += coeff_a * (prim_a(T) - prim_a(t))
        res += coeff_b * (prim_b(T) - prim_b(t))
        res += coeff_cross * (prim_cross(T) - prim_cross(t))

        return res

    def V(self, t: float, T: float) -> float:
        r"""
        Variance of the integrated short rate components V(t, T) = Var(∫_t^T (x(u) + y(u)) du).

        Closed-form formula from Brigo & Mercurio (2006, Eq. 4.14).
        """
        tau = T - t
        if tau <= 0:
            return 0.0

        a, b = self.a, self.b
        sigma, eta, rho = self.sigma, self.eta, self.rho

        term_a = (sigma**2 / a**2) * (
            tau + (2.0 / a) * math.exp(-a * tau) - (0.5 / a) * math.exp(-2.0 * a * tau) - 1.5 / a
        )
        term_b = (eta**2 / b**2) * (
            tau + (2.0 / b) * math.exp(-b * tau) - (0.5 / b) * math.exp(-2.0 * b * tau) - 1.5 / b
        )
        term_cross = (2.0 * rho * sigma * eta / (a * b)) * (
            tau
            + (math.exp(-a * tau) - 1.0) / a
            + (math.exp(-b * tau) - 1.0) / b
            - (math.exp(-(a + b) * tau) - 1.0) / (a + b)
        )

        return float(term_a + term_b + term_cross)

    # ------------------------------------------------------------------
    # Factor Distributions and Joint Moments
    # ------------------------------------------------------------------

    def factor_covariance(self, t: float) -> np.ndarray:
        r"""
        Covariance matrix of factors (x(t), y(t)) at time t given x(0)=y(0)=0:
            [[Var(x(t)), Cov(x(t), y(t))],
             [Cov(x(t), y(t)), Var(y(t))]]
        """
        if t <= 0:
            return np.zeros((2, 2))

        var_x = (self.sigma**2 / (2.0 * self.a)) * (1.0 - math.exp(-2.0 * self.a * t))
        var_y = (self.eta**2 / (2.0 * self.b)) * (1.0 - math.exp(-2.0 * self.b * t))
        cov_xy = (
            self.rho
            * (self.sigma * self.eta / (self.a + self.b))
            * (1.0 - math.exp(-(self.a + self.b) * t))
        )
        return np.array([[var_x, cov_xy], [cov_xy, var_y]], dtype=np.float64)

    def conditional_mean(self, t: float, x0: float = 0.0, y0: float = 0.0) -> float:
        r"""
        Expected short rate E[r(t) | x(0)=x0, y(0)=y0].
        """
        ex = x0 * math.exp(-self.a * t)
        ey = y0 * math.exp(-self.b * t)
        return ex + ey + self.phi(t)

    def conditional_variance(self, t: float) -> float:
        r"""
        Variance of the short rate Var(r(t)) = Var(x(t) + y(t)).
        """
        cov = self.factor_covariance(t)
        return float(cov[0, 0] + cov[1, 1] + 2.0 * cov[0, 1])

    # ------------------------------------------------------------------
    # Zero-Coupon Bond & Yield Analytics
    # ------------------------------------------------------------------

    def bond_price(
        self,
        t: float,
        T: float,
        x_t: float = 0.0,
        y_t: float = 0.0,
    ) -> float:
        r"""
        Zero-coupon bond price P(t, T) given factor levels x(t) and y(t).

        P(t, T) = exp( -∫_t^T φ(u)du - B(a, T-t) x_t - B(b, T-t) y_t + 0.5 V(t, T) )
        """
        if T <= t:
            return 1.0

        tau = T - t
        B_a = (1.0 - math.exp(-self.a * tau)) / self.a
        B_b = (1.0 - math.exp(-self.b * tau)) / self.b

        int_phi_val = self.int_phi(t, T)
        v_val = self.V(t, T)

        log_P = -int_phi_val - B_a * x_t - B_b * y_t + 0.5 * v_val
        return float(math.exp(log_P))

    def bond_yield(
        self,
        t: float,
        T: float,
        x_t: float = 0.0,
        y_t: float = 0.0,
    ) -> float:
        r"""
        Continuously compounded zero-coupon yield Y(t, T) = -ln P(t, T) / (T - t).
        """
        if T <= t:
            return self.conditional_mean(t, x_t, y_t)
        P = self.bond_price(t, T, x_t=x_t, y_t=y_t)
        return -math.log(P) / (T - t)

    # ------------------------------------------------------------------
    # European Bond Option Pricing (Brigo & Mercurio)
    # ------------------------------------------------------------------

    def european_call_bond_option(
        self,
        t: float,
        T: float,
        S: float,
        K: float,
        x_t: float = 0.0,
        y_t: float = 0.0,
    ) -> float:
        r"""
        Price at time t of a European call option with expiry T and strike K
        on a zero-coupon bond maturing at S (where t <= T < S).

        Analytical formula from Brigo & Mercurio (2006, Theorem 4.2.1).
        """
        if not (t <= T < S):
            raise ValueError(f"Require t <= T < S, got t={t}, T={T}, S={S}")
        if K <= 0:
            raise ValueError(f"Strike K must be positive, got {K}")

        P_t_T = self.bond_price(t, T, x_t, y_t)
        P_t_S = self.bond_price(t, S, x_t, y_t)

        tau_T = T - t
        tau_opt = S - T

        a, b = self.a, self.b
        sigma, eta, rho = self.sigma, self.eta, self.rho

        # Option volatility Sigma_p(t, T, S)
        sigma_x = (sigma / a) * (1.0 - math.exp(-a * tau_opt))
        sigma_y = (eta / b) * (1.0 - math.exp(-b * tau_opt))

        var_x_T = (1.0 - math.exp(-2.0 * a * tau_T)) / (2.0 * a)
        var_y_T = (1.0 - math.exp(-2.0 * b * tau_T)) / (2.0 * b)
        cov_xy_T = (1.0 - math.exp(-(a + b) * tau_T)) / (a + b)

        sig_sq = (
            sigma_x**2 * var_x_T + sigma_y**2 * var_y_T + 2.0 * rho * sigma_x * sigma_y * cov_xy_T
        )
        sig = math.sqrt(max(sig_sq, 1e-15))

        d1 = (math.log(P_t_S / (P_t_T * K)) + 0.5 * sig**2) / sig
        d2 = d1 - sig

        call_val = P_t_S * norm.cdf(d1) - K * P_t_T * norm.cdf(d2)
        return float(max(call_val, 0.0))

    def european_put_bond_option(
        self,
        t: float,
        T: float,
        S: float,
        K: float,
        x_t: float = 0.0,
        y_t: float = 0.0,
    ) -> float:
        r"""
        Price at time t of a European put option with expiry T and strike K
        on a zero-coupon bond maturing at S via put-call parity.
        """
        call = self.european_call_bond_option(t, T, S, K, x_t, y_t)
        P_t_T = self.bond_price(t, T, x_t, y_t)
        P_t_S = self.bond_price(t, S, x_t, y_t)
        put_val = call - P_t_S + K * P_t_T
        return float(max(put_val, 0.0))

    # ------------------------------------------------------------------
    # Process Overrides & Vectorized Simulation
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "G2PlusPlusProcess is a two-factor non-Lévy diffusion; it does not admit "
            "a stationary Lévy–Khintchine triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=None,
            notes=[
                "Two-factor Gaussian short-rate diffusion (G2++).",
                "Non-Lévy diffusion with state-dependent coupled OU factors.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {1: self.r0, 2: self.conditional_variance(1.0)}
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    def simulate(
        self,
        t_grid: np.ndarray,
        n_paths: int = 1,
        rng: Optional[np.random.Generator] = None,
        return_factors: bool = False,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        r"""
        Exact vectorized simulation of the G2++ short rate paths.

        Parameters
        ----------
        t_grid : np.ndarray
            1-D strictly increasing array of time points starting at 0.
        n_paths : int
            Number of simulated sample paths.
        rng : np.random.Generator, optional
            Numpy random number generator.
        return_factors : bool, optional
            If True, returns tuple (r_paths, x_paths, y_paths).
            If False, returns r_paths of shape (n_paths, len(t_grid)).
        """
        t_grid = np.asarray(t_grid, dtype=np.float64)
        if t_grid.ndim != 1 or len(t_grid) < 2:
            raise ValueError("t_grid must be a 1-D array with at least 2 points")
        if rng is None:
            rng = np.random.default_rng()

        n_steps = len(t_grid)
        x_paths = np.zeros((n_paths, n_steps), dtype=np.float64)
        y_paths = np.zeros((n_paths, n_steps), dtype=np.float64)

        for i in range(1, n_steps):
            dt = t_grid[i] - t_grid[i - 1]
            if dt <= 0:
                raise ValueError("t_grid must be strictly increasing")

            # Exact conditional transition for OU factors
            exp_a = math.exp(-self.a * dt)
            exp_b = math.exp(-self.b * dt)

            var_x = (self.sigma**2 / (2.0 * self.a)) * (1.0 - math.exp(-2.0 * self.a * dt))
            var_y = (self.eta**2 / (2.0 * self.b)) * (1.0 - math.exp(-2.0 * self.b * dt))
            cov_xy = (
                self.rho
                * (self.sigma * self.eta / (self.a + self.b))
                * (1.0 - math.exp(-(self.a + self.b) * dt))
            )

            step_cov = np.array([[var_x, cov_xy], [cov_xy, var_y]])
            step_chol = np.linalg.cholesky(step_cov)

            raw_z = rng.standard_normal(size=(2, n_paths))
            diffs = step_chol @ raw_z

            x_paths[:, i] = x_paths[:, i - 1] * exp_a + diffs[0]
            y_paths[:, i] = y_paths[:, i - 1] * exp_b + diffs[1]

        phi_vals = np.array([self.phi(t) for t in t_grid], dtype=np.float64)
        r_paths = x_paths + y_paths + phi_vals[np.newaxis, :]

        if return_factors:
            return r_paths, x_paths, y_paths
        return r_paths

    def __repr__(self) -> str:
        return (
            f"G2PlusPlusProcess(a={self.a}, b={self.b}, sigma={self.sigma}, "
            f"eta={self.eta}, rho={self.rho}, r0={self.r0})"
        )


# Aliases
G2PlusPlus = G2PlusPlusProcess
TwoFactorHullWhite = G2PlusPlusProcess
