r"""
Schwartz One-Factor (1997) commodity spot and futures diffusion model.

The Schwartz (1997) one-factor model describes the spot price S_t of a commodity
whose logarithm X_t = ln(S_t) follows a mean-reverting Ornstein–Uhlenbeck process:
    d ln S_t = κ(α - ln S_t) dt + σ dW_t,   S_0 > 0

By Itô's lemma, the spot price S_t itself satisfies the geometric SDE:
    dS_t = κ (α + σ² / (2κ) - ln S_t) S_t dt + σ S_t dW_t

Parameters
----------
κ > 0 : Speed of mean reversion
α ∈ ℝ : Long-term equilibrium log-price level
σ > 0 : Volatility parameter
S_0 > 0: Initial commodity spot price
λ     : Market price of risk parameter (default 0.0)

Theoretical Properties
----------------------
1. Exact Log-Normal Transition Law:
   Conditional on S_0, the log-spot price X_t = ln S_t is normally distributed:
       X_t | S_0 ~ N(μ_X(t), σ_X²(t))
   where:
       μ_X(t) = e^{-κt} ln S_0 + (1 - e^{-κt}) α
       σ_X²(t) = (σ² / (2κ)) (1 - e^{-2κt})

   Consequently, the commodity spot price S_t is strictly lognormally distributed:
       E[S_t | S_0] = exp(μ_X(t) + 0.5 σ_X²(t))
       Var(S_t | S_0) = exp(2 μ_X(t) + σ_X²(t)) (exp(σ_X²(t)) - 1)

2. Stationary Distribution:
   As t → ∞, the log-price converges to N(α, σ² / (2κ)), and S_t converges to
   a stationary LogNormal(α, σ² / (2κ)) law with stationary mean:
       E[S_∞] = exp(α + σ² / (4κ))
   and stationary variance:
       Var(S_∞) = exp(2α + σ² / (2κ)) (exp(σ² / (2κ)) - 1)

3. Closed-Form Futures Term Structure:
   Under the risk-neutral measure with risk-adjusted level α* = α - λσ / κ:
       F(S_t, t, T) = E^Q[S_T | S_t]
                    = exp(e^{-κ(T-t)} ln S_t + (1 - e^{-κ(T-t)}) α* + (σ² / (4κ)) (1 - e^{-2κ(T-t)}))

4. Closed-Form European Commodity Options:
   European call and put options on futures contracts possess exact Black-76 formulas
   derived from the log-normal transition structure.

Exactness Level: MOMENT_PROPAGATION (exact log-normal transition and analytical futures).

References
----------
Schwartz, E.S. (1997). The stochastic behavior of commodity prices: Implications
for valuation and hedging. *Journal of Finance*, 52(3), 923–973.

Gibson, R. & Schwartz, E.S. (1990). Stochastic convenience yield and the pricing
of oil contingent claims. *Journal of Finance*, 45(3), 959–976.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy.stats import lognorm, norm

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class SchwartzOneFactor(Process):
    r"""
    Schwartz One-Factor (1997) commodity spot diffusion model.

    d ln S_t = κ(α - ln S_t) dt + σ dW_t,   S_0 > 0.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed κ > 0.
    alpha : float
        Long-term mean log-price α.
    sigma : float
        Volatility parameter σ > 0.
    s0 : float, optional
        Initial spot price S_0 > 0 (defaults to exp(alpha)).
    lambd : float, optional
        Market price of risk λ (default 0.0).
    """

    def __init__(
        self,
        kappa: float = 1.2,
        alpha: float = 3.5,
        sigma: float = 0.35,
        s0: Optional[float] = None,
        lambd: float = 0.0,
    ) -> None:
        if kappa <= 0:
            raise ValueError(f"kappa must be strictly positive, got {kappa}")
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")

        self.kappa = float(kappa)
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.lambd = float(lambd)

        if s0 is None:
            self.s0 = float(math.exp(self.alpha))
        else:
            if s0 <= 0:
                raise ValueError(f"s0 must be strictly positive, got {s0}")
            self.s0 = float(s0)

        # Risk-neutral adjusted alpha: alpha* = alpha - lambda * sigma / kappa
        self.alpha_star = self.alpha - (self.lambd * self.sigma) / self.kappa

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"SchwartzOneFactor(kappa={self.kappa}, alpha={self.alpha}, sigma={self.sigma}, s0={self.s0})",
            math_note=(
                f"d ln S_t = {self.kappa}({self.alpha} - ln S_t)dt + {self.sigma}dW_t; "
                f"S_0={self.s0}, λ={self.lambd}, α*={self.alpha_star:.4f}"
            ),
            reference="Schwartz (1997), Journal of Finance 52(3), 923–973",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Log-Price & Spot Analytical Moments
    # ------------------------------------------------------------------

    def log_conditional_mean(self, t: float, s0: Optional[float] = None) -> float:
        r"""Expected log-spot price E[ln S_t | S_0]."""
        init_s = self.s0 if s0 is None else float(s0)
        exp_k = math.exp(-self.kappa * t)
        return exp_k * math.log(init_s) + (1.0 - exp_k) * self.alpha

    def log_conditional_variance(self, t: float) -> float:
        r"""Variance of the log-spot price Var(ln S_t | S_0)."""
        exp_2k = math.exp(-2.0 * self.kappa * t)
        return (self.sigma**2 / (2.0 * self.kappa)) * (1.0 - exp_2k)

    def conditional_mean(self, t: float, s0: Optional[float] = None) -> float:
        r"""Expected spot price E[S_t | S_0]."""
        mu_x = self.log_conditional_mean(t, s0)
        var_x = self.log_conditional_variance(t)
        return float(math.exp(mu_x + 0.5 * var_x))

    def conditional_variance(self, t: float, s0: Optional[float] = None) -> float:
        r"""Variance of the spot price Var(S_t | S_0)."""
        mu_x = self.log_conditional_mean(t, s0)
        var_x = self.log_conditional_variance(t)
        return float(math.exp(2.0 * mu_x + var_x) * (math.exp(var_x) - 1.0))

    @property
    def stationary_mean(self) -> float:
        r"""Stationary spot mean E[S_∞] = exp(α + σ² / (4κ))."""
        return float(math.exp(self.alpha + (self.sigma**2) / (4.0 * self.kappa)))

    @property
    def stationary_variance(self) -> float:
        r"""Stationary spot variance Var(S_∞)."""
        var_log = (self.sigma**2) / (2.0 * self.kappa)
        return float(math.exp(2.0 * self.alpha + var_log) * (math.exp(var_log) - 1.0))

    def transition_density(
        self,
        s: Union[float, np.ndarray],
        t: float,
        s0: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        r"""Transition probability density function p(S, t | S_0) (log-normal)."""
        if t <= 0:
            raise ValueError("t must be strictly positive")
        mu_x = self.log_conditional_mean(t, s0)
        sigma_x = math.sqrt(self.log_conditional_variance(t))
        s_arr = np.asarray(s, dtype=np.float64)
        pdf = lognorm.pdf(s_arr, s=sigma_x, scale=math.exp(mu_x))
        if np.isscalar(s):
            return float(pdf)
        return pdf

    # ------------------------------------------------------------------
    # Commodity Futures & Options Analytics
    # ------------------------------------------------------------------

    def futures_price(
        self,
        t: float,
        T: float,
        st: Optional[float] = None,
    ) -> float:
        r"""
        Futures contract price F(S_t, t, T) = E^Q[S_T | S_t].

        F(S_t, t, T) = exp( e^{-κ(T-t)} ln S_t + (1 - e^{-κ(T-t)}) α* + (σ² / (4κ)) (1 - e^{-2κ(T-t)}) )
        """
        if T < t:
            raise ValueError(f"Maturity T ({T}) must be >= observation time t ({t})")
        if T == t:
            return self.s0 if st is None else float(st)

        tau = T - t
        cur_s = self.s0 if st is None else float(st)
        exp_k = math.exp(-self.kappa * tau)
        exp_2k = math.exp(-2.0 * self.kappa * tau)

        log_f = (
            exp_k * math.log(cur_s)
            + (1.0 - exp_k) * self.alpha_star
            + (self.sigma**2 / (4.0 * self.kappa)) * (1.0 - exp_2k)
        )
        return float(math.exp(log_f))

    def european_futures_call(
        self,
        t: float,
        T: float,
        T_futures: float,
        K: float,
        r: float = 0.0,
        st: Optional[float] = None,
    ) -> float:
        r"""
        Price at time t of a European call option with expiry T and strike K
        written on a futures contract maturing at T_futures (t <= T <= T_futures).

        Black-76 pricing with exact integrated commodity volatility.
        """
        if not (t <= T <= T_futures):
            raise ValueError(
                f"Require t <= T <= T_futures, got t={t}, T={T}, T_futures={T_futures}"
            )
        if K <= 0:
            raise ValueError(f"Strike K must be positive, got {K}")

        tau_opt = T - t
        if tau_opt == 0.0:
            F = self.futures_price(t, T_futures, st)
            return float(max(F - K, 0.0))

        F = self.futures_price(t, T_futures, st)

        # Integrated variance of the futures price over [t, T]
        # d ln F(u, T_futures) has diffusion coefficient σ e^{-κ(T_futures - u)}
        # ∫_t^T σ² e^{-2κ(T_futures - u)} du = (σ² / (2κ)) e^{-2κ(T_futures - T)} (1 - e^{-2κ(T-t)})
        var_f = (
            (self.sigma**2 / (2.0 * self.kappa))
            * math.exp(-2.0 * self.kappa * (T_futures - T))
            * (1.0 - math.exp(-2.0 * self.kappa * tau_opt))
        )
        sig_f = math.sqrt(max(var_f, 1e-15))

        d1 = (math.log(F / K) + 0.5 * sig_f**2) / sig_f
        d2 = d1 - sig_f
        df = math.exp(-r * tau_opt)

        call_val = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
        return float(max(call_val, 0.0))

    def european_futures_put(
        self,
        t: float,
        T: float,
        T_futures: float,
        K: float,
        r: float = 0.0,
        st: Optional[float] = None,
    ) -> float:
        r"""
        Price at time t of a European put option with expiry T and strike K
        written on a futures contract maturing at T_futures.
        """
        call = self.european_futures_call(t, T, T_futures, K, r, st)
        F = self.futures_price(t, T_futures, st)
        df = math.exp(-r * (T - t))
        put_val = call - df * (F - K)
        return float(max(put_val, 0.0))

    # ------------------------------------------------------------------
    # Process Overrides & Vectorized Simulation
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "SchwartzOneFactor is an exponential Ornstein–Uhlenbeck diffusion; "
            "it does not possess stationary independent increments."
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
                "Schwartz (1997) One-Factor commodity model.",
                "Log-spot price follows an Ornstein–Uhlenbeck process.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        res: dict[int, float] = {
            1: self.conditional_mean(1.0),
            2: self.conditional_variance(1.0),
        }
        for k in range(3, order + 1):
            res[k] = 0.0
        return res

    def simulate(
        self,
        t_grid: np.ndarray,
        n_paths: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""
        Exact vectorized simulation of Schwartz one-factor commodity price paths.
        """
        t_grid = np.asarray(t_grid, dtype=np.float64)
        if t_grid.ndim != 1 or len(t_grid) < 2:
            raise ValueError("t_grid must be a 1-D array with at least 2 points")
        if rng is None:
            rng = np.random.default_rng()

        n_steps = len(t_grid)
        log_paths = np.zeros((n_paths, n_steps), dtype=np.float64)
        log_paths[:, 0] = math.log(self.s0)

        for i in range(1, n_steps):
            dt = t_grid[i] - t_grid[i - 1]
            if dt <= 0:
                raise ValueError("t_grid must be strictly increasing")

            exp_k = math.exp(-self.kappa * dt)
            cond_mean = exp_k * log_paths[:, i - 1] + (1.0 - exp_k) * self.alpha
            cond_std = math.sqrt(
                (self.sigma**2 / (2.0 * self.kappa)) * (1.0 - math.exp(-2.0 * self.kappa * dt))
            )

            z = rng.standard_normal(size=n_paths)
            log_paths[:, i] = cond_mean + cond_std * z

        return np.exp(log_paths)

    def __repr__(self) -> str:
        return (
            f"SchwartzOneFactor(kappa={self.kappa}, alpha={self.alpha}, "
            f"sigma={self.sigma}, s0={self.s0}, lambd={self.lambd})"
        )


# Aliases
ExponentialOU = SchwartzOneFactor
Schwartz1F = SchwartzOneFactor
