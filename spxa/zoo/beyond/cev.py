r"""
Constant Elasticity of Variance (CEV) diffusion process.

The CEV process satisfies the stochastic differential equation (SDE):
    dS_t = μ S_t dt + σ S_t^γ dW_t,   S_0 = s_0 > 0

where:
  μ ∈ ℝ : Drift rate (under risk-neutral pricing, μ = r - q)
  σ > 0 : Volatility scale parameter
  γ ≥ 0 : Elasticity exponent (power on asset price)
  s_0 > 0 : Initial asset price
  W_t   : Standard 1-dimensional Brownian motion

Local Volatility & Elasticity Structure
---------------------------------------
The return (percentage) volatility is state-dependent:
    σ_loc(S) = σ S^{γ - 1}

The elasticity of return variance with respect to spot price is constant:
    (d(σ_loc(S)²) / σ_loc(S)²) / (dS / S) = 2(γ - 1)

Classification of Regimes
-------------------------
1. Equity leverage regime (0 ≤ γ < 1):
   - Local volatility increases as asset price falls (leverage effect).
   - The origin S = 0 is an absorbing boundary (bankruptcy / default).
   - Once absorbed at zero, the process remains at zero: S_{t+u} = 0 for u ≥ 0.
   - Closed-form European option prices are given by Cox (1975) and Schroder (1989)
     in terms of the non-central chi-squared distribution.

2. Geometric Brownian Motion regime (γ = 1):
   - Local volatility is constant: σ_loc(S) = σ.
   - Reduces to the classical Black–Scholes–Merton model.
   - Zero is strictly inaccessible.

3. Commodity / inverse leverage regime (γ > 1):
   - Local volatility increases as asset price rises (common in agricultural / energy commodities).
   - Zero is strictly inaccessible; closed-form option prices follow Emanuel & MacBeth (1982).

Exactness Level
---------------
ExactnessLevel.MOMENT_PROPAGATION (state-dependent local volatility σ S^γ precludes
a Lévy–Khintchine triplet, but analytical non-central chi-squared option pricing,
absorption probabilities, transition laws, and exact simulation are available).

References
----------
Cox, J.C. (1975). Notes on option pricing I: Constant elasticity of diffusions.
    *Working paper*, Stanford University.
Schroder, M. (1989). Computing the constant elasticity of variance option pricing
    formula. *The Journal of Finance*, 44(1), 211–219.
Emanuel, D.C. & MacBeth, J.D. (1982). Further results of the constant elasticity
    of variance call option pricing model. *Journal of Financial and Quantitative Analysis*,
    17(4), 533–554.
Linetsky, V. & Mendoza, R. (2010). Constant elasticity of variance (CEV) diffusion model.
    In *Encyclopedia of Quantitative Finance*, R. Cont (Ed.), John Wiley & Sons.
Choi, J. & Wu, L. (2021). The equivalent constant-elasticity-of-variance (CEV) volatility
    of the stochastic-alpha-beta-rho (SABR) model. *Journal of Economic Dynamics and Control*,
    128, 104143.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.sim.exact import sample_cev
from spxa.story.narrator import CompositionNode, NodeKind


class CEVProcess(Process):
    r"""
    Constant Elasticity of Variance (CEV) diffusion process.

    dS_t = μ S_t dt + σ S_t^γ dW_t,   S_0 = s_0 > 0.

    Parameters
    ----------
    mu : float, default 0.0
        Drift parameter μ. Under risk-neutral pricing, μ = r - q.
    sigma : float, default 0.2
        Volatility scale parameter σ > 0.
    gamma : float, default 0.5
        Elasticity power exponent γ ≥ 0.
        - γ < 1: Equity leverage regime (absorbing zero, implied volatility skew).
        - γ = 1: Geometric Brownian Motion (constant volatility σ).
        - γ > 1: Commodity / inverse leverage regime.
    s0 : float, default 100.0
        Initial asset spot price s_0 > 0.
    r : float, optional
        Risk-free interest rate r. If specified, sets the default discount rate and
        aligns drift μ = r - q.
    q : float, default 0.0
        Continuous dividend yield q ≥ 0.

    Attributes
    ----------
    elasticity : float
        Elasticity of return variance 2(γ - 1).
    vol_elasticity : float
        Elasticity of return volatility γ - 1.
    is_absorbed_at_zero : bool
        True if the origin is an absorbing boundary (γ < 1).
    regime : str
        Regime name: 'equity_leverage' (γ < 1), 'geometric_brownian' (γ = 1),
        or 'commodity' (γ > 1).

    References
    ----------
    Cox (1975); Schroder (1989), *Journal of Finance* 44(1), 211–219.
    """

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 0.2,
        gamma: float = 0.5,
        s0: float = 100.0,
        r: Optional[float] = None,
        q: float = 0.0,
    ) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if s0 <= 0:
            raise ValueError(f"s0 must be strictly positive, got {s0}")
        if q < 0:
            raise ValueError(f"q must be non-negative, got {q}")

        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.s0 = float(s0)
        self.q = float(q)

        if r is not None:
            self.r = float(r)
            self.mu = float(r - q)
        else:
            self.mu = float(mu)
            self.r = float(mu + q)

        regime_label = (
            "equity leverage (absorbing 0)"
            if self.gamma < 1.0
            else ("GBM (constant vol)" if abs(self.gamma - 1.0) < 1e-7 else "commodity (inverse leverage)")
        )

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"CEVProcess(mu={self.mu}, sigma={self.sigma}, gamma={self.gamma}, s0={self.s0})",
            math_note=(
                f"dS_t = μ S_t dt + σ S_t^γ dW_t; μ={self.mu}, σ={self.sigma}, "
                f"γ={self.gamma}, s0={self.s0}; regime: {regime_label}; "
                f"variance elasticity 2(γ-1) = {2.0 * (self.gamma - 1.0):.2f}"
            ),
            reference="Cox (1975); Schroder (1989), Journal of Finance 44(1), 211–219",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def __repr__(self) -> str:
        return f"CEVProcess(mu={self.mu}, sigma={self.sigma}, gamma={self.gamma}, s0={self.s0})"

    # ------------------------------------------------------------------
    # Core structural properties
    # ------------------------------------------------------------------

    @property
    def elasticity(self) -> float:
        r"""
        Elasticity of return variance with respect to asset price:
        \frac{d(\sigma(S)^2) / \sigma(S)^2}{dS / S} = 2(\gamma - 1).
        """
        return 2.0 * (self.gamma - 1.0)

    @property
    def vol_elasticity(self) -> float:
        r"""
        Elasticity of return volatility with respect to asset price:
        \frac{d\sigma(S) / \sigma(S)}{dS / S} = \gamma - 1.
        """
        return self.gamma - 1.0

    @property
    def is_absorbed_at_zero(self) -> bool:
        r"""True if boundary at zero is absorbing (γ < 1)."""
        return self.gamma < 1.0

    @property
    def feller_condition(self) -> bool:
        r"""
        True if the origin is strictly inaccessible (γ ≥ 1).
        For γ < 1, the origin is an attainable absorbing boundary.
        """
        return self.gamma >= 1.0

    @property
    def regime(self) -> str:
        r"""
        Regime name:
        - 'equity_leverage' (γ < 1)
        - 'geometric_brownian' (γ = 1)
        - 'commodity' (γ > 1)
        """
        if abs(self.gamma - 1.0) < 1e-7:
            return "geometric_brownian"
        elif self.gamma < 1.0:
            return "equity_leverage"
        else:
            return "commodity"

    # ------------------------------------------------------------------
    # Local Volatility & Default / Absorption Probability
    # ------------------------------------------------------------------

    def local_volatility(self, S: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Local percentage return volatility:
            σ_loc(S) = σ S^{γ - 1}.

        Parameters
        ----------
        S : float or np.ndarray
            Asset price level S ≥ 0.

        Returns
        -------
        float or np.ndarray
            Local volatility value(s).
        """
        s_arr = np.asarray(S, dtype=float)
        if np.any(s_arr < 0.0):
            raise ValueError("Asset price S must be non-negative")

        if abs(self.gamma - 1.0) < 1e-7:
            vol = np.full_like(s_arr, self.sigma)
        elif self.gamma < 1.0:
            with np.errstate(divide="ignore"):
                vol = np.where(s_arr == 0.0, np.inf, self.sigma * (s_arr ** (self.gamma - 1.0)))
        else:
            vol = self.sigma * (s_arr ** (self.gamma - 1.0))

        return float(vol) if np.isscalar(S) else vol

    def absorption_probability(
        self,
        T: float,
        r: Optional[float] = None,
        q: Optional[float] = None,
    ) -> float:
        r"""
        Analytical absorption (default / bankruptcy) probability at zero:
            P(S_T = 0 | S_0 = s_0).

        For γ ≥ 1, the origin is strictly inaccessible, so P(S_T = 0) = 0.
        For γ < 1, the origin is an absorbing boundary with exact probability:
            P(S_T = 0) = 1 - F_{χ²}(x; 1 / (1 - γ))
        where F_{χ²} is the central chi-squared distribution function with
        1 / (1 - γ) degrees of freedom.

        Parameters
        ----------
        T : float
            Time horizon T ≥ 0.
        r : float, optional
            Risk-free rate. Defaults to `self.r`.
        q : float, optional
            Dividend yield. Defaults to `self.q`.

        Returns
        -------
        float
            Absorption probability ∈ [0, 1].
        """
        if T < 0:
            raise ValueError(f"Time T must be non-negative, got {T}")
        if T == 0:
            return 0.0
        if self.gamma >= 1.0:
            return 0.0

        val_r = self.r if r is None else float(r)
        val_q = self.q if q is None else float(q)
        nu = val_r - val_q
        beta_star = 1.0 - self.gamma

        if abs(nu) < 1e-12:
            k_const = 1.0 / ((beta_star**2) * (self.sigma**2) * T)
            x = k_const * (self.s0 ** (2.0 * beta_star))
        else:
            denom = (self.sigma**2) * beta_star * (math.exp(2.0 * nu * beta_star * T) - 1.0)
            k_const = 2.0 * nu / denom
            x = k_const * (self.s0 ** (2.0 * beta_star)) * math.exp(2.0 * nu * beta_star * T)

        df = 1.0 / beta_star
        prob = float(stats.chi2.sf(x, df=df))
        return min(max(prob, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Analytical Option Pricing (Schroder 1989 / Cox 1975)
    # ------------------------------------------------------------------

    def call_price(
        self,
        K: float,
        T: float,
        r: Optional[float] = None,
        q: Optional[float] = None,
    ) -> float:
        r"""
        Exact European call option price under the CEV model.

        Evaluated via Schroder's (1989) representation in terms of the
        non-central chi-squared distribution:
        - When γ < 1:
            C = S_0 e^{-q T} [1 - F_{χ²}(y; 2 + 1/β_*, x)] - K e^{-r T} F_{χ²}(x; 1/β_*, y)
        - When γ = 1:
            Standard Black–Scholes–Merton call price.
        - When γ > 1:
            Emanuel & MacBeth (1982) commodity formula.

        Parameters
        ----------
        K : float
            Strike price K > 0.
        T : float
            Time to maturity T ≥ 0.
        r : float, optional
            Risk-free rate. Defaults to `self.r`.
        q : float, optional
            Dividend yield. Defaults to `self.q`.

        Returns
        -------
        float
            Call option price.
        """
        if K <= 0:
            raise ValueError(f"Strike K must be strictly positive, got {K}")
        if T < 0:
            raise ValueError(f"Maturity T must be non-negative, got {T}")
        if T == 0:
            return max(self.s0 - K, 0.0)

        val_r = self.r if r is None else float(r)
        val_q = self.q if q is None else float(q)
        nu = val_r - val_q

        # Case 1: Geometric Brownian Motion (gamma == 1)
        if abs(self.gamma - 1.0) < 1e-7:
            d1 = (math.log(self.s0 / K) + (nu + 0.5 * self.sigma**2) * T) / (self.sigma * math.sqrt(T))
            d2 = d1 - self.sigma * math.sqrt(T)
            call = self.s0 * math.exp(-val_q * T) * stats.norm.cdf(d1) - K * math.exp(-val_r * T) * stats.norm.cdf(d2)
            return float(max(call, 0.0))

        # Case 2: Equity leverage regime (gamma < 1) - Schroder (1989)
        if self.gamma < 1.0:
            beta_star = 1.0 - self.gamma
            if abs(nu) < 1e-12:
                k_const = 1.0 / ((beta_star**2) * (self.sigma**2) * T)
                x = k_const * (self.s0 ** (2.0 * beta_star))
                y = k_const * (K ** (2.0 * beta_star))
            else:
                denom = (self.sigma**2) * beta_star * (math.exp(2.0 * nu * beta_star * T) - 1.0)
                k_const = 2.0 * nu / denom
                x = k_const * (self.s0 ** (2.0 * beta_star)) * math.exp(2.0 * nu * beta_star * T)
                y = k_const * (K ** (2.0 * beta_star))

            df1 = 2.0 + 1.0 / beta_star
            df2 = 1.0 / beta_star

            term1 = self.s0 * math.exp(-val_q * T) * stats.ncx2.sf(y, df=df1, nc=x)
            term2 = K * math.exp(-val_r * T) * stats.ncx2.cdf(x, df=df2, nc=y)
            return float(max(term1 - term2, 0.0))

        # Case 3: Commodity regime (gamma > 1) - Emanuel & MacBeth (1982)
        diff = self.gamma - 1.0
        if abs(nu) < 1e-12:
            omega = (self.sigma**2) * T
        else:
            omega = ((self.sigma**2) / (2.0 * nu * diff)) * (math.exp(2.0 * nu * diff * T) - 1.0)

        x = (self.s0 ** (-2.0 * diff)) / ((diff**2) * omega)
        y = ((K * math.exp(-nu * T)) ** (-2.0 * diff)) / ((diff**2) * omega)
        df1 = 1.0 / diff
        df2 = 2.0 + 1.0 / diff

        term1 = self.s0 * math.exp(-val_q * T) * stats.ncx2.sf(x, df=df1, nc=y)
        term2 = K * math.exp(-val_r * T) * stats.ncx2.cdf(y, df=df2, nc=x)
        return float(max(term1 - term2, 0.0))

    def put_price(
        self,
        K: float,
        T: float,
        r: Optional[float] = None,
        q: Optional[float] = None,
    ) -> float:
        r"""
        Exact European put option price under the CEV model.

        Satisfies put–call parity exactly:
            P(K, T) = C(K, T) - S_0 e^{-q T} + K e^{-r T}.
        """
        if K <= 0:
            raise ValueError(f"Strike K must be strictly positive, got {K}")
        if T < 0:
            raise ValueError(f"Maturity T must be non-negative, got {T}")
        if T == 0:
            return max(K - self.s0, 0.0)

        val_r = self.r if r is None else float(r)
        val_q = self.q if q is None else float(q)
        nu = val_r - val_q

        if abs(self.gamma - 1.0) < 1e-7:
            d1 = (math.log(self.s0 / K) + (nu + 0.5 * self.sigma**2) * T) / (self.sigma * math.sqrt(T))
            d2 = d1 - self.sigma * math.sqrt(T)
            put = K * math.exp(-val_r * T) * stats.norm.cdf(-d2) - self.s0 * math.exp(-val_q * T) * stats.norm.cdf(-d1)
            return float(max(put, 0.0))

        if self.gamma < 1.0:
            beta_star = 1.0 - self.gamma
            if abs(nu) < 1e-12:
                k_const = 1.0 / ((beta_star**2) * (self.sigma**2) * T)
                x = k_const * (self.s0 ** (2.0 * beta_star))
                y = k_const * (K ** (2.0 * beta_star))
            else:
                denom = (self.sigma**2) * beta_star * (math.exp(2.0 * nu * beta_star * T) - 1.0)
                k_const = 2.0 * nu / denom
                x = k_const * (self.s0 ** (2.0 * beta_star)) * math.exp(2.0 * nu * beta_star * T)
                y = k_const * (K ** (2.0 * beta_star))

            df1 = 2.0 + 1.0 / beta_star
            df2 = 1.0 / beta_star

            term1 = K * math.exp(-val_r * T) * stats.ncx2.sf(x, df=df2, nc=y)
            term2 = self.s0 * math.exp(-val_q * T) * stats.ncx2.cdf(y, df=df1, nc=x)
            return float(max(term1 - term2, 0.0))

        # For gamma > 1: evaluate via put-call parity
        call = self.call_price(K, T, r=val_r, q=val_q)
        put = call - self.s0 * math.exp(-val_q * T) + K * math.exp(-val_r * T)
        return float(max(put, 0.0))

    def price(
        self,
        K: float,
        T: float,
        option_type: str = "call",
        r: Optional[float] = None,
        q: Optional[float] = None,
    ) -> float:
        r"""Unified European option pricer for 'call' or 'put'."""
        if option_type == "call":
            return self.call_price(K, T, r=r, q=q)
        elif option_type == "put":
            return self.put_price(K, T, r=r, q=q)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    # ------------------------------------------------------------------
    # Greeks (Delta, Gamma, Vega)
    # ------------------------------------------------------------------

    def delta(
        self,
        K: float,
        T: float,
        option_type: str = "call",
        r: Optional[float] = None,
        q: Optional[float] = None,
        eps: float = 1e-4,
    ) -> float:
        r"""
        Delta Greek ∂V / ∂S_0 via high-precision central differencing.
        Satisfies call–put delta parity: Δ_call - Δ_put = e^{-q T}.
        """
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
        h = self.s0 * eps
        orig_s0 = self.s0

        try:
            self.s0 = orig_s0 + h
            price_up = self.call_price(K, T, r=r, q=q) if option_type == "call" else self.put_price(K, T, r=r, q=q)

            self.s0 = max(orig_s0 - h, 1e-6)
            h_eff = (orig_s0 + h) - self.s0
            price_down = self.call_price(K, T, r=r, q=q) if option_type == "call" else self.put_price(K, T, r=r, q=q)
        finally:
            self.s0 = orig_s0

        return float((price_up - price_down) / h_eff)

    def gamma_greek(
        self,
        K: float,
        T: float,
        r: Optional[float] = None,
        q: Optional[float] = None,
        eps: float = 1e-4,
    ) -> float:
        r"""
        Gamma Greek ∂²V / ∂S_0² via high-precision central differencing.
        Identical for European calls and puts.
        """
        h = self.s0 * eps
        orig_s0 = self.s0
        try:
            self.s0 = orig_s0
            price_mid = self.call_price(K, T, r=r, q=q)
            self.s0 = orig_s0 + h
            price_up = self.call_price(K, T, r=r, q=q)
            self.s0 = max(orig_s0 - h, 1e-6)
            price_down = self.call_price(K, T, r=r, q=q)
        finally:
            self.s0 = orig_s0

        return float((price_up - 2.0 * price_mid + price_down) / (h**2))

    def vega(
        self,
        K: float,
        T: float,
        r: Optional[float] = None,
        q: Optional[float] = None,
        eps: float = 1e-4,
    ) -> float:
        r"""
        Vega Greek ∂V / ∂σ via high-precision central differencing.
        Identical for European calls and puts.
        """
        h = self.sigma * eps
        orig_sigma = self.sigma
        try:
            self.sigma = orig_sigma + h
            price_up = self.call_price(K, T, r=r, q=q)
            self.sigma = max(orig_sigma - h, 1e-6)
            h_eff = (orig_sigma + h) - self.sigma
            price_down = self.call_price(K, T, r=r, q=q)
        finally:
            self.sigma = orig_sigma

        return float((price_up - price_down) / h_eff)

    # ------------------------------------------------------------------
    # Transition CDF & Conditional Moments
    # ------------------------------------------------------------------

    def transition_cdf(
        self,
        k: float,
        T: float,
        r: Optional[float] = None,
        q: Optional[float] = None,
    ) -> float:
        r"""
        Conditional cumulative distribution function P(S_T ≤ k | S_0 = s_0).
        """
        if T < 0:
            raise ValueError(f"Maturity T must be non-negative, got {T}")
        if k < 0:
            return 0.0
        if k == 0:
            return self.absorption_probability(T, r=r, q=q)
        if T == 0:
            return 1.0 if self.s0 <= k else 0.0

        val_r = self.r if r is None else float(r)
        val_q = self.q if q is None else float(q)
        nu = val_r - val_q

        if abs(self.gamma - 1.0) < 1e-7:
            mu_t = (nu - 0.5 * self.sigma**2) * T
            sig_t = self.sigma * math.sqrt(T)
            z = (math.log(k / self.s0) - mu_t) / sig_t
            return float(stats.norm.cdf(z))

        if self.gamma < 1.0:
            beta_star = 1.0 - self.gamma
            if abs(nu) < 1e-12:
                k_const = 1.0 / ((beta_star**2) * (self.sigma**2) * T)
                x = k_const * (self.s0 ** (2.0 * beta_star))
                y = k_const * (k ** (2.0 * beta_star))
            else:
                denom = (self.sigma**2) * beta_star * (math.exp(2.0 * nu * beta_star * T) - 1.0)
                k_const = 2.0 * nu / denom
                x = k_const * (self.s0 ** (2.0 * beta_star)) * math.exp(2.0 * nu * beta_star * T)
                y = k_const * (k ** (2.0 * beta_star))

            df = 1.0 / beta_star
            cdf_val = float(stats.ncx2.sf(x, df=df, nc=y))
            return min(max(cdf_val, 0.0), 1.0)

        # For gamma > 1: numerical derivative of call price w.r.t strike K
        h = max(k * 1e-4, 1e-5)
        c_plus = self.call_price(k + h, T, r=val_r, q=val_q)
        c_minus = self.call_price(max(k - h, 1e-6), T, r=val_r, q=val_q)
        dC_dK = (c_plus - c_minus) / (2.0 * h)
        prob = 1.0 + math.exp(val_r * T) * dC_dK
        return min(max(float(prob), 0.0), 1.0)

    def conditional_mean(self, t: float, s0: Optional[float] = None) -> float:
        r"""
        Conditional expectation E[S_t | S_0 = s_0] = s_0 e^{μ t}.

        Parameters
        ----------
        t : float
            Time horizon t ≥ 0.
        s0 : float, optional
            Initial state s_0 ≥ 0. Defaults to `self.s0`.
        """
        if t < 0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        val_s0 = self.s0 if s0 is None else float(s0)
        if val_s0 < 0:
            raise ValueError(f"s0 must be non-negative, got {val_s0}")
        return float(val_s0 * math.exp(self.mu * t))

    def conditional_variance(self, t: float, s0: Optional[float] = None) -> float:
        r"""
        Conditional variance Var(S_t | S_0 = s_0).

        - For γ = 1 (GBM):
            Var(S_t) = s_0² e^{2μ t} (e^{σ² t} - 1).
        - For γ = 0.5 (Square-Root / CIR):
            Var(S_t) = (σ² s_0 / μ) e^{μ t} (e^{μ t} - 1).
        - For general γ:
            Evaluated via numerical integration of the tail probability:
            E[S_t²] = ∫₀^∞ 2 s (1 - F(s)) ds.
        """
        if t < 0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        if t == 0:
            return 0.0
        val_s0 = self.s0 if s0 is None else float(s0)
        if val_s0 <= 0:
            return 0.0

        # Exact formula for gamma == 1.0 (GBM)
        if abs(self.gamma - 1.0) < 1e-7:
            mean = val_s0 * math.exp(self.mu * t)
            return float((mean**2) * (math.exp(self.sigma**2 * t) - 1.0))

        # Exact formula for gamma == 0.5 (CIR / Square-root)
        if abs(self.gamma - 0.5) < 1e-7:
            if abs(self.mu) < 1e-12:
                return float(val_s0 * (self.sigma**2) * t)
            else:
                return float(
                    (self.sigma**2 * val_s0 / self.mu)
                    * math.exp(self.mu * t)
                    * (math.exp(self.mu * t) - 1.0)
                )

        # General gamma: integrate E[S_t^2] = \int_0^\infty 2 s (1 - F(s)) ds
        mean = self.conditional_mean(t, s0=val_s0)
        approx_std = self.sigma * (val_s0**self.gamma) * math.sqrt(t)
        upper_limit = max(val_s0 * 10.0, mean + 10.0 * approx_std)

        def integrand(s: float) -> float:
            cdf = self.transition_cdf(s, t)
            return 2.0 * s * (1.0 - cdf)

        e_s2, _ = integrate.quad(integrand, 0.0, upper_limit, limit=100)
        var = max(e_s2 - mean**2, 0.0)
        return float(var)

    def conditional_std(self, t: float, s0: Optional[float] = None) -> float:
        r"""Conditional standard deviation std(S_t | S_0 = s_0)."""
        return float(math.sqrt(self.conditional_variance(t, s0=s0)))

    # ------------------------------------------------------------------
    # Exactness, Triplet & Properties
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason=(
                "CEVProcess has state-dependent local volatility σ(S) = σ S^{γ-1} and non-independent increments — "
                "it is not a Lévy process and has no Lévy–Khintchine triplet. "
                "Use .call_price(), .put_price(), .absorption_probability(), "
                ".conditional_mean(), .conditional_variance(), or .simulate()."
            ),
        )

    def _properties(self) -> ProcessProperties:
        regime_desc = (
            f"equity leverage regime (γ={self.gamma} < 1), 0 is absorbing"
            if self.gamma < 1.0
            else ("GBM regime (γ=1.0)" if abs(self.gamma - 1.0) < 1e-7 else f"commodity regime (γ={self.gamma} > 1)")
        )
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=(abs(self.mu) < 1e-12 and self.gamma <= 1.0),
            has_finite_variance=True,
            has_finite_mean=True,
            self_similarity_index=None,
            tail_index=None,
            is_subordinator=False,
            hurst_index=None,
            notes=[
                f"Constant Elasticity of Variance (CEV) process: dS_t = μ S_t dt + σ S_t^γ dW_t.",
                f"Parameters: μ={self.mu}, σ={self.sigma}, γ={self.gamma}, s0={self.s0}.",
                f"Local volatility: σ(S) = {self.sigma} * S^{self.gamma - 1.0:.3f}.",
                f"Regime: {regime_desc}.",
                "CEVProcess has state-dependent local volatility and non-independent increments.",
            ],
        )

    def _moment_propagation_cumulants(self, order: int) -> dict[int, float]:
        r"""Conditional cumulants at t = 1.0: κ_1 = mean, κ_2 = variance."""
        res: dict[int, float] = {}
        if order >= 1:
            res[1] = self.conditional_mean(1.0)
        if order >= 2:
            res[2] = self.conditional_variance(1.0)
        return res

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        n_steps: int,
        dt: Optional[float] = None,
        time_horizon: Optional[float] = None,
        n_paths: int = 1,
        rng: Optional[np.random.Generator] = None,
        method: str = "exact",
    ) -> np.ndarray:
        r"""
        Simulate sample paths of the CEV process.

        Parameters
        ----------
        n_steps : int
            Number of time steps.
        dt : float, optional
            Time step size Δt > 0.
        time_horizon : float, optional
            Terminal horizon T > 0.
        n_paths : int, default 1
            Number of independent sample paths.
        rng : np.random.Generator, optional
            Random number generator.
        method : {"exact", "milstein", "euler", "absorbed_euler"}, default "exact"
            Simulation scheme.

        Returns
        -------
        np.ndarray
            Path array of shape (n_paths, n_steps + 1) with paths[:, 0] == s0.
        """
        return sample_cev(
            mu=self.mu,
            sigma=self.sigma,
            gamma=self.gamma,
            s0=self.s0,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=time_horizon,
            rng=rng,
            method=method,
        )


# Aliases
CEV = CEVProcess
ConstantElasticityOfVariance = CEVProcess
