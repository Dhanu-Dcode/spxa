"""
Implied volatility surface morphing and static arbitrage safeguards.

Defines the continuous homotopy between implied volatility surfaces:

    w_α(k, T) = (1 - α) w_0(k, T) + α w_1(k, T)

where w(k, T) = σ_{IV}²(k, T) · T is the total implied variance, with rigorous
verification of:
  1. Calendar arbitrage absence: ∂_T w_α(k, T) ≥ 0
  2. Butterfly arbitrage absence: Durrleman-Gatheral condition g_α(k, T) ≥ 0
  3. Breeden-Litzenberger risk-neutral density extraction:
     q_α(K) = ∂²C_α / ∂K² ≥ 0

References:
  Gatheral, J. & Jacquier, A. (2014), Arbitrage-free SVI volatility surfaces;
  Carr, P. & Madan, D. (2005), A note on sufficient conditions for no arbitrage;
  Breeden, D. T. & Litzenberger, R. H. (1978), Prices of state-contingent claims.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Tuple

import numpy as np
from scipy.stats import norm

from spxa.morph.base import AbstractMorphEngine


class SurfaceMorphEngine(AbstractMorphEngine):
    r"""
    Morphing engine for total implied variance surfaces :math:`w(k, T)`.
    """

    def morph(
        self,
        vol_fn0: Callable[[float, float], float],
        vol_fn1: Callable[[float, float], float],
        alpha: float,
        spot: float = 100.0,
        rate: float = 0.0,
        **kwargs: Any,
    ) -> MorphedVolatilitySurface:
        return MorphedVolatilitySurface(
            vol_fn0=vol_fn0,
            vol_fn1=vol_fn1,
            alpha=alpha,
            spot=spot,
            rate=rate,
        )


class MorphedVolatilitySurface:
    r"""
    A morphed implied volatility surface parameterized by :math:`\alpha \in [0, 1]`.

    Parameters
    ----------
    vol_fn0 :
        Function :math:`(k, T) \mapsto \sigma_0(k, T)` for base model.
    vol_fn1 :
        Function :math:`(k, T) \mapsto \sigma_1(k, T)` for target model.
    alpha :
        Homotopy parameter in :math:`[0, 1]`.
    spot :
        Underlying spot price :math:`S_0`.
    rate :
        Risk-free interest rate :math:`r`.
    """

    def __init__(
        self,
        vol_fn0: Callable[[float, float], float],
        vol_fn1: Callable[[float, float], float],
        alpha: float = 0.5,
        spot: float = 100.0,
        rate: float = 0.0,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.vol_fn0 = vol_fn0
        self.vol_fn1 = vol_fn1
        self.alpha = float(alpha)
        self.spot = float(spot)
        self.rate = float(rate)

    def total_variance(self, k: float, T: float) -> float:
        r"""Morphed total variance :math:`w_\alpha(k, T) = (1 - \alpha) w_0 + \alpha w_1`."""
        T = max(1e-6, float(T))
        v0 = max(1e-6, float(self.vol_fn0(k, T)))
        v1 = max(1e-6, float(self.vol_fn1(k, T)))
        w0 = (v0**2) * T
        w1 = (v1**2) * T
        return (1.0 - self.alpha) * w0 + self.alpha * w1

    def implied_vol(self, k: float, T: float) -> float:
        r"""Morphed Black-Scholes implied volatility :math:`\sigma_\alpha(k, T) = \sqrt{w_\alpha / T}`."""
        T = max(1e-6, float(T))
        w = self.total_variance(k, T)
        return math.sqrt(max(1e-8, w / T))

    def evaluate_grid(
        self,
        k_grid: np.ndarray,
        t_grid: np.ndarray,
    ) -> np.ndarray:
        r"""Evaluate implied volatility surface over 2D grid of :math:`(k, T)`."""
        vol_matrix = np.zeros((len(t_grid), len(k_grid)), dtype=np.float64)
        for i, T in enumerate(t_grid):
            for j, k in enumerate(k_grid):
                vol_matrix[i, j] = self.implied_vol(float(k), float(T))
        return vol_matrix

    def check_calendar_arbitrage(
        self,
        k: float,
        t_grid: np.ndarray,
    ) -> Tuple[bool, np.ndarray]:
        r"""
        Verify calendar spread arbitrage condition :math:`\partial_T w_\alpha(k, T) \ge 0`.
        """
        w_vals = np.array([self.total_variance(k, float(T)) for T in t_grid])
        dw = np.diff(w_vals)
        is_arbitrage_free = bool(np.all(dw >= -1e-8))
        return is_arbitrage_free, dw

    def check_butterfly_arbitrage(
        self,
        k_grid: np.ndarray,
        T: float,
        eps: float = 1e-3,
    ) -> Tuple[bool, np.ndarray]:
        r"""
        Verify Gatheral-Jacquier butterfly arbitrage absence condition :math:`g_\alpha(k, T) \ge 0`:

        .. math::
            g(k) = \left(1 - \frac{k w'}{2w}\right)^2 - \frac{w'}{4}\left(\frac{1}{w} + \frac{1}{4}\right) + \frac{w''}{2} \ge 0
        """
        g_vals = np.zeros_like(k_grid, dtype=np.float64)

        for i, k in enumerate(k_grid):
            w = self.total_variance(k, T)
            w_plus = self.total_variance(k + eps, T)
            w_minus = self.total_variance(k - eps, T)

            w_prime = (w_plus - w_minus) / (2.0 * eps)
            w_second = (w_plus - 2.0 * w + w_minus) / (eps**2)

            term1 = (1.0 - (k * w_prime) / (2.0 * w)) ** 2
            term2 = (w_prime / 4.0) * (1.0 / w + 0.25)
            term3 = 0.5 * w_second

            g_vals[i] = term1 - term2 + term3

        is_arbitrage_free = bool(np.all(g_vals >= -1e-6))
        return is_arbitrage_free, g_vals

    def option_price(
        self,
        strike: float,
        T: float,
        is_call: bool = True,
    ) -> float:
        r"""Black-Scholes option price using morphed implied volatility."""
        forward = self.spot * math.exp(self.rate * T)
        k = math.log(strike / forward)
        sigma = self.implied_vol(k, T)
        vol_sqrt_t = sigma * math.sqrt(T)

        d1 = (-k + 0.5 * (sigma**2) * T) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        df = math.exp(-self.rate * T)

        if is_call:
            return df * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
        return df * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))

    def risk_neutral_density(
        self,
        strikes: np.ndarray,
        T: float,
        d_k: float = 1e-2,
    ) -> np.ndarray:
        r"""
        Extract Breeden-Litzenberger risk-neutral probability density :math:`q_\alpha(K)`:

        .. math::
            q_\alpha(K) = e^{r T} \frac{\partial^2 C_\alpha}{\partial K^2}
        """
        densities = np.zeros_like(strikes, dtype=np.float64)
        df_inv = math.exp(self.rate * T)

        for i, K in enumerate(strikes):
            c_mid = self.option_price(K, T, is_call=True)
            c_plus = self.option_price(K + d_k, T, is_call=True)
            c_minus = self.option_price(K - d_k, T, is_call=True)
            d2c_dk2 = (c_plus - 2.0 * c_mid + c_minus) / (d_k**2)
            densities[i] = max(0.0, df_inv * d2c_dk2)

        return densities
