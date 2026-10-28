"""
SABR (Stochastic Alpha Beta Rho) stochastic volatility model.

The SABR model (Hagan et al. 2002) is a two-factor diffusion for a forward asset
or interest rate F_t and its stochastic volatility α_t:

    dF_t = α_t F_t^β dW_t^F,    F_0 = f_0 > 0
    dα_t = ν α_t dW_t^α,        α_0 = α > 0

with correlation:
    E[dW_t^F dW_t^α] = ρ dt,   ρ ∈ [-1, 1]

where:
    - F_t is the forward price / swap rate (martingale under the forward measure).
    - α_t is the stochastic volatility (geometric Brownian motion without drift).
    - β ∈ [0, 1] is the CEV elasticity exponent:
        * β = 0 : Normal / Bachelier stochastic volatility (linear backbone).
        * 0 < β < 1 : CEV stochastic volatility (power backbone).
        * β = 1 : Lognormal / Black stochastic volatility (flat backbone).
    - ν ≥ 0 is the volatility of volatility (vol-of-vol).
    - ρ ∈ [-1, 1] is the spot-volatility correlation (controlling smile skew).

The model provides closed-form asymptotic Black-76 implied volatilities
via the celebrated Hagan et al. (2002) formula.

References
----------
Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002).
Managing smile risk. *Wilmott Magazine*, September, 84–108.

Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2014).
Arbitrage-free SABR. *Wilmott Magazine*, 2014(69), 60–75.

Bartlett, B. (2006). Hedging under SABR model.
*Wilmott Magazine*, July/August, 64–71.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
from scipy.stats import norm

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.sim.exact import sample_sabr
from spxa.story.narrator import CompositionNode, NodeKind


class SABRProcess(Process):
    r"""
    SABR (Stochastic Alpha Beta Rho) stochastic volatility model.

    .. math::

        dF_t = \alpha_t F_t^\beta dW_t^F, \quad F_0 = f_0 > 0 \\
        d\alpha_t = \nu \alpha_t dW_t^\alpha, \quad \alpha_0 = \alpha > 0

    with correlation :math:`\mathbb{E}[dW_t^F dW_t^\alpha] = \rho dt`.

    Parameters
    ----------
    alpha :
        Initial volatility parameter :math:`\alpha > 0`. Default 0.20.
    beta :
        CEV elasticity exponent :math:`\beta \in [0, 1]`. Default 0.50.
    nu :
        Volatility of volatility :math:`\nu \ge 0`. Default 0.40. May also be supplied as ``vol_of_vol``.
    rho :
        Correlation between forward and volatility :math:`\rho \in [-1, 1]`. Default -0.30.
    f0 :
        Initial forward rate / price :math:`F_0 = f_0 > 0`. Default 100.0. May also be supplied as ``s0`` or ``forward``.
    """

    def __init__(
        self,
        alpha: float = 0.20,
        beta: float = 0.50,
        nu: float = 0.40,
        rho: float = -0.30,
        f0: float = 100.0,
        **kwargs,
    ) -> None:
        nu_val = kwargs.get("vol_of_vol", nu)
        f0_val = kwargs.get("s0", kwargs.get("forward", f0))

        alpha = float(alpha)
        beta = float(beta)
        nu_val = float(nu_val)
        rho = float(rho)
        f0_val = float(f0_val)

        if alpha <= 0.0:
            raise ValueError(f"Initial volatility alpha must be strictly positive, got {alpha}")
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"Elasticity exponent beta must lie in [0, 1], got {beta}")
        if nu_val < 0.0:
            raise ValueError(f"Vol-of-vol nu must be non-negative, got {nu_val}")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"Correlation rho must lie in [-1, 1], got {rho}")
        if f0_val <= 0.0:
            raise ValueError(f"Forward price f0 must be strictly positive, got {f0_val}")

        self._alpha = alpha
        self._beta = beta
        self._nu = nu_val
        self._rho = rho
        self._f0 = f0_val

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"SABRProcess(alpha={self._alpha}, beta={self._beta}, nu={self._nu}, rho={self._rho}, f0={self._f0})",
            math_note=(
                f"dF_t = α_t F_t^{self._beta} dW_t^F, dα_t = {self._nu} α_t dW_t^α; "
                f"ρ={self._rho}, F_0={self._f0}, α_0={self._alpha}"
            ),
            reference="Hagan, P. S. et al. (2002), Wilmott Magazine, 84–108",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def __repr__(self) -> str:
        return (
            f"SABRProcess(alpha={self._alpha}, beta={self._beta}, nu={self._nu}, "
            f"rho={self._rho}, f0={self._f0})"
        )

    # ------------------------------------------------------------------
    # Model Parameters & Structural Properties
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """Initial stochastic volatility :math:`\\alpha > 0`."""
        return self._alpha

    @property
    def beta(self) -> float:
        """CEV elasticity exponent :math:`\\beta \\in [0, 1]`."""
        return self._beta

    @property
    def nu(self) -> float:
        """Volatility of volatility :math:`\\nu \\ge 0`."""
        return self._nu

    @property
    def vol_of_vol(self) -> float:
        """Alias for vol-of-vol :math:`\\nu`."""
        return self._nu

    @property
    def rho(self) -> float:
        """Forward-volatility correlation :math:`\\rho \\in [-1, 1]`."""
        return self._rho

    @property
    def f0(self) -> float:
        """Initial forward level :math:`F_0 > 0`."""
        return self._f0

    @property
    def backbone(self) -> str:
        """Qualitative volatility smile backbone regime."""
        if self._beta == 0.0:
            return "normal"
        if self._beta == 1.0:
            return "lognormal"
        return "cev"

    # ------------------------------------------------------------------
    # Exactness & Triplet
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "SABRProcess has ExactnessLevel.MOMENT_PROPAGATION and does not possess "
            "a standard Levy triplet: both the forward price F_t and instantaneous "
            "volatility alpha_t are coupled state-dependent diffusions with stochastic variance, "
            "violating the independent and stationary increments requirement of pure Levy processes. "
            "Hagan asymptotic implied volatility formulas, European option pricing, and simulation "
            "are available directly."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=True,  # Forward price is a local martingale
            has_finite_mean=True,
            has_finite_variance=True,
            is_subordinator=False,
            notes=[
                (
                    f"SABR stochastic volatility model (Hagan et al. 2002): "
                    f"dF_t = α_t F_t^{self._beta} dW_t^F, dα_t = {self._nu} α_t dW_t^α, ρ={self._rho}."
                ),
                f"Parameters: alpha={self._alpha}, beta={self._beta}, nu={self._nu}, rho={self._rho}, f0={self._f0}.",
            ],
        )

    def _degradation_reason(self) -> str:
        return "SABR stochastic volatility model has coupled state-dependent diffusion dynamics."

    # ------------------------------------------------------------------
    # Hagan et al. (2002) Implied Volatility Formula
    # ------------------------------------------------------------------

    def implied_volatility(
        self,
        K: Union[float, np.ndarray],
        T: float = 1.0,
    ) -> Union[float, np.ndarray]:
        r"""
        Black-76 implied volatility :math:`\sigma_B(K, T)` via Hagan et al. (2002).

        Evaluates the closed-form asymptotic formula:

        .. math::

            \sigma_B(K, T) = \frac{\alpha}{(f_0 K)^{(1-\beta)/2} \left[ 1 + \frac{(1-\beta)^2}{24} \ln^2(f_0/K) + \frac{(1-\beta)^4}{1920} \ln^4(f_0/K) \right]} \cdot \frac{z}{\chi(z)} \cdot \left[ 1 + \left( \frac{(1-\beta)^2}{24} \frac{\alpha^2}{(f_0 K)^{1-\beta}} + \frac{1}{4} \frac{\rho \beta \nu \alpha}{(f_0 K)^{(1-\beta)/2}} + \frac{2 - 3\rho^2}{24} \nu^2 \right) T \right]

        where :math:`z = \frac{\nu}{\alpha} (f_0 K)^{(1-\beta)/2} \ln(f_0 / K)` and
        :math:`\chi(z) = \ln\left( \frac{\sqrt{1 - 2\rho z + z^2} + z - \rho}{1 - \rho} \right)`.

        Parameters
        ----------
        K :
            Strike price or array of strikes (:math:`K > 0`).
        T :
            Time to maturity in years (:math:`T > 0`). Default 1.0.

        Returns
        -------
        float or np.ndarray
        """
        if T <= 0.0:
            raise ValueError(f"Maturity T must be strictly positive, got {T}")

        f0 = self._f0
        alpha = self._alpha
        beta = self._beta
        nu = self._nu
        rho = self._rho
        one_minus_beta = 1.0 - beta

        def _iv_single(strike: float) -> float:
            if strike <= 0.0:
                raise ValueError(f"Strike K must be strictly positive, got {strike}")

            # At-the-money (ATM) case
            if abs(strike - f0) < 1e-10:
                return float(self.atm_volatility(T=T))

            log_fk = math.log(f0 / strike)
            f_k_mid = (f0 * strike) ** (one_minus_beta / 2.0)

            if nu == 0.0:
                z_over_chi = 1.0
            else:
                z = (nu / alpha) * f_k_mid * log_fk
                if abs(z) < 1e-8:
                    z_over_chi = 1.0
                else:
                    arg = (math.sqrt(max(0.0, 1.0 - 2.0 * rho * z + z**2)) + z - rho) / (1.0 - rho)
                    chi = math.log(max(1e-15, arg))
                    z_over_chi = z / chi

            denom = f_k_mid * (
                1.0
                + (one_minus_beta**2 / 24.0) * (log_fk**2)
                + (one_minus_beta**4 / 1920.0) * (log_fk**4)
            )
            num = alpha * z_over_chi

            corr_factor = (
                1.0
                + (
                    ((one_minus_beta**2 / 24.0) * (alpha**2 / ((f0 * strike) ** one_minus_beta)))
                    + (0.25 * rho * beta * nu * alpha / f_k_mid)
                    + ((2.0 - 3.0 * rho**2) / 24.0) * (nu**2)
                )
                * T
            )

            vol = (num / denom) * corr_factor
            return float(max(1e-6, vol))

        K_arr = np.asarray(K)
        if K_arr.ndim == 0:
            return _iv_single(float(K_arr))
        return np.array([_iv_single(float(k)) for k in K_arr], dtype=float)

    def atm_volatility(self, T: float = 1.0) -> float:
        r"""
        At-the-money (ATM) Black implied volatility :math:`\sigma_B(f_0, T)`.

        .. math::

            \sigma_B(f_0, T) = \frac{\alpha}{f_0^{1-\beta}} \left[ 1 + \left( \frac{(1-\beta)^2}{24} \frac{\alpha^2}{f_0^{2(1-\beta)}} + \frac{1}{4} \frac{\rho \beta \nu \alpha}{f_0^{1-\beta}} + \frac{2 - 3\rho^2}{24} \nu^2 \right) T \right]
        """
        if T <= 0.0:
            raise ValueError(f"Maturity T must be strictly positive, got {T}")
        f0 = self._f0
        alpha = self._alpha
        beta = self._beta
        nu = self._nu
        rho = self._rho
        one_minus_beta = 1.0 - beta

        term1 = alpha / (f0**one_minus_beta)
        term2 = (
            1.0
            + (
                ((one_minus_beta**2 / 24.0) * (alpha**2 / (f0 ** (2.0 * one_minus_beta))))
                + (0.25 * rho * beta * nu * alpha / (f0**one_minus_beta))
                + ((2.0 - 3.0 * rho**2) / 24.0) * (nu**2)
            )
            * T
        )
        return float(term1 * term2)

    def implied_vol_smile(
        self,
        strikes: Union[list[float], np.ndarray],
        T: float = 1.0,
    ) -> np.ndarray:
        """Compute Black-76 implied volatility smile across an array of strikes."""
        return self.implied_volatility(strikes, T=T)

    # ------------------------------------------------------------------
    # European Option Pricing (Black-76 with Hagan Volatility)
    # ------------------------------------------------------------------

    def call_price(
        self,
        K: Union[float, np.ndarray],
        T: float = 1.0,
        r: float = 0.0,
    ) -> Union[float, np.ndarray]:
        r"""
        European call option price :math:`C(K, T)` on forward :math:`F_t` via Black-76:

        .. math::

            C(K, T) = e^{-r T} \left[ f_0 \Phi(d_1) - K \Phi(d_2) \right]
        """
        if T <= 0.0:
            raise ValueError(f"Maturity T must be strictly positive, got {T}")
        df = math.exp(-r * T)
        sigma = self.implied_volatility(K, T=T)

        def _call_single(k: float, vol: float) -> float:
            if vol <= 1e-8:
                return float(df * max(0.0, self._f0 - k))
            sqrt_t = math.sqrt(T)
            d1 = (math.log(self._f0 / k) + 0.5 * (vol**2) * T) / (vol * sqrt_t)
            d2 = d1 - vol * sqrt_t
            price = df * (self._f0 * norm.cdf(d1) - k * norm.cdf(d2))
            return float(max(price, df * max(0.0, self._f0 - k)))

        K_arr = np.asarray(K)
        if K_arr.ndim == 0:
            return _call_single(float(K_arr), float(sigma))
        return np.array(
            [_call_single(float(k), float(v)) for k, v in zip(K_arr, sigma)], dtype=float
        )

    def put_price(
        self,
        K: Union[float, np.ndarray],
        T: float = 1.0,
        r: float = 0.0,
    ) -> Union[float, np.ndarray]:
        r"""
        European put option price :math:`P(K, T)` via Put-Call parity:

        .. math::

            P(K, T) = C(K, T) - e^{-r T} (f_0 - K)
        """
        calls = self.call_price(K, T=T, r=r)
        df = math.exp(-r * T)
        K_arr = np.asarray(K, dtype=float)
        puts = calls - df * (self._f0 - K_arr)
        intrinsic = df * np.maximum(0.0, K_arr - self._f0)
        return np.maximum(puts, intrinsic)

    # ------------------------------------------------------------------
    # Forward & Volatility Moments
    # ------------------------------------------------------------------

    def expected_forward(self, t: float = 1.0) -> float:
        """Expected forward :math:`\\mathbb{E}[F_t] = f_0` (martingale property)."""
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        return self._f0

    def volatility_mean(self, t: float = 1.0) -> float:
        """Expected volatility :math:`\\mathbb{E}[\\alpha_t] = \\alpha` (geometric Brownian motion)."""
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        return self._alpha

    def volatility_variance(self, t: float = 1.0) -> float:
        """Variance of volatility :math:`\\mathrm{Var}(\\alpha_t) = \\alpha^2 (e^{\\nu^2 t} - 1)`."""
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        return float((self._alpha**2) * (math.exp(self._nu**2 * t) - 1.0))

    def log_volatility_mean(self, t: float = 1.0) -> float:
        """Expected log-volatility :math:`\\mathbb{E}[\\ln \\alpha_t] = \\ln \\alpha - \\frac{1}{2} \\nu^2 t`."""
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        return float(math.log(self._alpha) - 0.5 * (self._nu**2) * t)

    def log_volatility_variance(self, t: float = 1.0) -> float:
        """Variance of log-volatility :math:`\\mathrm{Var}(\\ln \\alpha_t) = \\nu^2 t`."""
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        return float((self._nu**2) * t)

    def conditional_mean(self, t: float = 1.0) -> float:
        """Conditional mean of forward price :math:`F_t`."""
        return self.expected_forward(t)

    def conditional_cumulants(self, t: float = 1.0, order: int = 1) -> dict[int, float]:
        """First-order cumulant of forward price :math:`F_t`."""
        if order < 1:
            raise ValueError(f"Order must be >= 1, got {order}")
        return {1: self.conditional_mean(t)}

    def cumulants(self, order: int) -> dict[int, float]:
        """Cumulants at horizon :math:`t = 1.0`."""
        return self.conditional_cumulants(t=1.0, order=order)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        n_steps: Optional[int] = None,
        dt: Optional[float] = None,
        time_horizon: Optional[float] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "log_euler",
        T: Optional[float] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of forward price :math:`F_t`.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_paths, n_steps + 1)`` with :math:`F_0 = f_0`.
        """
        final_steps = (
            n_steps if n_steps is not None else (num_steps if num_steps is not None else 100)
        )
        final_paths = (
            n_paths if n_paths is not None else (num_paths if num_paths is not None else 1)
        )
        final_t = (
            T
            if T is not None
            else (
                time_horizon if time_horizon is not None else (t_max if t_max is not None else 1.0)
            )
        )

        return sample_sabr(
            alpha=self._alpha,
            beta=self._beta,
            nu=self._nu,
            rho=self._rho,
            f0=self._f0,
            dt=dt,
            n_steps=final_steps,
            n_paths=final_paths,
            T=final_t,
            rng=rng,
            method=method,
            return_volatility=False,
        )

    def simulate_joint(
        self,
        n_steps: Optional[int] = None,
        dt: Optional[float] = None,
        time_horizon: Optional[float] = None,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        method: str = "log_euler",
        T: Optional[float] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Simulate coupled paths of forward price :math:`F_t` and stochastic volatility :math:`\alpha_t`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Paths of :math:`F_t` and :math:`\alpha_t`, each of shape ``(n_paths, n_steps + 1)``.
        """
        final_steps = (
            n_steps if n_steps is not None else (num_steps if num_steps is not None else 100)
        )
        final_paths = (
            n_paths if n_paths is not None else (num_paths if num_paths is not None else 1)
        )
        final_t = (
            T
            if T is not None
            else (
                time_horizon if time_horizon is not None else (t_max if t_max is not None else 1.0)
            )
        )

        return sample_sabr(
            alpha=self._alpha,
            beta=self._beta,
            nu=self._nu,
            rho=self._rho,
            f0=self._f0,
            dt=dt,
            n_steps=final_steps,
            n_paths=final_paths,
            T=final_t,
            rng=rng,
            method=method,
            return_volatility=True,
        )


# Aliases
SABRModel = SABRProcess
SABR = SABRProcess
