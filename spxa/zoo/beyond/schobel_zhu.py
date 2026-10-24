"""
Schöbel-Zhu (1999) Stochastic Volatility Model.

A continuous-time stochastic volatility model where instantaneous volatility
follows an Ornstein-Uhlenbeck Gaussian diffusion rather than CIR variance:
    dS_t = (r - q) S_t dt + σ_t S_t dW_t^S,             S_0 > 0
    dσ_t = κ (θ - σ_t) dt + τ dW_t^σ,                   σ_0 ∈ ℝ
    E[dW_t^S dW_t^σ] = ρ dt

Unlike the Heston model, volatility itself (rather than variance) is mean-reverting
and Gaussian, ensuring that variance v_t = σ_t² is always non-negative, admitting
affine characteristic functions and closed-form European option pricing.

Parameters
----------
sigma0 : float
    Initial volatility (σ_0 ∈ ℝ). Alias: v0.
kappa : float
    Mean-reversion speed of volatility (κ > 0).
theta : float
    Long-term mean level of volatility (θ ∈ ℝ).
tau : float
    Volatility of volatility (τ > 0).
rho : float
    Correlation between asset price and volatility shocks (ρ ∈ [-1, 1]).
s0 : float
    Initial asset price (S_0 > 0). Default 100.0.
rate : float
    Risk-free interest rate (r). Default 0.0.
dividend : float
    Continuous dividend yield (q). Default 0.0.

References
----------
Schöbel, R., & Zhu, J. (1999). An analytic model for option pricing with
    lognormal and non-lognormal volatility processes. Review of Derivatives
    Research, 3(1), 5–31.
Lord, R., & Kahl, C. (2007). Complex logarithm and implementation of the
    Heston and Schöbel-Zhu models. Journal of Computational Finance.
"""

from __future__ import annotations

import math
from typing import Optional, Union, Tuple

import numpy as np
from scipy import integrate, optimize  # type: ignore[import-untyped]
from scipy.integrate import solve_ivp  # type: ignore[import-untyped]

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class SchobelZhuProcess(Process):
    r"""
    Schöbel-Zhu (1999) stochastic volatility model.

    .. math::

        dS_t = (r - q) S_t dt + \sigma_t S_t dW_t^S \\
        d\sigma_t = \kappa (\theta - \sigma_t) dt + \tau dW_t^\sigma

    with :math:`\mathbb{E}[dW_t^S dW_t^\sigma] = \rho dt`.
    """

    def __init__(
        self,
        sigma0: float = 0.20,
        kappa: float = 1.5,
        theta: float = 0.20,
        tau: float = 0.30,
        rho: float = -0.5,
        s0: float = 100.0,
        rate: float = 0.0,
        dividend: float = 0.0,
    ) -> None:
        if kappa <= 0.0:
            raise ValueError(f"Mean-reversion speed 'kappa' must be strictly positive, got {kappa}")
        if tau <= 0.0:
            raise ValueError(f"Vol-of-vol 'tau' must be strictly positive, got {tau}")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"Correlation 'rho' must lie in [-1, 1], got {rho}")
        if s0 <= 0.0:
            raise ValueError(f"Initial asset price 's0' must be strictly positive, got {s0}")

        self._sigma0 = float(sigma0)
        self._kappa = float(kappa)
        self._theta = float(theta)
        self._tau = float(tau)
        self._rho = float(rho)
        self._s0 = float(s0)
        self._rate = float(rate)
        self._dividend = float(dividend)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"SchobelZhu(sigma0={self._sigma0}, kappa={self._kappa}, theta={self._theta}, tau={self._tau}, rho={self._rho})",
            math_note=(
                f"Schöbel-Zhu OU volatility diffusion: dσ_t = {self._kappa:.2f}({self._theta:.2f} - σ_t)dt + {self._tau:.2f}dW_t^σ, "
                f"ρ={self._rho:+.2f}"
            ),
            reference="Schöbel & Zhu (1999); Lord & Kahl (2007)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sigma0(self) -> float:
        r"""Initial volatility :math:`\sigma_0`."""
        return self._sigma0

    @property
    def v0(self) -> float:
        """Alias for initial volatility."""
        return self._sigma0

    @property
    def kappa(self) -> float:
        r"""Mean-reversion speed :math:`\kappa > 0`."""
        return self._kappa

    @property
    def theta(self) -> float:
        r"""Long-term mean volatility :math:`\theta`."""
        return self._theta

    @property
    def tau(self) -> float:
        r"""Volatility of volatility :math:`\tau > 0`."""
        return self._tau

    @property
    def rho(self) -> float:
        r"""Correlation :math:`\rho \in [-1, 1]`."""
        return self._rho

    @property
    def s0(self) -> float:
        r"""Initial asset price :math:`S_0 > 0`."""
        return self._s0

    @property
    def rate(self) -> float:
        """Risk-free interest rate."""
        return self._rate

    @property
    def dividend(self) -> float:
        """Continuous dividend yield."""
        return self._dividend

    # ------------------------------------------------------------------
    # Exactness & Triplet
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "SchobelZhuProcess has ExactnessLevel.MOMENT_PROPAGATION and does not possess "
            "a standard Levy triplet: instantaneous volatility sigma_t is a continuous "
            "Ornstein-Uhlenbeck diffusion coupled with asset price S_t, violating the "
            "independent and stationary increments condition of pure Levy processes. "
            "Characteristic function, option pricing via Gil-Pelaez inversion, and "
            "simulation are available directly."
        )

    def _properties(self) -> ProcessProperties:
        is_mg = math.isclose(self._rate - self._dividend, 0.0, abs_tol=1e-10)
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=is_mg,
            has_finite_mean=True,
            has_finite_variance=True,
            is_subordinator=False,
            notes=[
                (
                    f"Schöbel-Zhu stochastic volatility model: "
                    f"dσ_t = {self._kappa:.2f}({self._theta:.2f} - σ_t)dt + {self._tau:.2f}dW_t^σ, ρ={self._rho:+.2f}."
                ),
                f"S0={self._s0}, sigma0={self._sigma0}, r={self._rate}, q={self._dividend}.",
            ],
        )

    def _degradation_reason(self) -> str:
        return "Schöbel-Zhu model features coupled state-dependent OU stochastic volatility."

    # ------------------------------------------------------------------
    # Volatility Moments
    # ------------------------------------------------------------------

    def volatility_mean(self, t: float = 1.0) -> float:
        r"""
        Expected volatility :math:`\mathbb{E}[\sigma_t \mid \sigma_0] = \theta + (\sigma_0 - \theta) e^{-\kappa t}`.
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        return float(self._theta + (self._sigma0 - self._theta) * math.exp(-self._kappa * t))

    def volatility_variance(self, t: float = 1.0) -> float:
        r"""
        Variance of volatility :math:`\mathrm{Var}(\sigma_t \mid \sigma_0) = \frac{\tau^2}{2\kappa}(1 - e^{-2\kappa t})`.
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        return float(
            (self._tau**2 / (2.0 * self._kappa)) * (1.0 - math.exp(-2.0 * self._kappa * t))
        )

    def variance_mean(self, t: float = 1.0) -> float:
        r"""
        Expected instantaneous variance :math:`\mathbb{E}[\sigma_t^2] = \mathrm{Var}(\sigma_t) + (\mathbb{E}[\sigma_t])^2`.
        """
        m = self.volatility_mean(t)
        v = self.volatility_variance(t)
        return float(v + m**2)

    def variance_variance(self, t: float = 1.0) -> float:
        r"""
        Variance of instantaneous variance :math:`\mathrm{Var}(\sigma_t^2) = 2 \mathrm{Var}(\sigma_t)^2 + 4 (\mathbb{E}[\sigma_t])^2 \mathrm{Var}(\sigma_t)`.
        """
        m = self.volatility_mean(t)
        v = self.volatility_variance(t)
        return float(2.0 * (v**2) + 4.0 * (m**2) * v)

    @property
    def stationary_volatility_mean(self) -> float:
        """Asymptotic mean of volatility :math:`\theta`."""
        return self._theta

    @property
    def stationary_volatility_var(self) -> float:
        r"""Asymptotic variance of volatility :math:`\frac{\tau^2}{2\kappa}`."""
        return self._tau**2 / (2.0 * self._kappa)

    @property
    def stationary_variance_mean(self) -> float:
        r"""Asymptotic mean of instantaneous variance :math:`\theta^2 + \frac{\tau^2}{2\kappa}`."""
        return self._theta**2 + self.stationary_volatility_var

    # ------------------------------------------------------------------
    # Characteristic Function
    # ------------------------------------------------------------------

    def char_func(self, u: Union[float, complex, np.ndarray], t: float = 1.0) -> np.ndarray:
        r"""
        Characteristic function of log-price :math:`x_t = \ln(S_t / S_0)`.

        .. math::

            \phi(u; t) = \mathbb{E}\left[ \exp(i u \ln(S_t/S_0)) \right] = \exp\left( C(u, t) + D(u, t)\sigma_0 + E(u, t)\sigma_0^2 + i u (r - q) t \right)
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")

        def _cf_single(u_val: complex) -> complex:
            if abs(u_val) < 1e-12:
                return 1.0 + 0.0j

            a_E = 2.0 * (self._tau**2)
            b_E = 2.0 * (self._rho * self._tau * 1j * u_val - self._kappa)
            c_E = -0.5 * (u_val**2 + 1j * u_val)

            def odes(s: float, y: np.ndarray) -> list[complex]:
                E_val, D_val = y[0], y[1]
                dE = a_E * (E_val**2) + b_E * E_val + c_E
                dD = (a_E * E_val + 0.5 * b_E) * D_val + 2.0 * self._kappa * self._theta * E_val
                dC = (
                    self._kappa * self._theta * D_val
                    + (self._tau**2) * E_val
                    + 0.5 * (self._tau**2) * (D_val**2)
                )
                return [dE, dD, dC]

            sol = solve_ivp(odes, [0.0, t], [0j, 0j, 0j], t_eval=[t], rtol=1e-8, atol=1e-8)
            E, D, C = sol.y[:, 0]
            drift = 1j * u_val * (self._rate - self._dividend) * t
            return complex(np.exp(drift + C + D * self._sigma0 + E * (self._sigma0**2)))

        u_arr = np.asarray(u, dtype=complex)
        if u_arr.ndim == 0:
            return _cf_single(complex(u_arr))
        return np.array([_cf_single(val) for val in u_arr])

    # ------------------------------------------------------------------
    # European Option Pricing & Parity
    # ------------------------------------------------------------------

    def call_price(
        self,
        K: Union[float, np.ndarray],
        T: float = 1.0,
        r: Optional[float] = None,
        q: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        r"""
        European call option price via Gil-Pelaez Fourier inversion.
        """
        rate = self._rate if r is None else r
        div = self._dividend if q is None else q

        def _price_single(strike: float) -> float:
            k_log = math.log(strike / self._s0)
            phi_minus_i = self.char_func(-1j, t=T)

            def integrand1(u: float) -> float:
                if u == 0.0:
                    return 0.0
                val = (self.char_func(u - 1j, t=T) * np.exp(-1j * u * k_log)) / (
                    1j * u * phi_minus_i
                )
                return float(val.real)

            def integrand2(u: float) -> float:
                if u == 0.0:
                    return 0.0
                val = (self.char_func(u, t=T) * np.exp(-1j * u * k_log)) / (1j * u)
                return float(val.real)

            res1, _ = integrate.quad(integrand1, 1e-6, 50.0, limit=100)
            res2, _ = integrate.quad(integrand2, 1e-6, 50.0, limit=100)

            pi1 = 0.5 + res1 / math.pi
            pi2 = 0.5 + res2 / math.pi

            c = self._s0 * math.exp(-div * T) * pi1 - strike * math.exp(-rate * T) * pi2
            intrinsic = max(0.0, self._s0 * math.exp(-div * T) - strike * math.exp(-rate * T))
            return float(max(intrinsic, c))

        K_arr = np.asarray(K)
        if K_arr.ndim == 0:
            return _price_single(float(K_arr))
        return np.array([_price_single(float(k)) for k in K_arr], dtype=float)

    def put_price(
        self,
        K: Union[float, np.ndarray],
        T: float = 1.0,
        r: Optional[float] = None,
        q: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        r"""
        European put option price via Put-Call parity.
        """
        rate = self._rate if r is None else r
        div = self._dividend if q is None else q
        calls = self.call_price(K, T=T, r=rate, q=div)

        df_r = math.exp(-rate * T)
        df_q = math.exp(-div * T)
        K_arr = np.asarray(K, dtype=float)

        puts = calls - self._s0 * df_q + K_arr * df_r
        intrinsic = np.maximum(0.0, K_arr * df_r - self._s0 * df_q)
        return np.maximum(puts, intrinsic)

    def implied_volatility(
        self,
        K: Union[float, np.ndarray],
        T: float = 1.0,
        r: Optional[float] = None,
        q: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        r"""
        Black-Scholes implied volatility inverted from call price.
        """
        rate = self._rate if r is None else r
        div = self._dividend if q is None else q
        calls = self.call_price(K, T=T, r=rate, q=div)

        def _bs_call(s: float, k: float, t: float, sig: float) -> float:
            if sig <= 1e-8:
                return max(0.0, s * math.exp(-div * t) - k * math.exp(-rate * t))
            d1 = (math.log(s / k) + (rate - div + 0.5 * sig**2) * t) / (sig * math.sqrt(t))
            d2 = d1 - sig * math.sqrt(t)
            from scipy.stats import norm  # type: ignore[import-untyped]

            return s * math.exp(-div * t) * float(norm.cdf(d1)) - k * math.exp(-rate * t) * float(
                norm.cdf(d2)
            )

        def _iv_single(k: float, target: float) -> float:
            intrinsic = max(0.0, self._s0 * math.exp(-div * T) - k * math.exp(-rate * T))
            if target <= intrinsic + 1e-9:
                return 1e-4

            def obj(sig: float) -> float:
                return _bs_call(self._s0, k, T, sig) - target

            try:
                return float(optimize.brentq(obj, 1e-4, 5.0, xtol=1e-6))
            except Exception:
                return float("nan")

        K_arr = np.asarray(K)
        if K_arr.ndim == 0:
            return _iv_single(float(K_arr), float(calls))
        return np.array(
            [_iv_single(float(k), float(c)) for k, c in zip(K_arr, np.asarray(calls))], dtype=float
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        n_steps: int = 100,
        dt: Optional[float] = None,
        time_horizon: Optional[float] = None,
        n_paths: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of asset price :math:`S_t`.
        """
        from spxa.sim.exact import sample_schobel_zhu

        return sample_schobel_zhu(
            sigma0=self._sigma0,
            kappa=self._kappa,
            theta=self._theta,
            tau=self._tau,
            rho=self._rho,
            s0=self._s0,
            rate=self._rate,
            dividend=self._dividend,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=time_horizon,
            rng=rng,
            return_volatility=False,
        )

    def simulate_joint(
        self,
        n_steps: int = 100,
        dt: Optional[float] = None,
        time_horizon: Optional[float] = None,
        n_paths: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Simulate coupled paths of asset price :math:`S_t` and volatility :math:`\sigma_t`.
        """
        from spxa.sim.exact import sample_schobel_zhu

        return sample_schobel_zhu(
            sigma0=self._sigma0,
            kappa=self._kappa,
            theta=self._theta,
            tau=self._tau,
            rho=self._rho,
            s0=self._s0,
            rate=self._rate,
            dividend=self._dividend,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=time_horizon,
            rng=rng,
            return_volatility=True,
        )

    def __repr__(self) -> str:
        return (
            f"SchobelZhuProcess(sigma0={self._sigma0}, kappa={self._kappa}, "
            f"theta={self._theta}, tau={self._tau}, rho={self._rho}, s0={self._s0})"
        )


# Aliases
SchobelZhu = SchobelZhuProcess
SchobelZhuModel = SchobelZhuProcess
SZModel = SchobelZhuProcess
