"""
Schwartz Two-Factor Commodity Model.

Canonical continuous-time commodity pricing model introduced by Eduardo Schwartz (1997).
The model captures mean reversion in commodity convenience yields and stochastic spot prices.

Physical Measure (P):
    dS_t = (μ - δ_t) S_t dt + σ_1 S_t dW_1(t)
    dδ_t = κ (α - δ_t) dt + σ_2 dW_2(t)
    dW_1(t) dW_2(t) = ρ dt

Risk-Neutral Measure (Q):
    dS_t = (r - δ_t) S_t dt + σ_1 S_t dW_1^Q(t)
    dδ_t = κ (α_hat - δ_t) dt + σ_2 dW_2^Q(t)
    where α_hat = α - λ / κ is the risk-neutral long-term convenience yield target,
    and λ is the market price of convenience yield risk.

Log-Spot Transformation (X_t = ln S_t):
    dX_t = (r - δ_t - 0.5 σ_1²) dt + σ_1 dW_1^Q(t)

Because (X_t, δ_t) form a joint bivariate linear Gaussian diffusion system:
    - Futures price F(S_t, δ_t, τ) has an exact closed-form exponential formula (Schwartz 1997 eq. 12).
    - Log-spot prices X_T are normally distributed with exact conditional mean and variance.
    - European call and put options on spot and futures have closed-form Black-76 formulas.
    - Put-Call parity is satisfied exactly.

References
----------
Schwartz, E. S. (1997). The stochastic behavior of commodity prices: Implications for
    valuation and hedging. The Journal of Finance, 52(3), 923–973.
Bjerksund, P. (1991). Contingent claims evaluation when interest rates and convenience
    yields are stochastic. Working Paper, Norwegian School of Economics.
"""

from __future__ import annotations

import math
from typing import Optional, Union, Tuple

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class SchwartzTwoFactor(Process):
    r"""
    Schwartz Two-Factor commodity model (Schwartz, 1997).

    .. math::

        dS_t = (r - \delta_t) S_t dt + \sigma_1 S_t dW_1^Q(t), \quad S_0 > 0 \\
        d\delta_t = \kappa (\hat{\alpha} - \delta_t) dt + \sigma_2 dW_2^Q(t), \quad \delta_0 \in \mathbb{R} \\
        \mathbb{E}[dW_1^Q(t) dW_2^Q(t)] = \rho dt

    Parameters
    ----------
    s0 : float
        Initial commodity spot price :math:`S_0 > 0`. Default 20.0.
    delta0 : float
        Initial convenience yield :math:`\delta_0 \in \mathbb{R}`. Default 0.05.
    kappa : float
        Mean-reversion speed of convenience yield :math:`\kappa > 0`. Default 1.2.
    alpha : float
        Long-term convenience yield level :math:`\alpha \in \mathbb{R}`. Default 0.08.
    sigma1 : float
        Commodity spot price volatility :math:`\sigma_1 > 0`. Default 0.30.
    sigma2 : float
        Convenience yield volatility :math:`\sigma_2 > 0`. Default 0.20.
    rho : float
        Correlation between spot and convenience yield shocks :math:`\rho \in [-1, 1]`. Default 0.60.
    rate : float
        Risk-free interest rate :math:`r`. Default 0.05.
    lambda_yield : float
        Market price of convenience yield risk :math:`\lambda`. Default 0.0.
    alpha_hat : float, optional
        Directly specify risk-neutral target :math:`\hat{\alpha} = \alpha - \lambda/\kappa`.
    """

    def __init__(
        self,
        s0: float = 20.0,
        delta0: float = 0.05,
        kappa: float = 1.2,
        alpha: float = 0.08,
        sigma1: float = 0.30,
        sigma2: float = 0.20,
        rho: float = 0.60,
        rate: float = 0.05,
        lambda_yield: float = 0.0,
        alpha_hat: Optional[float] = None,
    ) -> None:
        if s0 <= 0.0:
            raise ValueError(f"Initial spot price 's0' must be strictly positive, got {s0}")
        if kappa <= 0.0:
            raise ValueError(f"Mean-reversion speed 'kappa' must be strictly positive, got {kappa}")
        if sigma1 <= 0.0:
            raise ValueError(f"Spot volatility 'sigma1' must be strictly positive, got {sigma1}")
        if sigma2 <= 0.0:
            raise ValueError(f"Yield volatility 'sigma2' must be strictly positive, got {sigma2}")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"Correlation 'rho' must be in [-1, 1], got {rho}")

        self._s0 = float(s0)
        self._delta0 = float(delta0)
        self._kappa = float(kappa)
        self._alpha = float(alpha)
        self._sigma1 = float(sigma1)
        self._sigma2 = float(sigma2)
        self._rho = float(rho)
        self._rate = float(rate)
        self._lambda_yield = float(lambda_yield)

        if alpha_hat is not None:
            self._alpha_hat = float(alpha_hat)
        else:
            self._alpha_hat = float(self._alpha - self._lambda_yield / self._kappa)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"Schwartz2F(S0={self._s0:.2f}, δ0={self._delta0:.2f}, κ={self._kappa:.2f}, σ1={self._sigma1:.2f}, σ2={self._sigma2:.2f}, ρ={self._rho:.2f})",
            math_note=(
                f"dS_t = (r - δ_t)S_t dt + σ1 S_t dW1, dδ_t = κ(α_hat - δ_t)dt + σ2 dW2, "
                f"α_hat={self._alpha_hat:.4f}"
            ),
            reference="Schwartz (1997) Journal of Finance 52(3)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def s0(self) -> float:
        """Initial spot price :math:`S_0 > 0`."""
        return self._s0

    @property
    def delta0(self) -> float:
        """Initial convenience yield :math:`\\delta_0 \\in \\mathbb{R}`."""
        return self._delta0

    @property
    def kappa(self) -> float:
        """Mean-reversion speed :math:`\\kappa > 0`."""
        return self._kappa

    @property
    def alpha(self) -> float:
        """Physical convenience yield target :math:`\\alpha`."""
        return self._alpha

    @property
    def alpha_hat(self) -> float:
        """Risk-neutral convenience yield target :math:`\\hat{\\alpha} = \\alpha - \\lambda/\\kappa`."""
        return self._alpha_hat

    @property
    def sigma1(self) -> float:
        """Spot volatility :math:`\\sigma_1 > 0`."""
        return self._sigma1

    @property
    def sigma2(self) -> float:
        """Convenience yield volatility :math:`\\sigma_2 > 0`."""
        return self._sigma2

    @property
    def rho(self) -> float:
        """Correlation :math:`\\rho \\in [-1, 1]`."""
        return self._rho

    @property
    def rate(self) -> float:
        """Risk-free rate :math:`r`."""
        return self._rate

    @property
    def lambda_yield(self) -> float:
        """Market price of convenience yield risk :math:`\\lambda`."""
        return self._lambda_yield

    @property
    def stationary_yield_mean(self) -> float:
        """Stationary convenience yield mean :math:`\\hat{\\alpha}`."""
        return self._alpha_hat

    @property
    def stationary_yield_variance(self) -> float:
        r"""Stationary convenience yield variance :math:`\frac{\sigma_2^2}{2\kappa}`."""
        return (self._sigma2**2) / (2.0 * self._kappa)

    @property
    def stationary_yield_std(self) -> float:
        """Stationary convenience yield standard deviation."""
        return math.sqrt(self.stationary_yield_variance)

    # ------------------------------------------------------------------
    # Exactness & Triplet
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason=(
                "SchwartzTwoFactor is a two-factor coupled diffusion with state-dependent convenience "
                "yield drift (r - delta_t)S_t, violating the independent and stationary increments "
                "condition of pure Levy processes. Exact futures prices, option analytics, and "
                "bivariate Gaussian moments are available directly."
            ),
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_mean=True,
            has_finite_variance=True,
            is_subordinator=False,
            notes=[
                f"Schwartz (1997) Two-Factor Commodity Model with spot S0={self._s0}, yield delta0={self._delta0}.",
                f"Parameters: κ={self._kappa}, σ1={self._sigma1}, σ2={self._sigma2}, ρ={self._rho}.",
            ],
        )

    # ------------------------------------------------------------------
    # Convenience Yield Dynamics
    # ------------------------------------------------------------------

    def yield_mean(self, t: float = 1.0) -> float:
        r"""
        Exact conditional expectation :math:`\mathbb{E}[\delta_t \mid \delta_0]`.

        .. math::

            \mathbb{E}[\delta_t \mid \delta_0] = \delta_0 e^{-\kappa t} + \hat{\alpha}(1 - e^{-\kappa t})
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        exp_kt = math.exp(-self._kappa * t)
        return float(self._delta0 * exp_kt + self._alpha_hat * (1.0 - exp_kt))

    def yield_variance(self, t: float = 1.0) -> float:
        r"""
        Exact conditional variance :math:`\mathrm{Var}(\delta_t \mid \delta_0)`.

        .. math::

            \mathrm{Var}(\delta_t \mid \delta_0) = \frac{\sigma_2^2}{2\kappa}(1 - e^{-2\kappa t})
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        if t == 0.0:
            return 0.0
        return float(
            ((self._sigma2**2) / (2.0 * self._kappa)) * (1.0 - math.exp(-2.0 * self._kappa * t))
        )

    def yield_std(self, t: float = 1.0) -> float:
        """Exact conditional standard deviation of convenience yield."""
        return math.sqrt(self.yield_variance(t))

    # ------------------------------------------------------------------
    # Log-Spot Dynamics & Moments
    # ------------------------------------------------------------------

    def log_spot_mean(self, t: float = 1.0) -> float:
        r"""
        Exact expectation :math:`\mathbb{E}[\ln S_t \mid S_0, \delta_0]`.

        .. math::

            \mathbb{E}[\ln S_t] = \ln S_0 + (r - \frac{1}{2}\sigma_1^2 - \hat{\alpha})t - (\delta_0 - \hat{\alpha})\frac{1 - e^{-\kappa t}}{\kappa}
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        if t == 0.0:
            return math.log(self._s0)

        exp_kt = math.exp(-self._kappa * t)
        factor = (1.0 - exp_kt) / self._kappa
        drift_term = (self._rate - 0.5 * (self._sigma1**2) - self._alpha_hat) * t
        yield_term = (self._delta0 - self._alpha_hat) * factor
        return float(math.log(self._s0) + drift_term - yield_term)

    def log_spot_variance(self, t: float = 1.0) -> float:
        r"""
        Exact variance :math:`\mathrm{Var}(\ln S_t \mid S_0, \delta_0)`.

        .. math::

            \mathrm{Var}(\ln S_t) = \sigma_1^2 t + \frac{\sigma_2^2}{\kappa^2}\left(t - 2\frac{1 - e^{-\kappa t}}{\kappa} + \frac{1 - e^{-2\kappa t}}{2\kappa}\right)
                                    - 2\frac{\rho \sigma_1 \sigma_2}{\kappa}\left(t - \frac{1 - e^{-\kappa t}}{\kappa}\right)
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        if t == 0.0:
            return 0.0

        k = self._kappa
        s1 = self._sigma1
        s2 = self._sigma2
        rho = self._rho
        exp_kt = math.exp(-k * t)
        exp_2kt = math.exp(-2.0 * k * t)

        term1 = (s1**2) * t
        term2 = ((s2**2) / (k**2)) * (t - 2.0 * (1.0 - exp_kt) / k + (1.0 - exp_2kt) / (2.0 * k))
        term3 = -2.0 * (rho * s1 * s2 / k) * (t - (1.0 - exp_kt) / k)
        return float(max(0.0, term1 + term2 + term3))

    def log_spot_std(self, t: float = 1.0) -> float:
        """Exact standard deviation of log-spot price."""
        return math.sqrt(self.log_spot_variance(t))

    def covariance_log_spot_yield(self, t: float = 1.0) -> float:
        r"""
        Exact covariance :math:`\mathrm{Cov}(\ln S_t, \delta_t \mid S_0, \delta_0)`.

        .. math::

            \mathrm{Cov}(\ln S_t, \delta_t) = \frac{\rho \sigma_1 \sigma_2}{\kappa}(1 - e^{-\kappa t}) - \frac{\sigma_2^2}{2\kappa^2}(1 - e^{-\kappa t})^2
        """
        if t <= 0.0:
            return 0.0
        k = self._kappa
        exp_kt = math.exp(-k * t)
        one_minus_exp = 1.0 - exp_kt
        term1 = (self._rho * self._sigma1 * self._sigma2 / k) * one_minus_exp
        term2 = ((self._sigma2**2) / (2.0 * (k**2))) * (one_minus_exp**2)
        return float(term1 - term2)

    # ------------------------------------------------------------------
    # Futures Pricing (Schwartz 1997 Proposition 1, eq. 12)
    # ------------------------------------------------------------------

    def _futures_a(self, tau: float) -> float:
        r"""Deterministic function :math:`A(\tau)` in the futures pricing formula."""
        if tau <= 0.0:
            return 0.0
        k = self._kappa
        r = self._rate
        ahat = self._alpha_hat
        s1 = self._sigma1
        s2 = self._sigma2
        rho = self._rho

        exp_kt = math.exp(-k * tau)
        exp_2kt = math.exp(-2.0 * k * tau)

        term1 = (r - ahat + 0.5 * (s2**2) / (k**2) - (s1 * s2 * rho) / k) * tau
        term2 = 0.25 * (s2**2) * (1.0 - exp_2kt) / (k**3)
        term3 = (k * ahat + s1 * s2 * rho - (s2**2) / k) * (1.0 - exp_kt) / (k**2)
        return float(term1 + term2 + term3)

    def futures_price(
        self,
        tau: Union[float, np.ndarray],
        s: Optional[Union[float, np.ndarray]] = None,
        delta: Optional[Union[float, np.ndarray]] = None,
    ) -> Union[float, np.ndarray]:
        r"""
        Exact closed-form futures price :math:`F(S_t, \delta_t, \tau)` with time-to-maturity :math:`\tau = T - t`.

        .. math::

            F(S_t, \delta_t, \tau) = S_t \exp\left( -\delta_t \frac{1 - e^{-\kappa \tau}}{\kappa} + A(\tau) \right)
        """
        tau_arr = np.asarray(tau, dtype=float)
        s_val = self._s0 if s is None else s
        s_arr = np.asarray(s_val, dtype=float)
        d_val = self._delta0 if delta is None else delta
        d_arr = np.asarray(d_val, dtype=float)

        def _single_fwd(t_val: float, spot_val: float, yield_val: float) -> float:
            if t_val <= 0.0:
                return float(spot_val)
            a_val = self._futures_a(t_val)
            factor = (1.0 - math.exp(-self._kappa * t_val)) / self._kappa
            return float(spot_val * math.exp(-yield_val * factor + a_val))

        vec_func = np.vectorize(_single_fwd)
        res = vec_func(tau_arr, s_arr, d_arr)
        if tau_arr.ndim == 0 and s_arr.ndim == 0 and d_arr.ndim == 0:
            return float(res)
        return res

    def futures_curve(
        self,
        maturities: Union[float, np.ndarray],
        s: Optional[Union[float, np.ndarray]] = None,
        delta: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        """Convenience method returning futures prices across an array of maturities."""
        return np.asarray(self.futures_price(tau=maturities, s=s, delta=delta))

    # ------------------------------------------------------------------
    # Option Pricing on Spot (Black-76 Exact Closed Form)
    # ------------------------------------------------------------------

    def call_price(self, strike: float, time_to_maturity: float = 1.0) -> float:
        r"""
        Exact closed-form European call option price on commodity spot :math:`S_T`.

        Derived from the exact lognormal distribution of :math:`S_T` under :math:`\mathbb{Q}`:

        .. math::

            C = e^{-r T} [F(T) \mathcal{N}(d_1) - K \mathcal{N}(d_2)]
        """
        if strike <= 0.0:
            raise ValueError(f"Strike must be strictly positive, got {strike}")
        if time_to_maturity <= 0.0:
            return float(max(0.0, self._s0 - strike))

        tau = time_to_maturity
        f = float(self.futures_price(tau=tau))
        var_x = self.log_spot_variance(t=tau)
        total_std = math.sqrt(var_x)

        d1 = (math.log(f / strike) + 0.5 * var_x) / total_std
        d2 = d1 - total_std
        discount = math.exp(-self._rate * tau)
        return float(discount * (f * stats.norm.cdf(d1) - strike * stats.norm.cdf(d2)))

    def put_price(self, strike: float, time_to_maturity: float = 1.0) -> float:
        r"""
        Exact closed-form European put option price on commodity spot :math:`S_T`.

        .. math::

            P = e^{-r T} [K \mathcal{N}(-d_2) - F(T) \mathcal{N}(-d_1)]
        """
        if strike <= 0.0:
            raise ValueError(f"Strike must be strictly positive, got {strike}")
        if time_to_maturity <= 0.0:
            return float(max(0.0, strike - self._s0))

        tau = time_to_maturity
        f = float(self.futures_price(tau=tau))
        var_x = self.log_spot_variance(t=tau)
        total_std = math.sqrt(var_x)

        d1 = (math.log(f / strike) + 0.5 * var_x) / total_std
        d2 = d1 - total_std
        discount = math.exp(-self._rate * tau)
        return float(discount * (strike * stats.norm.cdf(-d2) - f * stats.norm.cdf(-d1)))

    # ------------------------------------------------------------------
    # Characteristic Function
    # ------------------------------------------------------------------

    def char_func(self, u: Union[float, complex, np.ndarray], t: float = 1.0) -> np.ndarray:
        r"""
        Exact characteristic function of log-spot price :math:`\phi(u; t) = \mathbb{E}[e^{i u \ln S_t}]`.

        .. math::

            \phi(u; t) = \exp\left( i u \mathbb{E}[\ln S_t] - \frac{1}{2} u^2 \mathrm{Var}(\ln S_t) \right)
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        u_arr = np.asarray(u, dtype=complex)
        mean_x = self.log_spot_mean(t)
        var_x = self.log_spot_variance(t)
        exponent = 1j * u_arr * mean_x - 0.5 * (u_arr**2) * var_x
        res = np.exp(exponent)
        return res if res.ndim > 0 else res.item()

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
        *,
        T: Optional[float] = None,  # noqa: N803
    ) -> np.ndarray:
        """Simulate sample paths of commodity spot price :math:`S_t`."""
        from spxa.sim.exact import sample_schwartz_two_factor

        horizon = time_horizon if time_horizon is not None else T
        return sample_schwartz_two_factor(
            s0=self._s0,
            delta0=self._delta0,
            kappa=self._kappa,
            alpha_hat=self._alpha_hat,
            sigma1=self._sigma1,
            sigma2=self._sigma2,
            rho=self._rho,
            rate=self._rate,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=horizon,
            rng=rng,
            return_yield=False,
        )

    def simulate_joint(
        self,
        n_steps: int = 100,
        dt: Optional[float] = None,
        time_horizon: Optional[float] = None,
        n_paths: int = 1,
        rng: Optional[np.random.Generator] = None,
        *,
        T: Optional[float] = None,  # noqa: N803
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate joint sample paths of spot prices :math:`S_t` and convenience yields :math:`\\delta_t`."""
        from spxa.sim.exact import sample_schwartz_two_factor

        horizon = time_horizon if time_horizon is not None else T
        return sample_schwartz_two_factor(
            s0=self._s0,
            delta0=self._delta0,
            kappa=self._kappa,
            alpha_hat=self._alpha_hat,
            sigma1=self._sigma1,
            sigma2=self._sigma2,
            rho=self._rho,
            rate=self._rate,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=horizon,
            rng=rng,
            return_yield=True,
        )

    def __repr__(self) -> str:
        return (
            f"SchwartzTwoFactor(s0={self._s0}, delta0={self._delta0}, kappa={self._kappa}, "
            f"alpha_hat={self._alpha_hat}, sigma1={self._sigma1}, sigma2={self._sigma2}, "
            f"rho={self._rho}, rate={self._rate})"
        )


# Aliases
Schwartz97 = SchwartzTwoFactor
Schwartz2F = SchwartzTwoFactor
SchwartzTwoFactorModel = SchwartzTwoFactor
