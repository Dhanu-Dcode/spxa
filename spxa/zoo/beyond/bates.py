"""
Bates (1996) stochastic volatility with jumps model.

The Bates model augments the Heston (1993) stochastic volatility model with
log-normal Merton (1976) jumps in the underlying asset price:

    dS_t = (μ - q - λ k̄) S_t dt + √V_t S_t dW_t^S + S_{t-} (e^Y - 1) dN_t
    dV_t = κ(θ - V_t) dt + σ_v √V_t dW_t^V

with correlation:
    E[dW_t^S dW_t^V] = ρ dt

where:
    - N_t is a Poisson process with jump arrival rate λ ≥ 0.
    - Y ~ N(μ_J, σ_J^2) is the normal jump size in log-price.
    - k̄ = E[e^Y - 1] = exp(μ_J + ½ σ_J^2) - 1 is the expected relative jump.
    - -λ k̄ is the Merton drift compensator ensuring the discounted asset is a
      martingale under the risk-neutral measure.

Under log-price coordinates X_t = ln(S_t / S_0) with initial value X_0 = 0:
    dX_t = (μ - q - λ k̄ - ½ V_t) dt + √V_t dW_t^S + Y dN_t
    dV_t = κ(θ - V_t) dt + σ_v √V_t dW_t^V

Since the Poisson jumps are independent of the Brownian motions, the characteristic
function of X_t factors into the product of the Heston characteristic function
(evaluated using the continuous Albrecher et al. 2007 formulation) and the
compensated compound Poisson characteristic function:

    φ_{Bates}(u, t) = φ_{Heston}(u, t) · exp(λ t (exp(i u μ_J - ½ σ_J^2 u^2) - 1) - i u λ k̄ t)

References
----------
Bates, D. S. (1996). Jumps and stochastic volatility: Exchange rate processes
implicit in Deutsche Mark options. *The Review of Financial Studies*, 9(1), 69–107.

Heston, S. L. (1993). A closed-form solution for options with stochastic
volatility with applications to bond and currency options.
*The Review of Financial Studies*, 6(2), 327–343.

Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007).
The little Heston trap. *Wilmott Magazine*, 83–92.

Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous.
*Journal of Financial Economics*, 3(1–2), 125–144.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.sim.exact import sample_bates
from spxa.story.narrator import CompositionNode, NodeKind


class BatesProcess(Process):
    r"""
    Bates (1996) stochastic volatility with jumps model.

    Couples the Heston (1993) CIR variance diffusion with Merton (1976)
    log-normal price jumps:

    .. math::

        dX_t = \left(\mu - \lambda \bar{k} - \frac{1}{2} V_t\right) dt + \sqrt{V_t} \, dW_t^S + Y \, dN_t, \quad X_0 = 0 \\
        dV_t = \kappa (\theta - V_t) dt + \sigma_v \sqrt{V_t} \, dW_t^V, \quad V_0 = v_0

    with correlation :math:`\mathbb{E}[dW_t^S dW_t^V] = \rho \, dt`, Poisson intensity :math:`\lambda`,
    and jump size :math:`Y \sim \mathcal{N}(\mu_J, \sigma_J^2)`.

    Parameters
    ----------
    v0 :
        Initial variance :math:`V_0 = v_0 > 0`. Default 0.04 (:math:`\sqrt{v_0} = 20\%`).
    kappa :
        Rate of mean reversion for variance :math:`\kappa > 0`. Default 2.0.
    theta :
        Long-term variance level :math:`\theta > 0`. Default 0.04.
    sigma_v :
        Volatility of variance :math:`\sigma_v > 0`. May also be supplied as ``sigma`` or ``xi``.
    rho :
        Correlation between asset and variance Brownian motions :math:`\rho \in [-1, 1]`. Default -0.5.
    rate :
        Jump arrival intensity :math:`\lambda \ge 0`. Default 1.0. May also be supplied as ``lambd`` or ``jump_rate``.
    jump_mean :
        Mean jump size :math:`\mu_J \in \mathbb{R}`. Default -0.05. May also be supplied as ``mu_j``.
    jump_std :
        Jump size standard deviation :math:`\sigma_J > 0`. Default 0.10. May also be supplied as ``sigma_j``.
    mu :
        Continuous drift of asset log-return :math:`\mu \in \mathbb{R}` (e.g. :math:`r - q`). Default 0.0.
    s0 :
        Initial asset spot price :math:`S_0 > 0`. Default 1.0.
    """

    def __init__(
        self,
        v0: float = 0.04,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma_v: Optional[float] = None,
        rho: float = -0.5,
        rate: float = 1.0,
        jump_mean: float = -0.05,
        jump_std: float = 0.10,
        mu: float = 0.0,
        s0: float = 1.0,
        **kwargs,
    ) -> None:
        if sigma_v is None:
            sigma_v = kwargs.get("sigma", kwargs.get("xi", 0.3))

        rate_val = kwargs.get("lambd", kwargs.get("jump_rate", kwargs.get("lambda_", rate)))
        jump_mean_val = kwargs.get("mu_j", kwargs.get("jump_mu", jump_mean))
        jump_std_val = kwargs.get("sigma_j", kwargs.get("jump_sigma", jump_std))

        v0 = float(v0)
        kappa = float(kappa)
        theta = float(theta)
        sigma_v = float(sigma_v)
        rho = float(rho)
        rate_val = float(rate_val)
        jump_mean_val = float(jump_mean_val)
        jump_std_val = float(jump_std_val)
        mu = float(mu)
        s0 = float(s0)

        if v0 <= 0.0:
            raise ValueError(f"Initial variance v0 must be strictly positive, got {v0}")
        if kappa <= 0.0:
            raise ValueError(f"Mean reversion rate kappa must be strictly positive, got {kappa}")
        if theta <= 0.0:
            raise ValueError(f"Long-term variance theta must be strictly positive, got {theta}")
        if sigma_v <= 0.0:
            raise ValueError(f"Volatility of variance sigma_v must be strictly positive, got {sigma_v}")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"Correlation rho must lie in [-1, 1], got {rho}")
        if rate_val < 0.0:
            raise ValueError(f"Jump arrival rate lambda must be non-negative, got {rate_val}")
        if jump_std_val <= 0.0:
            raise ValueError(f"Jump standard deviation sigma_j must be strictly positive, got {jump_std_val}")
        if s0 <= 0.0:
            raise ValueError(f"Spot price s0 must be strictly positive, got {s0}")

        self._v0 = v0
        self._kappa = kappa
        self._theta = theta
        self._sigma_v = sigma_v
        self._rho = rho
        self._rate = rate_val
        self._jump_mean = jump_mean_val
        self._jump_std = jump_std_val
        self._mu = mu
        self._s0 = s0

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=(
                f"BatesProcess(v0={self._v0}, kappa={self._kappa}, theta={self._theta}, "
                f"sigma_v={self._sigma_v}, rho={self._rho}, rate={self._rate}, "
                f"jump_mean={self._jump_mean}, jump_std={self._jump_std}, mu={self._mu}, s0={self._s0})"
            ),
            math_note=(
                f"dX_t = (μ - λk̄ - ½V_t)dt + √V_t dW_t^S + Y dN_t; "
                f"dV_t = κ(θ - V_t)dt + σ_v √V_t dW_t^V; ρ={self._rho}, λ={self._rate}"
            ),
            reference="Bates, D. S. (1996), Review of Financial Studies 9(1), 69–107",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def __repr__(self) -> str:
        return (
            f"BatesProcess(v0={self._v0}, kappa={self._kappa}, theta={self._theta}, "
            f"sigma_v={self._sigma_v}, rho={self._rho}, rate={self._rate}, "
            f"jump_mean={self._jump_mean}, jump_std={self._jump_std}, mu={self._mu}, s0={self._s0})"
        )

    # ------------------------------------------------------------------
    # Parameters & Properties
    # ------------------------------------------------------------------

    @property
    def v0(self) -> float:
        """Initial instantaneous variance :math:`V_0 = v_0 > 0`."""
        return self._v0

    @property
    def kappa(self) -> float:
        """Rate of mean reversion :math:`\\kappa > 0`."""
        return self._kappa

    @property
    def theta(self) -> float:
        """Long-term variance level :math:`\\theta > 0`."""
        return self._theta

    @property
    def sigma_v(self) -> float:
        """Volatility of variance :math:`\\sigma_v > 0`."""
        return self._sigma_v

    @property
    def rho(self) -> float:
        """Correlation between Brownian motions :math:`\\rho \\in [-1, 1]`."""
        return self._rho

    @property
    def rate(self) -> float:
        """Jump arrival intensity :math:`\\lambda \\ge 0`."""
        return self._rate

    @property
    def jump_mean(self) -> float:
        """Mean log-jump size :math:`\\mu_J \\in \\mathbb{R}`."""
        return self._jump_mean

    @property
    def jump_std(self) -> float:
        """Jump size standard deviation :math:`\\sigma_J > 0`."""
        return self._jump_std

    @property
    def mu(self) -> float:
        """Continuous drift rate :math:`\\mu`."""
        return self._mu

    @property
    def s0(self) -> float:
        """Spot price :math:`S_0 > 0`."""
        return self._s0

    @property
    def expected_relative_jump(self) -> float:
        r"""
        Expected relative price jump :math:`\bar{k} = \mathbb{E}[e^Y - 1] = \exp(\mu_J + \frac{1}{2}\sigma_J^2) - 1`.
        """
        return float(math.exp(self._jump_mean + 0.5 * self._jump_std ** 2) - 1.0)

    @property
    def drift_compensator(self) -> float:
        r"""
        Martingale drift compensator :math:`-\lambda \bar{k}`.
        """
        return float(-self._rate * self.expected_relative_jump)

    @property
    def feller_ratio(self) -> float:
        r"""The Feller condition ratio :math:`\frac{2\kappa\theta}{\sigma_v^2}`."""
        return (2.0 * self._kappa * self._theta) / (self._sigma_v ** 2)

    @property
    def is_feller_satisfied(self) -> bool:
        r"""True if :math:`2\kappa\theta > \sigma_v^2` (variance strictly positive)."""
        return self.feller_ratio > 1.0

    # ------------------------------------------------------------------
    # Exactness & Triplet
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "BatesProcess has ExactnessLevel.MOMENT_PROPAGATION and does not possess "
            "a standard Levy triplet: the instantaneous variance V_t is a continuous "
            "state-dependent CIR diffusion, violating the independent and stationary increments "
            "requirement of pure Levy processes. Distributional analytics, characteristic "
            "functions, cumulants, and Fourier option pricing are available directly."
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
                (
                    f"Bates (1996) stochastic volatility with jumps model: "
                    f"dX_t = (μ - λk̄ - ½V_t)dt + √V_t dW_t^S + Y dN_t; "
                    f"dV_t = κ(θ - V_t)dt + σ_v √V_t dW_t^V; ρ={self._rho}, λ={self._rate}."
                ),
                (
                    f"Parameters: v0={self._v0}, κ={self._kappa}, θ={self._theta}, "
                    f"σ_v={self._sigma_v}, ρ={self._rho}, λ={self._rate}, "
                    f"μ_J={self._jump_mean}, σ_J={self._jump_std}, μ={self._mu}, S0={self._s0}."
                ),
            ],
        )

    def _degradation_reason(self) -> str:
        return (
            "Bates (1996) stochastic volatility with jumps model has state-dependent "
            "stochastic variance V_t and compound Poisson jumps."
        )

    # ------------------------------------------------------------------
    # CIR Variance Process Analytics
    # ------------------------------------------------------------------

    def variance_mean(self, t: float) -> float:
        r"""Expected variance :math:`\mathbb{E}[V_t \mid V_0 = v_0]`."""
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        return self._theta + (self._v0 - self._theta) * np.exp(-self._kappa * t)

    def variance_var(self, t: float) -> float:
        r"""Variance of variance :math:`\mathrm{Var}(V_t \mid V_0 = v_0)`."""
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        exp_kt = np.exp(-self._kappa * t)
        c1 = self._v0 * (self._sigma_v ** 2 / self._kappa) * (exp_kt - exp_kt ** 2)
        c2 = self._theta * (self._sigma_v ** 2 / (2.0 * self._kappa)) * ((1.0 - exp_kt) ** 2)
        return float(c1 + c2)

    @property
    def stationary_variance_mean(self) -> float:
        r"""Stationary mean of variance :math:`\theta`."""
        return self._theta

    @property
    def stationary_variance_var(self) -> float:
        r"""Stationary variance of variance :math:`\frac{\theta \sigma_v^2}{2\kappa}`."""
        return float((self._theta * self._sigma_v ** 2) / (2.0 * self._kappa))

    def expected_integrated_variance(self, t: float) -> float:
        r"""
        Expected integrated variance :math:`\mathbb{E}\left[\int_0^t V_s \, ds\right]`.

        .. math::

            \mathbb{E}\left[\int_0^t V_s \, ds\right] = \theta t + (v_0 - \theta) \frac{1 - e^{-\kappa t}}{\kappa}
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        if t == 0.0:
            return 0.0
        exp_kt = np.exp(-self._kappa * t)
        return float(self._theta * t + (self._v0 - self._theta) * (1.0 - exp_kt) / self._kappa)

    # ------------------------------------------------------------------
    # Characteristic Function (Bates 1996 / Albrecher et al. 2007)
    # ------------------------------------------------------------------

    def char_func(self, u: Union[float, np.ndarray], t: float = 1.0) -> Union[complex, np.ndarray]:
        r"""
        Characteristic function of log-return :math:`X_t = \ln(S_t / S_0)`.

        Evaluates :math:`\phi_{Bates}(u, t) = \mathbb{E}\left[e^{i u X_t}\right] = \phi_{Heston}(u, t) \cdot \phi_{Jump}(u, t)`
        where :math:`\phi_{Heston}` uses the continuous Albrecher et al. (2007) formulation
        and :math:`\phi_{Jump}` incorporates the Merton drift compensator :math:`-\lambda \bar{k}`.

        Parameters
        ----------
        u :
            Real or complex frequency parameter (scalar or numpy array).
        t :
            Evaluation horizon :math:`t > 0`. Default 1.0.

        Returns
        -------
        complex or np.ndarray
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        if t == 0.0:
            if np.isscalar(u):
                return 1.0 + 0.0j
            return np.ones_like(u, dtype=complex)

        u_arr = np.asarray(u)
        is_scalar = u_arr.ndim == 0

        i = 1j
        k = self._kappa
        th = self._theta
        sig = self._sigma_v
        rho = self._rho
        sig2 = sig ** 2

        # 1. Heston component (Albrecher et al. 2007)
        term1 = k - i * rho * sig * u_arr
        d = np.sqrt(term1 ** 2 + sig2 * (u_arr ** 2 + i * u_arr))
        g = (term1 - d) / (term1 + d)

        exp_dt = np.exp(-d * t)
        one_minus_g_exp = 1.0 - g * exp_dt
        one_minus_g = 1.0 - g

        D = ((term1 - d) / sig2) * ((1.0 - exp_dt) / one_minus_g_exp)
        log_term = np.log(one_minus_g_exp / one_minus_g)
        C = (k * th / sig2) * ((term1 - d) * t - 2.0 * log_term)

        cf_heston = np.exp(C + D * self._v0 + i * u_arr * self._mu * t)

        # 2. Merton Jump component with compensator
        if self._rate > 0.0:
            k_bar = self.expected_relative_jump
            jump_exp = np.exp(i * u_arr * self._jump_mean - 0.5 * (self._jump_std ** 2) * (u_arr ** 2)) - 1.0
            cf_jump = np.exp(self._rate * t * jump_exp - i * u_arr * self._rate * k_bar * t)
        else:
            cf_jump = 1.0

        cf_bates = cf_heston * cf_jump

        if is_scalar:
            return complex(cf_bates)
        return cf_bates

    def asset_char_func(self, u: Union[float, np.ndarray], t: float = 1.0) -> Union[complex, np.ndarray]:
        r"""
        Characteristic function of log-price :math:`\ln S_t = \ln S_0 + X_t`.

        .. math::

            \mathbb{E}\left[e^{i u \ln S_t}\right] = e^{i u \ln S_0} \mathbb{E}\left[e^{i u X_t}\right]
        """
        u_arr = np.asarray(u)
        phase = np.exp(1j * u_arr * np.log(self._s0))
        cf_x = self.char_func(u, t=t)
        return phase * cf_x

    # ------------------------------------------------------------------
    # Moments & Cumulants of Log-Return X_t
    # ------------------------------------------------------------------

    def conditional_mean(self, t: float = 1.0) -> float:
        r"""
        Expected log-return :math:`\mathbb{E}[X_t]`:

        .. math::

            \mathbb{E}[X_t] = (\mu - \lambda \bar{k} + \lambda \mu_J) t - \frac{1}{2} \mathbb{E}\left[\int_0^t V_s \, ds\right]
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        k_bar = self.expected_relative_jump
        drift_term = (self._mu - self._rate * k_bar + self._rate * self._jump_mean) * t
        diff_term = 0.5 * self.expected_integrated_variance(t)
        return float(drift_term - diff_term)

    def conditional_cumulants(self, t: float = 1.0, order: int = 4) -> dict[int, float]:
        r"""
        High-precision cumulants :math:`\kappa_1(t), \dots, \kappa_{\mathrm{order}}(t)` of :math:`X_t`.

        Computed by numerical differentiation of :math:`\ln \phi_{Bates}(u, t)` at :math:`u = 0`
        with order-optimised step sizes.
        """
        if order < 1:
            raise ValueError(f"Cumulant order must be at least 1, got {order}")
        if t <= 0.0:
            return {n: 0.0 for n in range(1, order + 1)}

        def log_cf(u_val: float) -> complex:
            return np.log(self.char_func(u_val, t=t))

        kappas: dict[int, float] = {}

        # kappa_1 = (1/i) * d/du log_cf(0) with h = 1e-5
        h1 = 1e-5
        d1 = (log_cf(h1) - log_cf(-h1)) / (2.0 * h1)
        kappas[1] = float(np.real(d1 / 1j))

        if order >= 2:
            # kappa_2 = - d^2/du^2 log_cf(0) with h = 1e-4
            h2 = 1e-4
            d2 = (log_cf(h2) - 2.0 * log_cf(0.0) + log_cf(-h2)) / (h2 ** 2)
            kappas[2] = float(np.real(-d2))

        if order >= 3:
            # kappa_3 = i * d^3/du^3 log_cf(0) with h = 5e-3
            h3 = 5e-3
            d3 = (log_cf(2 * h3) - 2 * log_cf(h3) + 2 * log_cf(-h3) - log_cf(-2 * h3)) / (2.0 * h3 ** 3)
            kappas[3] = float(np.real(1j * d3))

        if order >= 4:
            # kappa_4 = d^4/du^4 log_cf(0) with h = 1.5e-2
            h4 = 1.5e-2
            d4 = (log_cf(2 * h4) - 4 * log_cf(h4) + 6 * log_cf(0.0) - 4 * log_cf(-h4) + log_cf(-2 * h4)) / (h4 ** 4)
            kappas[4] = float(np.real(d4))

        for n in range(5, order + 1):
            kappas[n] = float(np.nan)

        return kappas

    def conditional_variance(self, t: float = 1.0) -> float:
        r"""Variance :math:`\mathrm{Var}(X_t)` of the log-return."""
        return self.conditional_cumulants(t=t, order=2)[2]

    def cumulants(self, order: int) -> dict[int, float]:
        r"""Cumulants of :math:`X_1` (at horizon :math:`t = 1.0`)."""
        return self.conditional_cumulants(t=1.0, order=order)

    # ------------------------------------------------------------------
    # European Option Pricing via Gil-Pelaez Fourier Inversion
    # ------------------------------------------------------------------

    def call_price(
        self,
        K: Union[float, np.ndarray],
        T: float = 1.0,
        r: Optional[float] = None,
        q: float = 0.0,
    ) -> Union[float, np.ndarray]:
        r"""
        European call option price :math:`C(S_0, K, T)` via Gil-Pelaez Fourier inversion.

        .. math::

            C = S_0 e^{-q T} P_1 - K e^{-r T} P_2

        where :math:`P_1, P_2` are risk-neutral probabilities:

        .. math::

            P_j = \frac{1}{2} + \frac{1}{\pi} \int_0^\infty \operatorname{Re}\left( \frac{e^{-i u \ln(K / S_0)} \psi_j(u, T)}{i u} \right) du

        with :math:`\psi_2(u) = \phi_{Bates}(u, T)` and :math:`\psi_1(u) = e^{-\mu T} \phi_{Bates}(u - i, T)`.

        Parameters
        ----------
        K :
            Strike price or array of strikes (:math:`K > 0`).
        T :
            Time to maturity in years (:math:`T > 0`). Default 1.0.
        r :
            Risk-free rate. If None, uses drift :math:`\mu`. Default None.
        q :
            Continuous dividend yield. Default 0.0.

        Returns
        -------
        float or np.ndarray
        """
        if T <= 0.0:
            raise ValueError(f"Maturity T must be strictly positive, got {T}")
        r_val = float(self._mu) if r is None else float(r)
        q_val = float(q)

        def _price_single(strike: float) -> float:
            if strike <= 0.0:
                raise ValueError(f"Strike K must be strictly positive, got {strike}")

            ln_k_s0 = np.log(strike / self._s0)

            def integrand1(u: float) -> float:
                if u < 1e-12:
                    return 0.0
                cf1 = np.exp(-self._mu * T) * self.char_func(u - 1j, t=T)
                val = np.exp(-1j * u * ln_k_s0) * cf1 / (1j * u)
                return float(np.real(val))

            def integrand2(u: float) -> float:
                if u < 1e-12:
                    return 0.0
                cf2 = self.char_func(u, t=T)
                val = np.exp(-1j * u * ln_k_s0) * cf2 / (1j * u)
                return float(np.real(val))

            p1_int, _ = quad(integrand1, 1e-8, 100.0, limit=200)
            p2_int, _ = quad(integrand2, 1e-8, 100.0, limit=200)

            P1 = float(np.clip(0.5 + (1.0 / np.pi) * p1_int, 0.0, 1.0))
            P2 = float(np.clip(0.5 + (1.0 / np.pi) * p2_int, 0.0, 1.0))

            price = self._s0 * np.exp(-q_val * T) * P1 - strike * np.exp(-r_val * T) * P2
            intrinsic = max(0.0, self._s0 * np.exp(-q_val * T) - strike * np.exp(-r_val * T))
            return float(max(price, intrinsic))

        K_arr = np.asarray(K)
        if K_arr.ndim == 0:
            return _price_single(float(K_arr))
        return np.array([_price_single(float(k)) for k in K_arr], dtype=float)

    def put_price(
        self,
        K: Union[float, np.ndarray],
        T: float = 1.0,
        r: Optional[float] = None,
        q: float = 0.0,
    ) -> Union[float, np.ndarray]:
        r"""
        European put option price :math:`P(S_0, K, T)` via Put-Call parity.

        .. math::

            P = C - S_0 e^{-q T} + K e^{-r T}
        """
        r_val = float(self._mu) if r is None else float(r)
        calls = self.call_price(K, T=T, r=r_val, q=q)
        K_arr = np.asarray(K, dtype=float)
        puts = calls - self._s0 * np.exp(-q * T) + K_arr * np.exp(-r_val * T)
        intrinsic = np.maximum(0.0, K_arr * np.exp(-r_val * T) - self._s0 * np.exp(-q * T))
        return np.maximum(puts, intrinsic)

    def implied_volatility(
        self,
        K: Union[float, np.ndarray],
        T: float = 1.0,
        r: Optional[float] = None,
        q: float = 0.0,
    ) -> Union[float, np.ndarray]:
        r"""
        Black-Scholes implied volatility :math:`\sigma_{\mathrm{BS}}(K, T)` inverted from Bates call price.
        """
        r_val = float(self._mu) if r is None else float(r)
        q_val = float(q)

        def _bs_call(s: float, k: float, t: float, rate: float, div: float, vol: float) -> float:
            if vol <= 1e-8:
                return float(max(0.0, s * np.exp(-div * t) - k * np.exp(-rate * t)))
            d1 = (np.log(s / k) + (rate - div + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
            d2 = d1 - vol * np.sqrt(t)
            return float(s * np.exp(-div * t) * norm.cdf(d1) - k * np.exp(-rate * t) * norm.cdf(d2))

        def _iv_single(strike: float) -> float:
            call_mkt = float(self.call_price(strike, T=T, r=r_val, q=q_val))
            intrinsic = float(max(0.0, self._s0 * np.exp(-q_val * T) - strike * np.exp(-r_val * T)))
            upper_bound = float(self._s0 * np.exp(-q_val * T))
            if call_mkt <= intrinsic + 1e-10:
                return 0.0
            if call_mkt >= upper_bound - 1e-10:
                return 5.0

            def obj(v: float) -> float:
                return _bs_call(self._s0, strike, T, r_val, q_val, v) - call_mkt

            try:
                return float(brentq(obj, 1e-5, 5.0, xtol=1e-6))
            except Exception:
                return float(np.nan)

        K_arr = np.asarray(K)
        if K_arr.ndim == 0:
            return _iv_single(float(K_arr))
        return np.array([_iv_single(float(k)) for k in K_arr], dtype=float)

    def implied_vol_smile(
        self,
        strikes: Union[list[float], np.ndarray],
        T: float = 1.0,
        r: Optional[float] = None,
        q: float = 0.0,
    ) -> np.ndarray:
        r"""Compute Black-Scholes implied volatilities across an array of strikes."""
        return self.implied_volatility(strikes, T=T, r=r, q=q)

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
        method: str = "full_truncation",
    ) -> np.ndarray:
        r"""
        Simulate sample paths of log-price :math:`X_t = \ln(S_t / S_0)`.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_paths, n_steps + 1)`` with :math:`X_0 = 0`.
        """
        return sample_bates(
            v0=self._v0,
            kappa=self._kappa,
            theta=self._theta,
            sigma_v=self._sigma_v,
            rho=self._rho,
            rate=self._rate,
            jump_mean=self._jump_mean,
            jump_std=self._jump_std,
            mu=self._mu,
            s0=self._s0,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=time_horizon,
            rng=rng,
            method=method,
            return_variance=False,
        )

    def simulate_joint(
        self,
        n_steps: int = 100,
        dt: Optional[float] = None,
        time_horizon: Optional[float] = None,
        n_paths: int = 1,
        rng: Optional[np.random.Generator] = None,
        method: str = "full_truncation",
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Simulate coupled paths of log-price :math:`X_t` and instantaneous variance :math:`V_t`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Paths of :math:`X_t` and :math:`V_t`, each of shape ``(n_paths, n_steps + 1)``.
        """
        return sample_bates(
            v0=self._v0,
            kappa=self._kappa,
            theta=self._theta,
            sigma_v=self._sigma_v,
            rho=self._rho,
            rate=self._rate,
            jump_mean=self._jump_mean,
            jump_std=self._jump_std,
            mu=self._mu,
            s0=self._s0,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=time_horizon,
            rng=rng,
            method=method,
            return_variance=True,
        )


# Aliases
BatesModel = BatesProcess
Bates96 = BatesProcess
SVJ = BatesProcess
