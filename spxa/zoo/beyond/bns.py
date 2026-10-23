"""
Barndorff-Nielsen & Shephard (BNS) Stochastic Volatility Model.

Canonical non-Gaussian Ornstein-Uhlenbeck stochastic volatility model introduced
by Ole E. Barndorff-Nielsen and Neil Shephard (2001).

Model Dynamics:
    dS_t = (r - q + (β + 0.5) σ_t² - λ k_Z(ρ)) S_t dt + σ_t S_t dW_t + (e^ρ - 1) S_t- dZ_{λ t}
    X_t = ln(S_t / S_0):
    dX_t = (μ + β σ_t²) dt + σ_t dW_t + ρ dZ_{λ t}
    dσ_t² = -λ σ_t² dt + dZ_{λ t},   σ_0² > 0

where:
    - W_t: standard 1D Brownian motion
    - Z_t: Background Driving Lévy Process (BDLP), a compound Poisson process with rate α
      and jump sizes Y ~ Exp(β_bdlp), guaranteeing a stationary Gamma(α, β_bdlp) distribution
      for instantaneous variance σ_t²
    - λ > 0: mean-reversion rate of variance
    - ρ ≤ 0: leverage parameter (negative return jump with positive variance shock)
    - β: drift risk-premium parameter (typically -0.5 for risk-neutral pricing)
    - μ = (r - q) - λ k_Z(ρ): risk-neutral drift ensuring discounted stock price is a martingale

Features:
    - Continuous-time OU variance driven by positive Lévy jumps (no negative variance).
    - Exact Gamma stationary distribution for variance: E[σ_∞²] = α / β_bdlp.
    - Exact integrated variance expectation and variance moments.
    - Exact continuous characteristic function via Fourier-Laplace transform.
    - European Call and Put option pricing via Gil-Pelaez Fourier inversion with exact Put-Call parity.
    - Implied volatility smile computation.
    - Vectorized coupled simulation of stock price and variance paths.

References
----------
Barndorff-Nielsen, O. E., & Shephard, N. (2001). Non-Gaussian Ornstein–Uhlenbeck-based
    models and some of their applications in financial economics. Journal of the Royal
    Statistical Society: Series B, 63(2), 167–241.
Nicolato, E., & Venardos, E. (2003). Option pricing in stochastic volatility models of
    the Ornstein–Uhlenbeck type. Mathematical Finance, 13(4), 445–466.
Schoutens, W. (2003). Lévy Processes in Finance: Pricing Financial Derivatives.
    John Wiley & Sons, Chapter 7.2.
"""

from __future__ import annotations

import math
from typing import Optional, Union, Tuple, Dict, Any

import numpy as np
from scipy import integrate, optimize, stats  # type: ignore[import-untyped]

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class BNSModel(Process):
    r"""
    Barndorff-Nielsen & Shephard (BNS) stochastic volatility model (2001).

    .. math::

        dX_t = (\mu + \beta \sigma_t^2) dt + \sigma_t dW_t + \rho dZ_{\lambda t}, \quad X_0 = 0 \\
        d\sigma_t^2 = -\lambda \sigma_t^2 dt + dZ_{\lambda t}, \quad \sigma_0^2 > 0

    where :math:`Z_t` is a compound Poisson BDLP with rate :math:`\alpha` and jump size :math:`\mathrm{Exp}(\beta_{bdlp})`.

    Parameters
    ----------
    lambda_param : float
        Variance mean-reversion rate :math:`\lambda > 0`. Default 1.5.
    alpha : float
        BDLP Gamma shape parameter :math:`\alpha > 0`. Default 2.0.
    beta_bdlp : float
        BDLP Gamma rate parameter :math:`\beta_{bdlp} > 0`. Default 25.0.
    rho : float
        Leverage coefficient :math:`\rho \le 0`. Default -0.50.
    beta : float
        Drift risk parameter :math:`\beta`. Default -0.50.
    sigma0_sq : float
        Initial variance :math:`\sigma_0^2 > 0`. Default 0.04.
    s0 : float
        Initial asset price :math:`S_0 > 0`. Default 100.0.
    rate : float
        Risk-free interest rate :math:`r`. Default 0.05.
    dividend : float
        Continuous dividend yield :math:`q`. Default 0.0.
    """

    def __init__(
        self,
        lambda_param: float = 1.5,
        alpha: float = 2.0,
        beta_bdlp: float = 25.0,
        rho: float = -0.50,
        beta: float = -0.50,
        sigma0_sq: float = 0.04,
        s0: float = 100.0,
        rate: float = 0.05,
        dividend: float = 0.0,
        lam: Optional[float] = None,
        a: Optional[float] = None,
        b: Optional[float] = None,
        v0: Optional[float] = None,
    ) -> None:
        l_val = lam if lam is not None else lambda_param
        if l_val <= 0.0:
            raise ValueError(f"Mean-reversion rate 'lambda_param' must be strictly positive, got {l_val}")

        a_val = a if a is not None else alpha
        if a_val <= 0.0:
            raise ValueError(f"BDLP parameter 'alpha' must be strictly positive, got {a_val}")

        b_val = b if b is not None else beta_bdlp
        if b_val <= 0.0:
            raise ValueError(f"BDLP parameter 'beta_bdlp' must be strictly positive, got {b_val}")

        v0_val = v0 if v0 is not None else sigma0_sq
        if v0_val <= 0.0:
            raise ValueError(f"Initial variance 'sigma0_sq' must be strictly positive, got {v0_val}")

        if s0 <= 0.0:
            raise ValueError(f"Initial asset price 's0' must be strictly positive, got {s0}")

        if rho > 0.0:
            raise ValueError(f"Leverage parameter 'rho' must be non-positive (rho <= 0), got {rho}")

        # Check b - rho > 0 for martingale drift
        if b_val - rho <= 0.0:
            raise ValueError(f"Integrability condition b - rho > 0 violated: {b_val} - {rho} <= 0")

        self._lambda = float(l_val)
        self._alpha = float(a_val)
        self._beta_bdlp = float(b_val)
        self._rho = float(rho)
        self._beta = float(beta)
        self._sigma0_sq = float(v0_val)
        self._s0 = float(s0)
        self._rate = float(rate)
        self._dividend = float(dividend)

        # Risk-neutral drift: mu = (r - q) - lambda * k_Z(rho)
        # where k_Z(theta) = alpha * theta / (beta_bdlp - theta)
        k_rho = (self._alpha * self._rho) / (self._beta_bdlp - self._rho)
        self._mu = (self._rate - self._dividend) - self._lambda * k_rho

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"BNS(λ={self._lambda:.2f}, α={self._alpha:.2f}, β_bdlp={self._beta_bdlp:.2f}, ρ={self._rho:.2f})",
            math_note=(
                f"dX_t = (μ + β σ_t²)dt + σ_t dW_t + ρ dZ_{{λt}}, dσ_t² = -λ σ_t² dt + dZ_{{λt}}, "
                f"E[σ_∞²]={self._alpha/self._beta_bdlp:.4f}"
            ),
            reference="Barndorff-Nielsen & Shephard (2001) JRSS-B 63(2)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lambda_param(self) -> float:
        """Variance mean-reversion rate :math:`\\lambda > 0`."""
        return self._lambda

    @property
    def lam(self) -> float:
        """Alias for lambda_param."""
        return self._lambda

    @property
    def alpha(self) -> float:
        """BDLP Gamma shape parameter :math:`\\alpha > 0`."""
        return self._alpha

    @property
    def a(self) -> float:
        """Alias for alpha."""
        return self._alpha

    @property
    def beta_bdlp(self) -> float:
        """BDLP Gamma rate parameter :math:`\\beta_{bdlp} > 0`."""
        return self._beta_bdlp

    @property
    def b(self) -> float:
        """Alias for beta_bdlp."""
        return self._beta_bdlp

    @property
    def rho(self) -> float:
        """Leverage parameter :math:`\\rho \\le 0`."""
        return self._rho

    @property
    def beta(self) -> float:
        """Drift risk parameter :math:`\\beta`."""
        return self._beta

    @property
    def sigma0_sq(self) -> float:
        """Initial variance :math:`\\sigma_0^2 > 0`."""
        return self._sigma0_sq

    @property
    def v0(self) -> float:
        """Alias for sigma0_sq."""
        return self._sigma0_sq

    @property
    def s0(self) -> float:
        """Initial asset price :math:`S_0 > 0`."""
        return self._s0

    @property
    def rate(self) -> float:
        """Risk-free interest rate :math:`r`."""
        return self._rate

    @property
    def dividend(self) -> float:
        """Continuous dividend yield :math:`q`."""
        return self._dividend

    @property
    def mu(self) -> float:
        """Risk-neutral drift :math:`\\mu`."""
        return self._mu

    @property
    def stationary_variance_mean(self) -> float:
        r"""Stationary variance mean :math:`\mathbb{E}[\sigma_\infty^2] = \frac{\alpha}{\beta_{bdlp}}`."""
        return self._alpha / self._beta_bdlp

    @property
    def stationary_variance_variance(self) -> float:
        r"""Stationary variance of variance :math:`\mathrm{Var}(\sigma_\infty^2) = \frac{\alpha}{\beta_{bdlp}^2}`."""
        return self._alpha / (self._beta_bdlp ** 2)

    @property
    def stationary_variance_std(self) -> float:
        """Stationary variance standard deviation."""
        return math.sqrt(self.stationary_variance_variance)

    # ------------------------------------------------------------------
    # Exactness & Triplet
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason=(
                "BNSModel is a stochastic volatility jump-diffusion with non-Gaussian Ornstein-Uhlenbeck "
                "stochastic variance σ_t², violating the independent increments condition of pure Levy "
                "processes. Exact variance moments, continuous characteristic function, option pricing, "
                "and path simulation are available directly."
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
                f"Barndorff-Nielsen & Shephard (2001) model with λ={self._lambda}, α={self._alpha}, β_bdlp={self._beta_bdlp}.",
                f"Stationary variance mean: {self.stationary_variance_mean:.4f}, leverage ρ={self._rho}.",
            ],
        )

    # ------------------------------------------------------------------
    # Variance Moments
    # ------------------------------------------------------------------

    def variance_mean(self, t: float = 1.0) -> float:
        r"""
        Exact conditional expectation of variance :math:`\mathbb{E}[\sigma_t^2 \mid \sigma_0^2]`.

        .. math::

            \mathbb{E}[\sigma_t^2] = \sigma_0^2 e^{-\lambda t} + \frac{\alpha}{\beta_{bdlp}}(1 - e^{-\lambda t})
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        exp_lt = math.exp(-self._lambda * t)
        return float(self._sigma0_sq * exp_lt + self.stationary_variance_mean * (1.0 - exp_lt))

    def variance_variance(self, t: float = 1.0) -> float:
        r"""
        Exact conditional variance of variance :math:`\mathrm{Var}(\sigma_t^2 \mid \sigma_0^2)`.

        .. math::

            \mathrm{Var}(\sigma_t^2) = \frac{\alpha}{\beta_{bdlp}^2}(1 - e^{-2\lambda t})
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        if t == 0.0:
            return 0.0
        return float(self.stationary_variance_variance * (1.0 - math.exp(-2.0 * self._lambda * t)))

    def integrated_variance_mean(self, t: float = 1.0) -> float:
        r"""
        Exact expectation of integrated variance :math:`\mathbb{E}\left[\int_0^t \sigma_s^2 ds \mid \sigma_0^2\right]`.

        .. math::

            \mathbb{E}\left[\int_0^t \sigma_s^2 ds\right] = \sigma_0^2 \frac{1 - e^{-\lambda t}}{\lambda}
            + \frac{\alpha}{\beta_{bdlp}}\left(t - \frac{1 - e^{-\lambda t}}{\lambda}\right)
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        if t == 0.0:
            return 0.0
        exp_lt = math.exp(-self._lambda * t)
        factor = (1.0 - exp_lt) / self._lambda
        term1 = self._sigma0_sq * factor
        term2 = self.stationary_variance_mean * (t - factor)
        return float(term1 + term2)

    # ------------------------------------------------------------------
    # Characteristic Function
    # ------------------------------------------------------------------

    def char_func(self, u: Union[float, complex, np.ndarray], t: float = 1.0) -> np.ndarray:
        r"""
        Exact characteristic function :math:`\phi(u; t) = \mathbb{E}[e^{i u \ln(S_t / S_0)}]`.

        .. math::

            \phi(u; t) = \exp\left( i u \mu t + f(t) \sigma_0^2 + \lambda \int_0^t \kappa_Z\left( \frac{k(u)}{\lambda}(1 - e^{-\lambda s}) + i u \rho \right) ds \right)
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        u_arr = np.asarray(u, dtype=complex)
        if t == 0.0:
            res = np.ones_like(u_arr, dtype=complex)
            return res if res.ndim > 0 else res.item()

        def _eval_single(u_val: complex) -> complex:
            k_u = 1j * u_val * self._beta - 0.5 * (u_val ** 2)
            f_t = (k_u / self._lambda) * (1.0 - math.exp(-self._lambda * t))

            def _int_real(s: float) -> float:
                theta_s = (k_u / self._lambda) * (1.0 - math.exp(-self._lambda * s)) + 1j * u_val * self._rho
                val = (self._alpha * theta_s) / (self._beta_bdlp - theta_s)
                return float(val.real)

            def _int_imag(s: float) -> float:
                theta_s = (k_u / self._lambda) * (1.0 - math.exp(-self._lambda * s)) + 1j * u_val * self._rho
                val = (self._alpha * theta_s) / (self._beta_bdlp - theta_s)
                return float(val.imag)

            ir, _ = integrate.quad(_int_real, 0.0, t, limit=100)
            ii, _ = integrate.quad(_int_imag, 0.0, t, limit=100)
            jump_integral = ir + 1j * ii

            exponent = 1j * u_val * self._mu * t + f_t * self._sigma0_sq + self._lambda * jump_integral
            return complex(np.exp(exponent))

        vec_func = np.vectorize(_eval_single)
        res = vec_func(u_arr)
        return res if res.ndim > 0 else res.item()

    # ------------------------------------------------------------------
    # European Option Pricing (Gil-Pelaez Inversion)
    # ------------------------------------------------------------------

    def call_price(self, strike: float, time_to_maturity: float = 1.0) -> float:
        r"""
        European call option price via Gil-Pelaez Fourier inversion.
        """
        if strike <= 0.0:
            raise ValueError(f"Strike must be strictly positive, got {strike}")
        if time_to_maturity <= 0.0:
            return float(max(0.0, self._s0 - strike))

        tau = time_to_maturity
        k_log = math.log(strike / self._s0)

        cf_minus_i = self.char_func(-1j, t=tau)

        def _p1_integrand(u: float) -> float:
            if u == 0.0:
                return 0.0
            num = np.exp(-1j * u * k_log) * self.char_func(u - 1j, t=tau)
            denom = 1j * u * cf_minus_i
            return float((num / denom).real)

        def _p2_integrand(u: float) -> float:
            if u == 0.0:
                return 0.0
            num = np.exp(-1j * u * k_log) * self.char_func(u, t=tau)
            denom = 1j * u
            return float((num / denom).real)

        i1, _ = integrate.quad(_p1_integrand, 1e-7, 100.0, limit=200)
        i2, _ = integrate.quad(_p2_integrand, 1e-7, 100.0, limit=200)

        p1 = 0.5 + i1 / math.pi
        p2 = 0.5 + i2 / math.pi

        disc_r = math.exp(-self._rate * tau)
        disc_q = math.exp(-self._dividend * tau)

        call_val = self._s0 * disc_q * p1 - strike * disc_r * p2
        return float(max(0.0, max(call_val, self._s0 * disc_q - strike * disc_r)))

    def put_price(self, strike: float, time_to_maturity: float = 1.0) -> float:
        r"""
        European put option price via exact Put-Call parity.
        """
        call = self.call_price(strike=strike, time_to_maturity=time_to_maturity)
        disc_r = math.exp(-self._rate * time_to_maturity)
        disc_q = math.exp(-self._dividend * time_to_maturity)
        put_val = call - self._s0 * disc_q + strike * disc_r
        return float(max(0.0, put_val))

    def implied_volatility(self, strike: float, time_to_maturity: float = 1.0) -> float:
        """Black-Scholes implied volatility for European call option."""
        call = self.call_price(strike=strike, time_to_maturity=time_to_maturity)
        disc_r = math.exp(-self._rate * time_to_maturity)
        disc_q = math.exp(-self._dividend * time_to_maturity)
        forward = self._s0 * math.exp((self._rate - self._dividend) * time_to_maturity)

        intrinsic = max(0.0, self._s0 * disc_q - strike * disc_r)
        if call <= intrinsic + 1e-12:
            return 0.0

        def _bs_err(vol: float) -> float:
            d1 = (math.log(forward / strike) + 0.5 * (vol ** 2) * time_to_maturity) / (vol * math.sqrt(time_to_maturity))
            d2 = d1 - vol * math.sqrt(time_to_maturity)
            bs_call = disc_r * (forward * stats.norm.cdf(d1) - strike * stats.norm.cdf(d2))
            return bs_call - call

        try:
            return float(optimize.brentq(_bs_err, 1e-4, 5.0, xtol=1e-6))
        except Exception:
            return float("nan")

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
        """Simulate sample paths of asset price :math:`S_t`."""
        from spxa.sim.exact import sample_bns
        return sample_bns(
            lambda_param=self._lambda,
            alpha=self._alpha,
            beta_bdlp=self._beta_bdlp,
            rho=self._rho,
            beta=self._beta,
            sigma0_sq=self._sigma0_sq,
            s0=self._s0,
            rate=self._rate,
            dividend=self._dividend,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=time_horizon,
            rng=rng,
            return_variance=False,
        )

    def simulate_joint(
        self,
        n_steps: int = 100,
        dt: Optional[float] = None,
        time_horizon: Optional[float] = None,
        n_paths: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate joint sample paths of asset price :math:`S_t` and variance :math:`\\sigma_t^2`."""
        from spxa.sim.exact import sample_bns
        return sample_bns(
            lambda_param=self._lambda,
            alpha=self._alpha,
            beta_bdlp=self._beta_bdlp,
            rho=self._rho,
            beta=self._beta,
            sigma0_sq=self._sigma0_sq,
            s0=self._s0,
            rate=self._rate,
            dividend=self._dividend,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=time_horizon,
            rng=rng,
            return_variance=True,
        )

    def __repr__(self) -> str:
        return (
            f"BNSModel(lambda_param={self._lambda}, alpha={self._alpha}, beta_bdlp={self._beta_bdlp}, "
            f"rho={self._rho}, beta={self._beta}, sigma0_sq={self._sigma0_sq}, s0={self._s0}, rate={self._rate})"
        )


# Aliases
BNSProcess = BNSModel
BarndorffNielsenShephard = BNSModel
BNS = BNSModel
