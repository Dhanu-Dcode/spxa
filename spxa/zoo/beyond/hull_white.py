"""
Hull-White (1990) one-factor short-rate model (Extended Vasicek).

The Hull-White (1990, 1993) one-factor model describes the evolution of the
instantaneous short interest rate r_t as an affine Gaussian diffusion:

    dr_t = (θ(t) - a r_t) dt + σ dW_t,   r_0 ∈ ℝ

where:
    - a > 0 is the speed of mean reversion.
    - σ > 0 is the short-rate volatility.
    - θ(t) is a time-dependent drift parameter chosen to fit the initial term
      structure of interest rates. In the baseline stationary formulation,
      θ(t) = a · b where b ∈ ℝ is the long-term reversion level (Vasicek 1977).

Under risk-neutral pricing measure ℚ, the model admits closed-form analytical
solutions for zero-coupon bond prices, forward rates, yield curves, European bond
options (Jamshidian 1989), caplets, floorlets, and exact path simulation without
time-discretization error.

Zero-Coupon Bond Price
----------------------
The price at time t of a zero-coupon bond maturing at T ≥ t is:

    P(t, T) = A(t, T) exp(-B(t, T) r_t)

where:
    B(t, T) = (1 - exp(-a (T - t))) / a
    ln A(t, T) = (b - σ² / (2 a²)) · (B(t, T) - (T - t)) - (σ² / (4 a)) · B(t, T)²

Bond Option Pricing (Jamshidian 1989 / Hull-White 1990)
-------------------------------------------------------
A European call option with expiry T on a discount bond maturing at S > T, strike K:

    C(t, T, S, K) = P(t, S) Φ(d_1) - K P(t, T) Φ(d_2)
    P_put(t, T, S, K) = K P(t, T) Φ(-d_2) - P(t, S) Φ(-d_1)

where:
    σ_P = σ · B(T, S) · √((1 - exp(-2 a (T - t))) / (2 a))
    d_1 = (ln(P(t, S) / (K · P(t, T))) + ½ σ_P²) / σ_P
    d_2 = d_1 - σ_P

References
----------
Hull, J., & White, A. (1990). Pricing interest-rate-derivative securities.
*The Review of Financial Studies*, 3(4), 573–592.

Hull, J., & White, A. (1993). One-factor interest-rate models and the valuation
of interest-rate derivative securities.
*Journal of Financial and Quantitative Analysis*, 28(2), 235–254.

Jamshidian, F. (1989). An exact bond option formula.
*The Journal of Finance*, 44(1), 205–209.

Vasicek, O. (1977). An equilibrium characterization of the term structure.
*Journal of Financial Economics*, 5(2), 177–188.
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
from spxa.sim.exact import sample_hull_white
from spxa.story.narrator import CompositionNode, NodeKind


class HullWhiteProcess(Process):
    r"""
    Hull-White (1990) one-factor Gaussian short-rate model.

    .. math::

        dr_t = a(b - r_t) dt + \sigma dW_t, \quad r_0 \in \mathbb{R}

    where :math:`a > 0` is the mean-reversion speed, :math:`b \in \mathbb{R}` is the
    long-term baseline reversion level (:math:`\theta = a \cdot b`), and :math:`\sigma > 0`
    is the instantaneous volatility.

    Parameters
    ----------
    a :
        Mean-reversion speed :math:`a > 0`. Default 0.10. May also be supplied as ``kappa`` or ``speed``.
    sigma :
        Short-rate volatility :math:`\sigma > 0`. Default 0.02. May also be supplied as ``vol``.
    b :
        Long-term rate level :math:`b \in \mathbb{R}`. Default 0.03.
    r0 :
        Initial short rate :math:`r_0 \in \mathbb{R}`. Default 0.03.
    theta :
        Optional constant drift :math:`\theta = a \cdot b`. If provided, overrides ``b`` with ``theta / a``.
    """

    def __init__(
        self,
        a: float = 0.10,
        sigma: float = 0.02,
        b: float = 0.03,
        r0: float = 0.03,
        theta: Optional[float] = None,
        **kwargs,
    ) -> None:
        a_val = kwargs.get("kappa", kwargs.get("speed", a))
        sigma_val = kwargs.get("vol", sigma)

        a_val = float(a_val)
        sigma_val = float(sigma_val)
        r0 = float(r0)

        if theta is not None:
            b_val = float(theta) / a_val
        else:
            b_val = float(b)

        if a_val <= 0.0:
            raise ValueError(f"Mean-reversion speed 'a' must be strictly positive, got {a_val}")
        if sigma_val <= 0.0:
            raise ValueError(f"Volatility 'sigma' must be strictly positive, got {sigma_val}")

        self._a = a_val
        self._sigma = sigma_val
        self._b = b_val
        self._r0 = r0

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"HullWhiteProcess(a={self._a}, sigma={self._sigma}, b={self._b}, r0={self._r0})",
            math_note=f"dr_t = {self._a}({self._b} - r_t)dt + {self._sigma}dW_t, r_0={self._r0}",
            reference="Hull, J., & White, A. (1990), Review of Financial Studies 3(4), 573–592",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    def __repr__(self) -> str:
        return f"HullWhiteProcess(a={self._a}, sigma={self._sigma}, b={self._b}, r0={self._r0})"

    # ------------------------------------------------------------------
    # Model Parameters & Structural Properties
    # ------------------------------------------------------------------

    @property
    def a(self) -> float:
        """Mean-reversion speed :math:`a > 0`."""
        return self._a

    @property
    def kappa(self) -> float:
        """Alias for mean-reversion speed :math:`a`."""
        return self._a

    @property
    def sigma(self) -> float:
        """Instantaneous volatility :math:`\\sigma > 0`."""
        return self._sigma

    @property
    def b(self) -> float:
        """Long-term mean rate level :math:`b \\in \\mathbb{R}`."""
        return self._b

    @property
    def theta(self) -> float:
        """Drift level :math:`\\theta = a \\cdot b`."""
        return float(self._a * self._b)

    @property
    def r0(self) -> float:
        """Initial short rate :math:`r_0 \\in \\mathbb{R}`."""
        return self._r0

    # ------------------------------------------------------------------
    # Exactness & Triplet
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "HullWhiteProcess has ExactnessLevel.MOMENT_PROPAGATION and does not possess "
            "a standard Levy triplet: the drift a(b - r_t) is state-dependent, violating "
            "the independent and stationary increments requirement of pure Levy processes. "
            "Gaussian transition densities, exact zero-coupon bond pricing, analytical bond "
            "options, and exact simulation are available directly."
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
                    f"Hull-White (1990) one-factor short-rate model: "
                    f"dr_t = {self._a}({self._b} - r_t)dt + {self._sigma}dW_t, r_0={self._r0}."
                ),
                f"Parameters: a={self._a}, sigma={self._sigma}, b={self._b}, r0={self._r0}.",
            ],
        )

    def _degradation_reason(self) -> str:
        return "Hull-White (1990) short-rate model has state-dependent mean-reverting drift."

    # ------------------------------------------------------------------
    # Distributional Moments & Analytics
    # ------------------------------------------------------------------

    def conditional_mean(self, t: float = 1.0, r_init: Optional[float] = None) -> float:
        r"""
        Expected short rate :math:`\mathbb{E}[r_t \mid r_0]`:

        .. math::

            \mathbb{E}[r_t \mid r_0] = b + (r_0 - b) e^{-a t}
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        r_start = self._r0 if r_init is None else float(r_init)
        return float(self._b + (r_start - self._b) * math.exp(-self._a * t))

    def conditional_variance(self, t: float = 1.0) -> float:
        r"""
        Variance of the short rate :math:`\mathrm{Var}(r_t \mid r_0)`:

        .. math::

            \mathrm{Var}(r_t \mid r_0) = \frac{\sigma^2}{2a} (1 - e^{-2at})
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        if t == 0.0:
            return 0.0
        return float((self._sigma**2 / (2.0 * self._a)) * (1.0 - math.exp(-2.0 * self._a * t)))

    def conditional_std(self, t: float = 1.0) -> float:
        r"""Standard deviation :math:`\sqrt{\mathrm{Var}(r_t \mid r_0)}`."""
        return float(math.sqrt(self.conditional_variance(t)))

    @property
    def stationary_mean(self) -> float:
        r"""Long-term stationary mean :math:`b`."""
        return self._b

    @property
    def stationary_variance(self) -> float:
        r"""Long-term stationary variance :math:`\frac{\sigma^2}{2a}`."""
        return float(self._sigma**2 / (2.0 * self._a))

    def integrated_mean(self, t: float, r_init: Optional[float] = None) -> float:
        r"""
        Expected integrated short rate :math:`\mathbb{E}\left[\int_0^t r_s \, ds \mid r_0\right]`:

        .. math::

            \mathbb{E}\left[\int_0^t r_s \, ds\right] = b t + (r_0 - b) \frac{1 - e^{-at}}{a}
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        if t == 0.0:
            return 0.0
        r_start = self._r0 if r_init is None else float(r_init)
        return float(self._b * t + (r_start - self._b) * (1.0 - math.exp(-self._a * t)) / self._a)

    def integrated_variance(self, t: float) -> float:
        r"""
        Variance of integrated short rate :math:`\mathrm{Var}\left(\int_0^t r_s \, ds \mid r_0\right)`:

        .. math::

            \mathrm{Var}\left(\int_0^t r_s \, ds\right) = \frac{\sigma^2}{a^2} \left[ t - \frac{2(1 - e^{-at})}{a} + \frac{1 - e^{-2at}}{2a} \right]
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        if t == 0.0:
            return 0.0
        term1 = t
        term2 = -2.0 * (1.0 - math.exp(-self._a * t)) / self._a
        term3 = (1.0 - math.exp(-2.0 * self._a * t)) / (2.0 * self._a)
        return float((self._sigma**2 / (self._a**2)) * (term1 + term2 + term3))

    def autocovariance(self, t: float, s: float) -> float:
        r"""
        Autocovariance :math:`\mathrm{Cov}(r_t, r_s \mid r_0)`:

        .. math::

            \mathrm{Cov}(r_t, r_s) = \frac{\sigma^2}{2a} \left( e^{-a|t - s|} - e^{-a(t + s)} \right)
        """
        if t < 0.0 or s < 0.0:
            raise ValueError("Evaluation times t and s must be non-negative")
        return float(
            (self._sigma**2 / (2.0 * self._a))
            * (math.exp(-self._a * abs(t - s)) - math.exp(-self._a * (t + s)))
        )

    def stationary_autocovariance(self, tau: float) -> float:
        r"""Stationary autocovariance :math:`\frac{\sigma^2}{2a} e^{-a |\tau|}`."""
        return float((self._sigma**2 / (2.0 * self._a)) * math.exp(-self._a * abs(tau)))

    def char_func(self, u: Union[float, np.ndarray], t: float = 1.0) -> Union[complex, np.ndarray]:
        r"""
        Characteristic function of short rate :math:`r_t`:

        .. math::

            \phi_{r_t}(u) = \exp\left( i u \mathbb{E}[r_t] - \frac{1}{2} u^2 \mathrm{Var}(r_t) \right)
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        m = self.conditional_mean(t)
        v = self.conditional_variance(t)
        u_arr = np.asarray(u)
        is_scalar = u_arr.ndim == 0
        cf_val = np.exp(1j * u_arr * m - 0.5 * (u_arr**2) * v)
        if is_scalar:
            return complex(cf_val)
        return cf_val

    def conditional_cumulants(self, t: float = 1.0, order: int = 4) -> dict[int, float]:
        r"""
        Cumulants :math:`\kappa_1(t), \dots, \kappa_{\mathrm{order}}(t)` of Gaussian short rate :math:`r_t`.
        """
        if order < 1:
            raise ValueError(f"Cumulant order must be at least 1, got {order}")
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        kappas = {1: self.conditional_mean(t), 2: self.conditional_variance(t)}
        for n in range(3, order + 1):
            kappas[n] = 0.0  # Higher cumulants vanish for Gaussian distributions
        return kappas

    def cumulants(self, order: int) -> dict[int, float]:
        """Cumulants of short rate at horizon :math:`t = 1.0`."""
        return self.conditional_cumulants(t=1.0, order=order)

    # ------------------------------------------------------------------
    # Zero-Coupon Bond Pricing & Term Structure
    # ------------------------------------------------------------------

    def B(self, t: float, T: float) -> float:
        r"""
        Affine bond pricing exponent coefficient :math:`B(t, T)`:

        .. math::

            B(t, T) = \frac{1 - e^{-a(T - t)}}{a}
        """
        if T < t:
            raise ValueError(f"Maturity T ({T}) must be >= observation time t ({t})")
        if T == t:
            return 0.0
        return float((1.0 - math.exp(-self._a * (T - t))) / self._a)

    def A(self, t: float, T: float) -> float:
        r"""
        Affine bond pricing scale coefficient :math:`A(t, T)`:

        .. math::

            \ln A(t, T) = \left(b - \frac{\sigma^2}{2a^2}\right)(B(t, T) - (T - t)) - \frac{\sigma^2}{4a} B(t, T)^2
        """
        if T < t:
            raise ValueError(f"Maturity T ({T}) must be >= observation time t ({t})")
        if T == t:
            return 1.0
        tau = T - t
        B_val = self.B(t, T)
        term1 = (self._b - (self._sigma**2) / (2.0 * self._a**2)) * (B_val - tau)
        term2 = -(self._sigma**2 / (4.0 * self._a)) * (B_val**2)
        return float(math.exp(term1 + term2))

    def bond_price(
        self,
        T: Union[float, np.ndarray],
        r: Optional[Union[float, np.ndarray]] = None,
        t: float = 0.0,
    ) -> Union[float, np.ndarray]:
        r"""
        Zero-coupon bond price :math:`P(t, T)`:

        .. math::

            P(t, T) = A(t, T) \exp(-B(t, T) r_t)

        Parameters
        ----------
        T :
            Maturity time or array of maturities (:math:`T \ge t`).
        r :
            Short rate at time t. If None, uses :math:`r_0`. May be scalar or array. Default None.
        t :
            Current observation time :math:`t \ge 0`. Default 0.0.

        Returns
        -------
        float or np.ndarray
        """
        r_arr = np.asarray(self._r0 if r is None else r, dtype=float)
        T_arr = np.asarray(T, dtype=float)

        if T_arr.ndim == 0 and r_arr.ndim == 0:
            mat = float(T_arr)
            if mat < t:
                raise ValueError(f"Maturity T ({mat}) must be >= observation time t ({t})")
            if mat == t:
                return 1.0
            return float(self.A(t, mat) * math.exp(-self.B(t, mat) * float(r_arr)))

        if T_arr.ndim == 0 and r_arr.ndim > 0:
            mat = float(T_arr)
            if mat < t:
                raise ValueError(f"Maturity T ({mat}) must be >= observation time t ({t})")
            if mat == t:
                return np.ones_like(r_arr)
            return self.A(t, mat) * np.exp(-self.B(t, mat) * r_arr)

        def _p_single(mat: float) -> float:
            if mat < t:
                raise ValueError(f"Maturity T ({mat}) must be >= observation time t ({t})")
            if mat == t:
                return 1.0
            return float(self.A(t, mat) * math.exp(-self.B(t, mat) * float(r_arr)))

        return np.array([_p_single(float(mat)) for mat in T_arr], dtype=float)

    def yield_to_maturity(
        self,
        T: Union[float, np.ndarray],
        r: Optional[float] = None,
        t: float = 0.0,
    ) -> Union[float, np.ndarray]:
        r"""
        Continuously compounded zero-coupon spot yield :math:`y(t, T) = -\frac{\ln P(t, T)}{T - t}`.
        """
        r_val = self._r0 if r is None else float(r)

        def _y_single(mat: float) -> float:
            if mat <= t:
                return r_val
            p = self.bond_price(mat, r=r_val, t=t)
            return float(-math.log(p) / (mat - t))

        T_arr = np.asarray(T)
        if T_arr.ndim == 0:
            return _y_single(float(T_arr))
        return np.array([_y_single(float(mat)) for mat in T_arr], dtype=float)

    def instantaneous_forward_rate(
        self,
        T: Union[float, np.ndarray],
        r: Optional[float] = None,
        t: float = 0.0,
    ) -> Union[float, np.ndarray]:
        r"""
        Instantaneous forward rate :math:`f(t, T) = -\frac{\partial \ln P(t, T)}{\partial T}`:

        .. math::

            f(t, T) = b + (r_t - b) e^{-a(T-t)} - \frac{\sigma^2}{2a^2}(1 - e^{-a(T-t)})^2
        """
        r_val = self._r0 if r is None else float(r)

        def _f_single(mat: float) -> float:
            if mat < t:
                raise ValueError(f"Maturity T ({mat}) must be >= observation time t ({t})")
            tau = mat - t
            exp_atau = math.exp(-self._a * tau)
            drift_part = self._b + (r_val - self._b) * exp_atau
            convexity_part = (self._sigma**2 / (2.0 * self._a**2)) * ((1.0 - exp_atau) ** 2)
            return float(drift_part - convexity_part)

        T_arr = np.asarray(T)
        if T_arr.ndim == 0:
            return _f_single(float(T_arr))
        return np.array([_f_single(float(mat)) for mat in T_arr], dtype=float)

    # ------------------------------------------------------------------
    # European Bond Options (Jamshidian 1989 / Hull-White 1990)
    # ------------------------------------------------------------------

    def bond_call_price(
        self,
        T: float,
        S: float,
        K: float,
        r: Optional[float] = None,
        t: float = 0.0,
    ) -> float:
        r"""
        European call option on a zero-coupon bond maturing at :math:`S > T`, strike :math:`K`, option expiry :math:`T`.

        .. math::

            C(t, T, S, K) = P(t, S) \Phi(d_1) - K P(t, T) \Phi(d_2)

        where:

        .. math::

            \sigma_P = \sigma B(T, S) \sqrt{\frac{1 - e^{-2a(T - t)}}{2a}} \\
            d_1 = \frac{\ln\left(\frac{P(t, S)}{K P(t, T)}\right) + \frac{1}{2}\sigma_P^2}{\sigma_P}, \quad d_2 = d_1 - \sigma_P
        """
        if not (t <= T < S):
            raise ValueError(f"Requirement t <= T < S violated: t={t}, T={T}, S={S}")
        if K <= 0.0:
            raise ValueError(f"Strike K must be strictly positive, got {K}")

        r_val = self._r0 if r is None else float(r)
        p_t_S = float(self.bond_price(S, r=r_val, t=t))
        p_t_T = float(self.bond_price(T, r=r_val, t=t))

        tau = T - t
        sigma_P = (
            self._sigma
            * self.B(T, S)
            * math.sqrt((1.0 - math.exp(-2.0 * self._a * tau)) / (2.0 * self._a))
        )

        if sigma_P < 1e-12:
            return float(max(0.0, p_t_S - K * p_t_T))

        d1 = (math.log(p_t_S / (K * p_t_T)) + 0.5 * sigma_P**2) / sigma_P
        d2 = d1 - sigma_P
        call_val = p_t_S * norm.cdf(d1) - K * p_t_T * norm.cdf(d2)
        return float(max(0.0, call_val))

    def bond_put_price(
        self,
        T: float,
        S: float,
        K: float,
        r: Optional[float] = None,
        t: float = 0.0,
    ) -> float:
        r"""
        European put option on a zero-coupon bond maturing at :math:`S > T`, strike :math:`K`, option expiry :math:`T`:

        .. math::

            P_{\mathrm{put}}(t, T, S, K) = K P(t, T) \Phi(-d_2) - P(t, S) \Phi(-d_1)
        """
        if not (t <= T < S):
            raise ValueError(f"Requirement t <= T < S violated: t={t}, T={T}, S={S}")
        if K <= 0.0:
            raise ValueError(f"Strike K must be strictly positive, got {K}")

        r_val = self._r0 if r is None else float(r)
        p_t_S = float(self.bond_price(S, r=r_val, t=t))
        p_t_T = float(self.bond_price(T, r=r_val, t=t))

        tau = T - t
        sigma_P = (
            self._sigma
            * self.B(T, S)
            * math.sqrt((1.0 - math.exp(-2.0 * self._a * tau)) / (2.0 * self._a))
        )

        if sigma_P < 1e-12:
            return float(max(0.0, K * p_t_T - p_t_S))

        d1 = (math.log(p_t_S / (K * p_t_T)) + 0.5 * sigma_P**2) / sigma_P
        d2 = d1 - sigma_P
        put_val = K * p_t_T * norm.cdf(-d2) - p_t_S * norm.cdf(-d1)
        return float(max(0.0, put_val))

    def caplet_price(
        self,
        T: float,
        strike_rate: float,
        tau: float = 0.5,
        r: Optional[float] = None,
        t: float = 0.0,
    ) -> float:
        r"""
        Caplet option pricing via bond put option equivalence.

        A caplet fixing at T, settling at :math:`S = T + \tau` on strike rate :math:`R_K`
        is equivalent to :math:`(1 + \tau R_K)` put options on a zero-coupon bond :math:`P(t, S)`
        with strike :math:`\frac{1}{1 + \tau R_K}`.
        """
        if strike_rate <= 0.0:
            raise ValueError(f"Strike rate must be strictly positive, got {strike_rate}")
        if tau <= 0.0:
            raise ValueError(f"Accrual fraction tau must be strictly positive, got {tau}")
        S = T + tau
        K_bond = 1.0 / (1.0 + tau * strike_rate)
        put_val = self.bond_put_price(T=T, S=S, K=K_bond, r=r, t=t)
        return float((1.0 + tau * strike_rate) * put_val)

    def floorlet_price(
        self,
        T: float,
        strike_rate: float,
        tau: float = 0.5,
        r: Optional[float] = None,
        t: float = 0.0,
    ) -> float:
        r"""
        Floorlet option pricing via bond call option equivalence.
        """
        if strike_rate <= 0.0:
            raise ValueError(f"Strike rate must be strictly positive, got {strike_rate}")
        if tau <= 0.0:
            raise ValueError(f"Accrual fraction tau must be strictly positive, got {tau}")
        S = T + tau
        K_bond = 1.0 / (1.0 + tau * strike_rate)
        call_val = self.bond_call_price(T=T, S=S, K=K_bond, r=r, t=t)
        return float((1.0 + tau * strike_rate) * call_val)

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
        method: str = "exact",
    ) -> np.ndarray:
        r"""
        Simulate sample paths of the Hull-White short rate :math:`r_t`.

        Parameters
        ----------
        n_steps :
            Number of time steps.
        dt :
            Step size :math:`\Delta t`.
        time_horizon :
            Total time horizon :math:`T`.
        n_paths :
            Number of independent paths.
        rng :
            NumPy random generator.
        method :
            ``'exact'`` (Gaussian conditional transition with zero discretization error)
            or ``'euler'``.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_paths, n_steps + 1)`` with :math:`r(0) = r_0`.
        """
        return sample_hull_white(
            a=self._a,
            sigma=self._sigma,
            b=self._b,
            r0=self._r0,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=time_horizon,
            rng=rng,
            method=method,
        )


# Aliases
HullWhite = HullWhiteProcess
HW1F = HullWhiteProcess
ExtendedVasicek = HullWhiteProcess
