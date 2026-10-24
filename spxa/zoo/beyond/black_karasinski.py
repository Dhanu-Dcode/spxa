"""
Black-Karasinski (1991) Lognormal Short-Rate Process.

A single-factor interest rate model where the short rate is lognormally distributed,
ensuring strictly positive interest rates at all horizons:
    d(ln r_t) = a (b - ln r_t) dt + σ dW_t,   r_0 > 0
    r_t = exp(x_t), where x_t = ln(r_t) is an Ornstein-Uhlenbeck process.

Parameters
----------
a : float
    Mean-reversion speed of the log-rate (a > 0). Alias: kappa.
sigma : float
    Volatility of the log-rate (sigma > 0).
b : float
    Long-term reversion target of ln(r_t). The median stationary short rate is exp(b).
r0 : float
    Initial short rate (r0 > 0). Default 0.03 (3.0%).

References
----------
Black, F., & Karasinski, P. (1991). Bond and option pricing when short rates
    are lognormal. Financial Analysts Journal, 47(4), 52–59.
Brigo, D., & Mercurio, F. (2006). Interest Rate Models — Theory and Practice.
    Springer Finance, Section 3.7.
"""

from __future__ import annotations

import math
from typing import Optional, Union, Tuple, Dict

import numpy as np

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class BlackKarasinskiProcess(Process):
    r"""
    Black-Karasinski (1991) one-factor lognormal short-rate model.

    .. math::

        d(\ln r_t) = a (b - \ln r_t) dt + \sigma dW_t, \quad r_0 > 0
        r_t = \exp(x_t), \quad x_0 = \ln(r_0)

    where :math:`x_t = \ln(r_t)` follows a Gaussian Ornstein-Uhlenbeck diffusion.

    Parameters
    ----------
    a : float
        Mean-reversion speed of the log-rate :math:`a > 0`. Alias ``kappa``.
    sigma : float
        Volatility of the log-rate :math:`\sigma > 0`.
    b : float, optional
        Long-term mean of :math:`\ln r_t`. If omitted, defaults to :math:`\ln(r_0)`.
    r0 : float, optional
        Initial spot short rate :math:`r_0 > 0`. Default 0.03.
    """

    def __init__(
        self,
        a: float = 0.1,
        sigma: float = 0.15,
        b: Optional[float] = None,
        r0: float = 0.03,
        kappa: Optional[float] = None,
    ) -> None:
        reversion_speed = kappa if kappa is not None else a
        if reversion_speed <= 0.0:
            raise ValueError(
                f"Mean-reversion speed 'a' must be strictly positive, got {reversion_speed}"
            )
        if sigma <= 0.0:
            raise ValueError(f"Volatility 'sigma' must be strictly positive, got {sigma}")
        if r0 <= 0.0:
            raise ValueError(f"Initial short rate 'r0' must be strictly positive, got {r0}")

        self._a = float(reversion_speed)
        self._sigma = float(sigma)
        self._r0 = float(r0)
        self._x0 = float(math.log(r0))
        self._b = float(b) if b is not None else self._x0

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"BlackKarasinski(a={self._a}, sigma={self._sigma}, b={self._b:.4f}, r0={self._r0})",
            math_note=(
                f"d(ln r_t) = {self._a:.3f}({self._b:.4f} - ln r_t)dt + {self._sigma:.3f}dW_t, "
                f"E[r_∞]={self.stationary_mean * 100:.2f}%"
            ),
            reference="Black & Karasinski (1991); Brigo & Mercurio (2006) §3.7",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Properties & Aliases
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
        r"""Volatility of log-rate :math:`\sigma > 0`."""
        return self._sigma

    @property
    def b(self) -> float:
        r"""Long-term mean level of log-rate :math:`\ln r_t`."""
        return self._b

    @property
    def theta(self) -> float:
        r"""Mean-reversion drift product :math:`\theta = a \cdot b`."""
        return self._a * self._b

    @property
    def r0(self) -> float:
        """Initial spot short rate :math:`r_0 > 0`."""
        return self._r0

    @property
    def x0(self) -> float:
        r"""Initial log-rate :math:`x_0 = \ln(r_0)`."""
        return self._x0

    # ------------------------------------------------------------------
    # Exactness & Triplet
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "BlackKarasinskiProcess has ExactnessLevel.MOMENT_PROPAGATION and does not possess "
            "a standard Levy triplet: the state variable x_t = ln(r_t) is an Ornstein-Uhlenbeck "
            "diffusion with state-dependent mean-reversion drift, violating the independent and "
            "stationary increments property of Levy processes. Conditional lognormal moments, "
            "exact OU transitions, and Monte Carlo bond pricing are available directly."
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
                    f"Black-Karasinski lognormal short-rate process: "
                    f"d(ln r_t) = {self._a:.3f}({self._b:.4f} - ln r_t)dt + {self._sigma:.3f}dW_t, r_0={self._r0}."
                ),
                f"Strictly positive rates r_t > 0 a.s. Stationary mean E[r_∞]={self.stationary_mean * 100:.2f}%.",
            ],
        )

    def _degradation_reason(self) -> str:
        return "Black-Karasinski short-rate process has state-dependent mean-reverting lognormal dynamics."

    # ------------------------------------------------------------------
    # Log-rate (x_t = ln r_t) Moments
    # ------------------------------------------------------------------

    def log_rate_mean(self, t: float = 1.0) -> float:
        r"""
        Conditional expectation of log-rate :math:`x_t = \ln(r_t)`.

        .. math::

            \mathbb{E}[x_t \mid x_0] = b + (x_0 - b) e^{-at}
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        return float(self._b + (self._x0 - self._b) * math.exp(-self._a * t))

    def log_rate_variance(self, t: float = 1.0) -> float:
        r"""
        Conditional variance of log-rate :math:`x_t = \ln(r_t)`.

        .. math::

            \mathrm{Var}(x_t \mid x_0) = \frac{\sigma^2}{2a} (1 - e^{-2at})
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        return float((self._sigma**2 / (2.0 * self._a)) * (1.0 - math.exp(-2.0 * self._a * t)))

    def log_rate_std(self, t: float = 1.0) -> float:
        """Conditional standard deviation of log-rate :math:`x_t`."""
        return math.sqrt(self.log_rate_variance(t))

    # ------------------------------------------------------------------
    # Short-rate (r_t = exp(x_t)) Moments
    # ------------------------------------------------------------------

    def conditional_mean(self, t: float = 1.0) -> float:
        r"""
        Exact conditional expectation of short rate :math:`\mathbb{E}[r_t \mid r_0]`.

        Since :math:`r_t = \exp(x_t)` where :math:`x_t \sim \mathcal{N}(\mu_x(t), \sigma_x^2(t))`:

        .. math::

            \mathbb{E}[r_t \mid r_0] = \exp\left(\mu_x(t) + \frac{1}{2} \sigma_x^2(t)\right)
        """
        mu_x = self.log_rate_mean(t)
        var_x = self.log_rate_variance(t)
        return float(math.exp(mu_x + 0.5 * var_x))

    def conditional_variance(self, t: float = 1.0) -> float:
        r"""
        Exact conditional variance of short rate :math:`\mathrm{Var}(r_t \mid r_0)`.

        .. math::

            \mathrm{Var}(r_t \mid r_0) = \exp(2\mu_x(t) + \sigma_x^2(t)) \left( \exp(\sigma_x^2(t)) - 1 \right)
        """
        mu_x = self.log_rate_mean(t)
        var_x = self.log_rate_variance(t)
        return float(math.exp(2.0 * mu_x + var_x) * (math.exp(var_x) - 1.0))

    def conditional_std(self, t: float = 1.0) -> float:
        r"""Exact conditional standard deviation of short rate :math:`\sqrt{\mathrm{Var}(r_t)}`."""
        return math.sqrt(self.conditional_variance(t))

    def conditional_cumulants(self, t: float = 1.0, order: int = 2) -> Dict[int, float]:
        """Dictionary of short-rate cumulants {1: mean, 2: variance}."""
        if order < 1:
            raise ValueError(f"Order must be >= 1, got {order}")
        cums = {1: self.conditional_mean(t)}
        if order >= 2:
            cums[2] = self.conditional_variance(t)
        return cums

    def cumulants(self, order: int = 2) -> Dict[int, float]:
        """Cumulants at horizon :math:`t = 1.0`."""
        return self.conditional_cumulants(t=1.0, order=order)

    # ------------------------------------------------------------------
    # Stationary Law
    # ------------------------------------------------------------------

    @property
    def stationary_log_mean(self) -> float:
        """Asymptotic mean of log-rate :math:`b`."""
        return self._b

    @property
    def stationary_log_variance(self) -> float:
        r"""Asymptotic variance of log-rate :math:`\frac{\sigma^2}{2a}`."""
        return self._sigma**2 / (2.0 * self._a)

    @property
    def stationary_median(self) -> float:
        """Asymptotic median short rate :math:`e^b`."""
        return float(math.exp(self._b))

    @property
    def stationary_mean(self) -> float:
        r"""
        Asymptotic stationary mean of short rate :math:`\mathbb{E}[r_\infty] = \exp\left(b + \frac{\sigma^2}{4a}\right)`.
        """
        return float(math.exp(self._b + self._sigma**2 / (4.0 * self._a)))

    @property
    def stationary_variance(self) -> float:
        r"""
        Asymptotic stationary variance of short rate :math:`\mathrm{Var}(r_\infty)`.
        """
        var_inf = self.stationary_log_variance
        return float(math.exp(2.0 * self._b + var_inf) * (math.exp(var_inf) - 1.0))

    # ------------------------------------------------------------------
    # Term Structure & Zero-Coupon Bond Pricing
    # ------------------------------------------------------------------

    def bond_price(
        self,
        T: Union[float, np.ndarray],
        n_paths: int = 15_000,
        n_steps_per_year: int = 30,
        rng: Optional[np.random.Generator] = None,
    ) -> Union[float, np.ndarray]:
        r"""
        Compute zero-coupon bond price :math:`P(0, T) = \mathbb{E}\left[\exp\left(-\int_0^T r_s ds\right)\right]`.

        Evaluated via Monte Carlo path simulation with exact log-rate Ornstein-Uhlenbeck increments
        and trapezoidal integration.

        Parameters
        ----------
        T : float or np.ndarray
            Maturity horizon(s) in years.
        n_paths : int
            Number of simulation paths.
        n_steps_per_year : int
            Discretization steps per year.
        rng : np.random.Generator, optional
            NumPy random generator.
        """
        T_arr = np.asarray(T, dtype=float)
        rng = rng or np.random.default_rng(2026)

        def _price_single(t_mat: float) -> float:
            if t_mat <= 0.0:
                return 1.0
            n_steps = max(5, int(round(t_mat * n_steps_per_year)))
            dt = t_mat / n_steps

            exp_adt = math.exp(-self._a * dt)
            std_step = math.sqrt(
                (self._sigma**2 / (2.0 * self._a)) * (1.0 - math.exp(-2.0 * self._a * dt))
            )

            curr_x = np.full(n_paths, self._x0, dtype=float)
            # Running trapezoidal sum
            int_r = 0.5 * np.exp(curr_x) * dt

            for k in range(n_steps - 1):
                z = rng.standard_normal(size=n_paths)
                curr_x = self._b + (curr_x - self._b) * exp_adt + std_step * z
                int_r += np.exp(curr_x) * dt

            # Final step
            z = rng.standard_normal(size=n_paths)
            curr_x = self._b + (curr_x - self._b) * exp_adt + std_step * z
            int_r += 0.5 * np.exp(curr_x) * dt

            discount_factors = np.exp(-int_r)
            return float(np.mean(discount_factors))

        if T_arr.ndim == 0:
            return _price_single(float(T_arr))
        return np.array([_price_single(float(t)) for t in T_arr], dtype=float)

    def yield_to_maturity(
        self,
        T: Union[float, np.ndarray],
        n_paths: int = 15_000,
        rng: Optional[np.random.Generator] = None,
    ) -> Union[float, np.ndarray]:
        r"""
        Continuously compounded zero-coupon spot yield :math:`y(0, T) = -\frac{\ln P(0, T)}{T}`.
        """
        T_arr = np.asarray(T, dtype=float)
        p = self.bond_price(T_arr, n_paths=n_paths, rng=rng)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = -np.log(p) / T_arr
        if T_arr.ndim == 0:
            return float(y)
        return y

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
        Simulate sample paths of the short rate :math:`r_t`.

        Returns
        -------
        np.ndarray
            Paths array of shape ``(n_paths, n_steps + 1)`` with :math:`r_t > 0`.
        """
        from spxa.sim.exact import sample_black_karasinski

        return sample_black_karasinski(
            a=self._a,
            sigma=self._sigma,
            b=self._b,
            r0=self._r0,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=time_horizon,
            rng=rng,
            return_log_rate=False,
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
        Simulate coupled paths of short rate :math:`r_t` and log-rate :math:`x_t = \ln(r_t)`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple ``(r_paths, x_paths)``, each of shape ``(n_paths, n_steps + 1)``.
        """
        from spxa.sim.exact import sample_black_karasinski

        return sample_black_karasinski(
            a=self._a,
            sigma=self._sigma,
            b=self._b,
            r0=self._r0,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=time_horizon,
            rng=rng,
            return_log_rate=True,
        )

    def __repr__(self) -> str:
        return (
            f"BlackKarasinskiProcess(a={self._a}, sigma={self._sigma}, "
            f"b={self._b:.4f}, r0={self._r0})"
        )


# Aliases
BlackKarasinski = BlackKarasinskiProcess
BKProcess = BlackKarasinskiProcess
BKModel = BlackKarasinskiProcess
