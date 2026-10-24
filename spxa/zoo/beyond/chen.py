"""
Chen Three-Factor Short Rate Model (Chen 1996).

A comprehensive three-factor term structure model with stochastic central tendency
(mean-reversion level) and stochastic volatility:

State variables:
    dr_t     = k (θ_t - r_t) dt + √(ζ_t r_t) dW_{1,t}
    dθ_t     = κ (θ_bar - θ_t) dt + α √θ_t dW_{2,t}
    dζ_t     = λ (ζ_bar - ζ_t) dt + β √ζ_t dW_{3,t}

where:
    - r_t ≥ 0 is the instantaneous short rate
    - θ_t ≥ 0 is the stochastic central tendency (stochastic mean)
    - ζ_t ≥ 0 is the stochastic variance / volatility factor
    - W_{1,t}, W_{2,t}, W_{3,t} are Brownian motions

Features:
    - Exact closed-form conditional expectations for short rate, central tendency, and variance
    - Exact 10-dimensional moment generator system for second moments, covariances, and variance
    - Feller conditions verification for central tendency and volatility processes
    - Zero-coupon bond pricing and term structure yield curve
    - High-performance coupled factor simulation

References
----------
Chen, L. (1996). Stochastic Mean and Stochastic Volatility - A Three-Factor Model
    of the Term Structure of Interest Rates and Its Application in Derivatives Pricing.
    Financial Markets, Institutions & Instruments, 5(1), 1–88.
Chen, L. (1996). A Three-Factor Model of Interest Rates.
    The Journal of Fixed Income, 6(3), 8–29.
"""

from __future__ import annotations

import math
from typing import Optional, Union, Tuple, Dict, Any, Sequence

import numpy as np
from scipy import linalg as la  # type: ignore[import-untyped]

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class ChenModel(Process):
    r"""
    Chen Three-Factor Short Rate Model.

    .. math::

        dr_t = k (\theta_t - r_t) dt + \sqrt{\zeta_t r_t} dW_{1,t} \\
        d\theta_t = \kappa (\bar{\theta} - \theta_t) dt + \alpha \sqrt{\theta_t} dW_{2,t} \\
        d\zeta_t = \lambda (\bar{\zeta} - \zeta_t) dt + \beta \sqrt{\zeta_t} dW_{3,t}

    Parameters
    ----------
    k : float
        Mean-reversion speed of the short rate towards central tendency :math:`\theta_t` (:math:`k > 0`).
    theta_bar : float
        Long-term central tendency level :math:`\bar{\theta} > 0`.
    kappa : float
        Mean-reversion speed of central tendency :math:`\theta_t` (:math:`\kappa > 0`).
    alpha : float
        Volatility parameter of central tendency (:math:`\alpha > 0`).
    zeta_bar : float
        Long-term variance level :math:`\bar{\zeta} > 0`.
    lambda_param : float
        Mean-reversion speed of variance factor :math:`\zeta_t` (:math:`\lambda > 0`).
    beta : float
        Volatility parameter of variance factor (:math:`\beta > 0`).
    r0 : float
        Initial short rate :math:`r_0 > 0`. Default 0.04.
    theta0 : float, optional
        Initial central tendency :math:`\theta_0 > 0`. Defaults to ``theta_bar``.
    zeta0 : float, optional
        Initial variance factor :math:`\zeta_0 > 0`. Defaults to ``zeta_bar``.
    """

    def __init__(
        self,
        k: float = 1.5,
        theta_bar: float = 0.05,
        kappa: float = 0.8,
        alpha: float = 0.04,
        zeta_bar: float = 0.02,
        lambda_param: float = 1.0,
        beta: float = 0.05,
        r0: float = 0.04,
        theta0: Optional[float] = None,
        zeta0: Optional[float] = None,
    ) -> None:
        if k <= 0.0:
            raise ValueError(f"k must be strictly positive, got {k}")
        if theta_bar <= 0.0:
            raise ValueError(f"theta_bar must be strictly positive, got {theta_bar}")
        if kappa <= 0.0:
            raise ValueError(f"kappa must be strictly positive, got {kappa}")
        if alpha <= 0.0:
            raise ValueError(f"alpha must be strictly positive, got {alpha}")
        if zeta_bar <= 0.0:
            raise ValueError(f"zeta_bar must be strictly positive, got {zeta_bar}")
        if lambda_param <= 0.0:
            raise ValueError(f"lambda_param must be strictly positive, got {lambda_param}")
        if beta <= 0.0:
            raise ValueError(f"beta must be strictly positive, got {beta}")
        if r0 <= 0.0:
            raise ValueError(f"r0 must be strictly positive, got {r0}")

        th0 = theta_bar if theta0 is None else float(theta0)
        z0 = zeta_bar if zeta0 is None else float(zeta0)

        if th0 <= 0.0:
            raise ValueError(f"theta0 must be strictly positive, got {th0}")
        if z0 <= 0.0:
            raise ValueError(f"zeta0 must be strictly positive, got {z0}")

        self._k = float(k)
        self._theta_bar = float(theta_bar)
        self._kappa = float(kappa)
        self._alpha = float(alpha)
        self._zeta_bar = float(zeta_bar)
        self._lambda = float(lambda_param)
        self._beta = float(beta)
        self._r0 = float(r0)
        self._theta0 = th0
        self._zeta0 = z0

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"ChenModel(k={self._k}, θ_bar={self._theta_bar}, ζ_bar={self._zeta_bar})",
            math_note="3-factor short rate with stochastic mean and stochastic variance",
            reference="Chen (1996)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def k(self) -> float:
        """Mean-reversion speed of short rate."""
        return self._k

    @property
    def theta_bar(self) -> float:
        """Long-term central tendency level."""
        return self._theta_bar

    @property
    def kappa(self) -> float:
        """Mean-reversion speed of central tendency."""
        return self._kappa

    @property
    def alpha(self) -> float:
        """Volatility parameter of central tendency."""
        return self._alpha

    @property
    def zeta_bar(self) -> float:
        """Long-term variance factor."""
        return self._zeta_bar

    @property
    def lambda_param(self) -> float:
        """Mean-reversion speed of variance factor."""
        return self._lambda

    @property
    def beta(self) -> float:
        """Volatility parameter of variance factor."""
        return self._beta

    @property
    def r0(self) -> float:
        """Initial short rate."""
        return self._r0

    @property
    def theta0(self) -> float:
        """Initial central tendency."""
        return self._theta0

    @property
    def zeta0(self) -> float:
        """Initial variance factor."""
        return self._zeta0

    @property
    def feller_theta(self) -> bool:
        r"""Whether Feller condition :math:`2 \kappa \bar{\theta} \ge \alpha^2` holds."""
        return (2.0 * self._kappa * self._theta_bar) >= (self._alpha ** 2)

    @property
    def feller_zeta(self) -> bool:
        r"""Whether Feller condition :math:`2 \lambda \bar{\zeta} \ge \beta^2` holds."""
        return (2.0 * self._lambda * self._zeta_bar) >= (self._beta ** 2)

    @property
    def stationary_mean_short_rate(self) -> float:
        """Stationary mean of the short rate :math:`\\bar{\\theta}`."""
        return self._theta_bar

    @property
    def stationary_mean_central_tendency(self) -> float:
        """Stationary mean of the central tendency :math:`\\bar{\\theta}`."""
        return self._theta_bar

    @property
    def stationary_mean_volatility(self) -> float:
        """Stationary mean of the variance factor :math:`\\bar{\\zeta}`."""
        return self._zeta_bar

    # ------------------------------------------------------------------
    # Exactness & Triplet
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason=(
                "ChenModel is a coupled three-factor diffusion model with state-dependent "
                "stochastic mean and volatility, violating the stationary and independent "
                "increments condition of pure Levy processes. Exact closed-form expectations, "
                "matrix exponential moments, term structure bond pricing, and coupled factor "
                "simulation are available directly."
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
                "Chen (1996) three-factor interest rate model.",
                f"Feller central tendency: {self.feller_theta}, Feller variance: {self.feller_zeta}.",
            ],
        )

    # ------------------------------------------------------------------
    # Exact Expectations & Matrix Exponential Moments
    # ------------------------------------------------------------------

    def conditional_mean_central_tendency(self, t: float = 1.0) -> float:
        r"""Exact conditional expectation :math:`\mathbb{E}[\theta_t \mid \theta_0]`."""
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        return float(self._theta_bar + (self._theta0 - self._theta_bar) * math.exp(-self._kappa * t))

    def conditional_mean_volatility(self, t: float = 1.0) -> float:
        r"""Exact conditional expectation :math:`\mathbb{E}[\zeta_t \mid \zeta_0]`."""
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        return float(self._zeta_bar + (self._zeta0 - self._zeta_bar) * math.exp(-self._lambda * t))

    def conditional_mean(self, t: float = 1.0) -> float:
        r"""
        Exact conditional expectation :math:`\mathbb{E}[r_t \mid r_0, \theta_0]`.

        .. math::

            \mathbb{E}[r_t] = e^{-k t} r_0 + \bar{\theta}(1 - e^{-k t}) +
            \frac{k(\theta_0 - \bar{\theta})}{\kappa - k} (e^{-k t} - e^{-\kappa t})
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        if t == 0.0:
            return self._r0

        exp_kt = math.exp(-self._k * t)
        base = exp_kt * self._r0 + self._theta_bar * (1.0 - exp_kt)

        if math.isclose(self._k, self._kappa, rel_tol=1e-7):
            correction = self._k * (self._theta0 - self._theta_bar) * t * exp_kt
        else:
            exp_kapt = math.exp(-self._kappa * t)
            correction = (self._k * (self._theta0 - self._theta_bar) / (self._kappa - self._k)) * (exp_kt - exp_kapt)

        return float(base + correction)

    def conditional_moments_system(self, t: float = 1.0) -> Dict[str, float]:
        r"""
        Solve the exact 10-dimensional linear moment ODE system via matrix exponential.

        Returns expectations and second moments:
        ``E[r^2]``, ``E[r*theta]``, ``E[theta^2]``, ``E[r*zeta]``, ``E[theta*zeta]``,
        ``E[zeta^2]``, ``E[r]``, ``E[theta]``, ``E[zeta]``.
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")

        # Vector: [r^2, r*th, th^2, r*zeta, th*zeta, zeta^2, r, th, zeta, 1]
        dim = 10
        a_mat = np.zeros((dim, dim), dtype=float)

        # 0: E[r^2]
        a_mat[0, 0] = -2.0 * self._k
        a_mat[0, 1] = 2.0 * self._k
        a_mat[0, 3] = 1.0

        # 1: E[r*th]
        a_mat[1, 1] = -(self._k + self._kappa)
        a_mat[1, 2] = self._k
        a_mat[1, 6] = self._kappa * self._theta_bar

        # 2: E[th^2]
        a_mat[2, 2] = -2.0 * self._kappa
        a_mat[2, 7] = 2.0 * self._kappa * self._theta_bar + (self._alpha ** 2)

        # 3: E[r*zeta]
        a_mat[3, 3] = -(self._k + self._lambda)
        a_mat[3, 4] = self._k
        a_mat[3, 6] = self._lambda * self._zeta_bar

        # 4: E[th*zeta]
        a_mat[4, 4] = -(self._kappa + self._lambda)
        a_mat[4, 8] = self._kappa * self._theta_bar
        a_mat[4, 7] = self._lambda * self._zeta_bar

        # 5: E[zeta^2]
        a_mat[5, 5] = -2.0 * self._lambda
        a_mat[5, 8] = 2.0 * self._lambda * self._zeta_bar + (self._beta ** 2)

        # 6: E[r]
        a_mat[6, 6] = -self._k
        a_mat[6, 7] = self._k

        # 7: E[th]
        a_mat[7, 7] = -self._kappa
        a_mat[7, 9] = self._kappa * self._theta_bar

        # 8: E[zeta]
        a_mat[8, 8] = -self._lambda
        a_mat[8, 9] = self._lambda * self._zeta_bar

        m0 = np.array([
            self._r0 ** 2,
            self._r0 * self._theta0,
            self._theta0 ** 2,
            self._r0 * self._zeta0,
            self._theta0 * self._zeta0,
            self._zeta0 ** 2,
            self._r0,
            self._theta0,
            self._zeta0,
            1.0,
        ], dtype=float)

        m_t = la.expm(a_mat * t) @ m0

        return {
            "r2": float(m_t[0]),
            "r_theta": float(m_t[1]),
            "theta2": float(m_t[2]),
            "r_zeta": float(m_t[3]),
            "theta_zeta": float(m_t[4]),
            "zeta2": float(m_t[5]),
            "r": float(m_t[6]),
            "theta": float(m_t[7]),
            "zeta": float(m_t[8]),
        }

    def conditional_second_moment(self, t: float = 1.0) -> float:
        """Exact conditional second moment :math:`\\mathbb{E}[r_t^2 \\mid r_0, \\theta_0, \\zeta_0]`."""
        return self.conditional_moments_system(t=t)["r2"]

    def conditional_variance(self, t: float = 1.0) -> float:
        r"""
        Exact conditional variance :math:`\mathrm{Var}(r_t \mid r_0, \theta_0, \zeta_0) = \mathbb{E}[r_t^2] - (\mathbb{E}[r_t])^2`.
        """
        moments = self.conditional_moments_system(t=t)
        return float(max(0.0, moments["r2"] - moments["r"] ** 2))

    def conditional_std(self, t: float = 1.0) -> float:
        """Exact conditional standard deviation."""
        return math.sqrt(self.conditional_variance(t=t))

    # ------------------------------------------------------------------
    # Term Structure Bond Pricing
    # ------------------------------------------------------------------

    def bond_price(
        self,
        maturity: float = 1.0,
        n_steps: int = 200,
        n_paths: int = 15000,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        r"""
        Compute Zero-Coupon Bond Price :math:`P(0, T) = \mathbb{E}\left[\exp\left(-\int_0^T r_s ds\right)\right]`.
        """
        if maturity <= 0.0:
            return 1.0

        r_paths, _, _ = self.simulate_factors(
            n_steps=n_steps, time_horizon=maturity, n_paths=n_paths, rng=rng
        )
        dt = maturity / n_steps
        trap_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
        integral_r = trap_func(r_paths, dx=dt, axis=1)
        return float(np.mean(np.exp(-integral_r)))

    def yield_curve(
        self,
        maturities: Sequence[float],
        n_steps: int = 200,
        n_paths: int = 15000,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""
        Compute Zero-Coupon Yield Curve :math:`y(\tau) = -\frac{\ln P(0, \tau)}{\tau}`.
        """
        mat_arr = np.asarray(maturities, dtype=float)
        yields = np.empty_like(mat_arr)
        for i, tau in enumerate(mat_arr):
            p = self.bond_price(maturity=tau, n_steps=n_steps, n_paths=n_paths, rng=rng)
            yields[i] = -math.log(max(p, 1e-12)) / tau
        return yields

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
        """Simulate sample paths of the short rate :math:`r_t`."""
        from spxa.sim.exact import sample_chen
        return sample_chen(
            k=self._k,
            theta_bar=self._theta_bar,
            kappa=self._kappa,
            alpha=self._alpha,
            zeta_bar=self._zeta_bar,
            lambda_param=self._lambda,
            beta=self._beta,
            r0=self._r0,
            theta0=self._theta0,
            zeta0=self._zeta0,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=time_horizon,
            rng=rng,
            return_factors=False,
        )

    def simulate_factors(
        self,
        n_steps: int = 100,
        dt: Optional[float] = None,
        time_horizon: Optional[float] = None,
        n_paths: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate coupled paths of :math:`(r_t, \\theta_t, \\zeta_t)`."""
        from spxa.sim.exact import sample_chen
        return sample_chen(
            k=self._k,
            theta_bar=self._theta_bar,
            kappa=self._kappa,
            alpha=self._alpha,
            zeta_bar=self._zeta_bar,
            lambda_param=self._lambda,
            beta=self._beta,
            r0=self._r0,
            theta0=self._theta0,
            zeta0=self._zeta0,
            dt=dt,
            n_steps=n_steps,
            n_paths=n_paths,
            T=time_horizon,
            rng=rng,
            return_factors=True,
        )

    def __repr__(self) -> str:
        return (
            f"ChenModel(k={self._k}, θ_bar={self._theta_bar}, κ={self._kappa}, "
            f"α={self._alpha}, ζ_bar={self._zeta_bar}, λ={self._lambda}, β={self._beta}, "
            f"r0={self._r0}, θ0={self._theta0}, ζ0={self._zeta0})"
        )


# Aliases
ChenThreeFactor = ChenModel
ChenShortRate = ChenModel
ChenProcess = ChenModel
