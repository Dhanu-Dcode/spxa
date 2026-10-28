"""
Jump-Extended Cox-Ingersoll-Ross (JCIR) Short Rate Process.

Introduced by Duffie & Garleanu (2001) and Maghsoodi (1996), and widely used
in sovereign credit and interest-rate modeling with positive jump shocks
(Brigo & Mercurio, 2006).

SDE:
    dr_t = κ (θ - r_t) dt + σ √r_t dW_t + dJ_t,   r_0 ≥ 0

where:
    - κ > 0: speed of mean reversion
    - θ > 0: baseline diffusion long-term mean
    - σ > 0: volatility-of-rate
    - J_t = ∑_{i=1}^{N_t} Y_i: compound Poisson process with rate λ_J ≥ 0
      and exponential jump sizes Y_i ~ Exp(mean=μ_J) with μ_J > 0
    - W_t: standard 1D Brownian motion independent of J_t

Features:
    - Strictly non-negative interest rates r_t ≥ 0.
    - Explicit stationary mean θ_∞ = θ + (λ_J μ_J) / κ.
    - Explicit stationary variance Var(r_∞) = (σ² θ_∞ + 2 λ_J μ_J²) / (2κ).
    - Exact conditional mean and conditional variance at any horizon t.
    - Closed-form exponential-affine zero-coupon bond pricing P(t, T) = exp(A(τ) - B(τ) r_t).
    - Closed-form forward rate f(t, T) and term structure of yields.
    - Exact conditional characteristic function via affine Riccati integration.
    - Exact non-central chi-squared + compound Poisson jump simulation.

References
----------
Duffie, D., & Garleanu, N. (2001). Risk and valuation of collateralized debt obligations.
    Financial Analysts Journal, 57(1), 41–59.
Maghsoodi, Y. (1996). Solution of the extended CIR term structure and bond option valuation.
    Mathematical Finance, 6(1), 89–109.
Brigo, D., & Mercurio, F. (2006). Interest Rate Models - Theory and Practice.
    Springer Finance, Section 3.9 & 3.10.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy import integrate  # type: ignore[import-untyped]

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet
from spxa.story.narrator import CompositionNode, NodeKind


class JCIRProcess(Process):
    r"""
    Jump-Extended Cox-Ingersoll-Ross (JCIR) process.

    .. math::

        dr_t = \kappa (\theta - r_t) dt + \sigma \sqrt{r_t} dW_t + dJ_t, \quad r_0 \ge 0

    where :math:`J_t = \sum_{i=1}^{N_t} Y_i` with arrival rate :math:`\lambda_J \ge 0`
    and exponential jump sizes :math:`Y_i \sim \mathrm{Exp}(\mu_J)`.

    Parameters
    ----------
    kappa : float
        Mean reversion speed :math:`\kappa > 0`. Default 1.0.
    theta : float
        Diffusion long-term mean :math:`\theta > 0`. Default 0.05.
    sigma : float
        Volatility-of-rate :math:`\sigma > 0`. Default 0.15.
    lambda_j : float
        Jump arrival intensity :math:`\lambda_J \ge 0`. Default 0.5.
    mu_j : float
        Mean jump size :math:`\mu_J > 0`. Default 0.02.
    r0 : float
        Initial short rate :math:`r_0 \ge 0`. Default 0.04.
    """

    def __init__(
        self,
        kappa: float = 1.0,
        theta: float = 0.05,
        sigma: float = 0.15,
        lambda_j: float = 0.5,
        mu_j: float = 0.02,
        r0: float = 0.04,
        jump_intensity: Optional[float] = None,
        jump_mean: Optional[float] = None,
    ) -> None:
        if kappa <= 0.0:
            raise ValueError(f"Mean reversion speed 'kappa' must be strictly positive, got {kappa}")
        if theta <= 0.0:
            raise ValueError(f"Long-term mean 'theta' must be strictly positive, got {theta}")
        if sigma <= 0.0:
            raise ValueError(f"Volatility 'sigma' must be strictly positive, got {sigma}")

        lam = jump_intensity if jump_intensity is not None else lambda_j
        if lam < 0.0:
            raise ValueError(f"Jump intensity 'lambda_j' must be non-negative, got {lam}")

        mj = jump_mean if jump_mean is not None else mu_j
        if mj <= 0.0:
            raise ValueError(f"Mean jump size 'mu_j' must be strictly positive, got {mj}")

        if r0 < 0.0:
            raise ValueError(f"Initial short rate 'r0' must be non-negative, got {r0}")

        self._kappa = float(kappa)
        self._theta = float(theta)
        self._sigma = float(sigma)
        self._lambda_j = float(lam)
        self._mu_j = float(mj)
        self._r0 = float(r0)

        # Derived parameters
        self._gamma = math.sqrt(self._kappa**2 + 2.0 * self._sigma**2)
        self._stationary_mean = self._theta + (self._lambda_j * self._mu_j) / self._kappa
        self._stationary_variance = (
            (self._sigma**2) * self._stationary_mean + 2.0 * self._lambda_j * (self._mu_j**2)
        ) / (2.0 * self._kappa)

        node = CompositionNode(
            kind=NodeKind.PRIMITIVE,
            name=f"JCIR(κ={self._kappa}, θ={self._theta}, σ={self._sigma}, λ_j={self._lambda_j}, μ_j={self._mu_j})",
            math_note=(
                f"dr_t = {self._kappa:.2f}({self._theta:.2f} - r_t)dt + {self._sigma:.2f}√r_t dW_t + dJ_t, "
                f"θ_∞={self._stationary_mean:.4f}"
            ),
            reference="Duffie & Garleanu (2001); Brigo & Mercurio (2006)",
        )
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION, _node=node)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def kappa(self) -> float:
        r"""Mean reversion speed :math:`\kappa > 0`."""
        return self._kappa

    @property
    def theta(self) -> float:
        r"""Diffusion long-term mean :math:`\theta > 0`."""
        return self._theta

    @property
    def sigma(self) -> float:
        r"""Volatility-of-rate :math:`\sigma > 0`."""
        return self._sigma

    @property
    def lambda_j(self) -> float:
        r"""Jump arrival intensity :math:`\lambda_J \ge 0`."""
        return self._lambda_j

    @property
    def mu_j(self) -> float:
        r"""Mean jump size :math:`\mu_J > 0`."""
        return self._mu_j

    @property
    def r0(self) -> float:
        r"""Initial short rate :math:`r_0 \ge 0`."""
        return self._r0

    @property
    def gamma(self) -> float:
        r"""Auxiliary parameter :math:`\gamma = \sqrt{\kappa^2 + 2\sigma^2}`."""
        return self._gamma

    @property
    def feller_ratio(self) -> float:
        r"""Feller ratio :math:`\frac{2\kappa\theta}{\sigma^2}` for the baseline CIR diffusion."""
        return (2.0 * self._kappa * self._theta) / (self._sigma**2)

    @property
    def feller_satisfied(self) -> bool:
        r"""Whether baseline diffusion satisfies Feller's condition :math:`2\kappa\theta \ge \sigma^2`."""
        return bool(self.feller_ratio >= 1.0)

    @property
    def stationary_mean(self) -> float:
        r"""Long-term stationary expectation :math:`\theta_\infty = \theta + \frac{\lambda_J \mu_J}{\kappa}`."""
        return self._stationary_mean

    @property
    def stationary_variance(self) -> float:
        r"""Long-term stationary variance :math:`\mathrm{Var}(r_\infty)`."""
        return self._stationary_variance

    @property
    def stationary_std(self) -> float:
        """Long-term stationary standard deviation."""
        return math.sqrt(self._stationary_variance)

    # ------------------------------------------------------------------
    # Exactness & Triplet
    # ------------------------------------------------------------------

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            method="triplet",
            required="EXACT",
            actual="MOMENT_PROPAGATION",
            reason=(
                "JCIRProcess has state-dependent diffusion sigma*sqrt(r_t) and serially correlated "
                "increments, violating the independent and stationary increments property of pure Levy "
                "processes. Conditional moments, stationary moments, exact zero-coupon bond pricing, "
                "characteristic function, and path simulation are available directly."
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
                f"Jump-Extended CIR process (JCIR) with κ={self._kappa}, θ={self._theta}, σ={self._sigma}.",
                f"Jump shocks: λ_j={self._lambda_j}, μ_j={self._mu_j}. Stationary mean: {self._stationary_mean:.4f}.",
            ],
        )

    # ------------------------------------------------------------------
    # Conditional Moments & Analytics
    # ------------------------------------------------------------------

    def conditional_mean(self, t: float = 1.0) -> float:
        r"""
        Exact conditional mean :math:`\mathbb{E}[r_t \mid r_0]`.

        .. math::

            \mathbb{E}[r_t \mid r_0] = r_0 e^{-\kappa t} + \theta_\infty (1 - e^{-\kappa t})
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        exp_kt = math.exp(-self._kappa * t)
        return float(self._r0 * exp_kt + self._stationary_mean * (1.0 - exp_kt))

    def conditional_variance(self, t: float = 1.0) -> float:
        r"""
        Exact conditional variance :math:`\mathrm{Var}(r_t \mid r_0)`.

        .. math::

            \mathrm{Var}(r_t \mid r_0) = \frac{\sigma^2 (r_0 - \theta_\infty)}{\kappa} (e^{-\kappa t} - e^{-2\kappa t})
                                          + \mathrm{Var}(r_\infty) (1 - e^{-2\kappa t})
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        if t == 0.0:
            return 0.0

        exp_1 = math.exp(-self._kappa * t)
        exp_2 = math.exp(-2.0 * self._kappa * t)
        term1 = (self._sigma**2 * (self._r0 - self._stationary_mean) / self._kappa) * (
            exp_1 - exp_2
        )
        term2 = self._stationary_variance * (1.0 - exp_2)
        return float(max(0.0, term1 + term2))

    def conditional_std(self, t: float = 1.0) -> float:
        """Exact conditional standard deviation."""
        return math.sqrt(self.conditional_variance(t))

    # ------------------------------------------------------------------
    # Term Structure & Zero-Coupon Bond Pricing
    # ------------------------------------------------------------------

    def _cir_b(self, tau: float) -> float:
        r"""Riccati coefficient :math:`B(\tau)`."""
        if tau <= 0.0:
            return 0.0
        u = math.exp(self._gamma * tau) - 1.0
        denom = (self._gamma + self._kappa) * u + 2.0 * self._gamma
        return 2.0 * u / denom

    def _cir_a(self, tau: float) -> float:
        r"""Standard CIR Riccati constant :math:`A_{\mathrm{CIR}}(\tau)`."""
        if tau <= 0.0:
            return 0.0
        u = math.exp(self._gamma * tau) - 1.0
        denom = (self._gamma + self._kappa) * u + 2.0 * self._gamma
        numer = 2.0 * self._gamma * math.exp((self._kappa + self._gamma) * tau / 2.0)
        return (2.0 * self._kappa * self._theta / (self._sigma**2)) * math.log(numer / denom)

    def _jump_a(self, tau: float) -> float:
        r"""Jump component :math:`A_{\mathrm{jump}}(\tau)` to Riccati coefficient."""
        if tau <= 0.0 or self._lambda_j == 0.0:
            return 0.0

        c1 = self._gamma + self._kappa + 2.0 * self._mu_j
        d = self._gamma - self._kappa - 2.0 * self._mu_j

        if abs(d) < 1e-12:
            integral_val = (2.0 * self._mu_j / c1) * (
                tau - (1.0 - math.exp(-self._gamma * tau)) / self._gamma
            )
        else:
            z_tau = math.exp(self._gamma * tau)
            f_tau = (-self._gamma * tau / d) + (2.0 * self._gamma / (d * c1)) * math.log(
                c1 * z_tau + d
            )
            f_0 = (2.0 * self._gamma / (d * c1)) * math.log(c1 + d)
            integral_val = (2.0 * self._mu_j / self._gamma) * (f_tau - f_0)

        return -self._lambda_j * integral_val

    def bond_price(
        self, tau: Union[float, np.ndarray], r: Optional[Union[float, np.ndarray]] = None
    ) -> Union[float, np.ndarray]:
        r"""
        Exact Zero-Coupon Bond price :math:`P(t, t+\tau) = \mathbb{E}\left[\exp\left(-\int_t^{t+\tau} r_s ds\right) \mid r_t = r\right]`.

        .. math::

            P(t, t+\tau) = \exp(A(\tau) - B(\tau) r_t)

        where :math:`A(\tau) = A_{\mathrm{CIR}}(\tau) + A_{\mathrm{jump}}(\tau)`.
        """
        tau_arr = np.asarray(tau, dtype=float)
        r_val = self._r0 if r is None else r
        r_arr = np.asarray(r_val, dtype=float)

        def _single_bond_price(t_val: float, rate_val: float) -> float:
            if t_val <= 0.0:
                return 1.0
            b_val = self._cir_b(t_val)
            a_val = self._cir_a(t_val) + self._jump_a(t_val)
            return float(math.exp(a_val - b_val * rate_val))

        vec_func = np.vectorize(_single_bond_price)
        res = vec_func(tau_arr, r_arr)
        if tau_arr.ndim == 0 and r_arr.ndim == 0:
            return float(res)
        return res

    def yield_curve(
        self, tau: Union[float, np.ndarray], r: Optional[Union[float, np.ndarray]] = None
    ) -> Union[float, np.ndarray]:
        r"""
        Zero-coupon yield :math:`y(t, t+\tau) = -\frac{\ln P(t, t+\tau)}{\tau}`.
        """
        tau_arr = np.asarray(tau, dtype=float)
        p = self.bond_price(tau=tau_arr, r=r)

        # Avoid division by zero at tau = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(tau_arr > 0.0, -np.log(p) / tau_arr, self._r0 if r is None else r)
        if tau_arr.ndim == 0:
            return float(y)
        return y

    def forward_rate(
        self, tau: Union[float, np.ndarray], r: Optional[Union[float, np.ndarray]] = None
    ) -> Union[float, np.ndarray]:
        r"""
        Instantaneous forward rate :math:`f(t, t+\tau) = -\frac{\partial \ln P(t, t+\tau)}{\partial \tau}`.

        .. math::

            f(t, t+\tau) = B'(\tau) r_t + \kappa \theta B(\tau) + \frac{\lambda_J \mu_J B(\tau)}{1 + \mu_J B(\tau)}
        """
        tau_arr = np.asarray(tau, dtype=float)
        r_val = self._r0 if r is None else r
        r_arr = np.asarray(r_val, dtype=float)

        def _single_forward(t_val: float, rate_val: float) -> float:
            if t_val <= 0.0:
                return float(rate_val)
            b = self._cir_b(t_val)
            b_prime = 1.0 - self._kappa * b - 0.5 * (self._sigma**2) * (b**2)
            jump_term = (self._lambda_j * self._mu_j * b) / (1.0 + self._mu_j * b)
            return float(b_prime * rate_val + self._kappa * self._theta * b + jump_term)

        vec_func = np.vectorize(_single_forward)
        res = vec_func(tau_arr, r_arr)
        if tau_arr.ndim == 0 and r_arr.ndim == 0:
            return float(res)
        return res

    # ------------------------------------------------------------------
    # Characteristic Function
    # ------------------------------------------------------------------

    def char_func(self, u: Union[float, complex, np.ndarray], t: float = 1.0) -> np.ndarray:
        r"""
        Exact conditional characteristic function :math:`\phi(u; t) = \mathbb{E}[e^{i u r_t} \mid r_0]`.
        """
        if t < 0.0:
            raise ValueError(f"Time t must be non-negative, got {t}")
        u_arr = np.asarray(u, dtype=complex)
        if t == 0.0:
            res = np.exp(1j * u_arr * self._r0)
            return res if res.ndim > 0 else res.item()

        c = 2.0 * self._kappa / (self._sigma**2 * (1.0 - math.exp(-self._kappa * t)))
        df = 4.0 * self._kappa * self._theta / (self._sigma**2)
        exp_kt = math.exp(-self._kappa * t)

        def _eval_single(u_val: complex) -> complex:
            denom = 1.0 - 1j * u_val / c
            phi_cir = (denom ** (-df / 2.0)) * np.exp(1j * u_val * self._r0 * exp_kt / denom)

            if self._lambda_j == 0.0:
                return complex(phi_cir)

            # Jump contribution integration
            def int_r(s: float) -> float:
                c_s = 2.0 * self._kappa / (self._sigma**2 * (1.0 - math.exp(-self._kappa * s)))
                beta_s = 1j * u_val * math.exp(-self._kappa * s) / (1.0 - 1j * u_val / c_s)
                val = self._mu_j * beta_s / (1.0 - self._mu_j * beta_s)
                return float(val.real)

            def int_i(s: float) -> float:
                c_s = 2.0 * self._kappa / (self._sigma**2 * (1.0 - math.exp(-self._kappa * s)))
                beta_s = 1j * u_val * math.exp(-self._kappa * s) / (1.0 - 1j * u_val / c_s)
                val = self._mu_j * beta_s / (1.0 - self._mu_j * beta_s)
                return float(val.imag)

            ir, _ = integrate.quad(int_r, 0.0, t)
            ii, _ = integrate.quad(int_i, 0.0, t)
            jump_factor = np.exp(self._lambda_j * (ir + 1j * ii))
            return complex(phi_cir * jump_factor)

        vec_eval = np.vectorize(_eval_single)
        res = vec_eval(u_arr)
        return res if res.ndim > 0 else res.item()

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
        T: Optional[float] = None,
        *,
        num_steps: Optional[int] = None,
        num_paths: Optional[int] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        r"""
        Simulate sample paths of :math:`r_t` using exact non-central chi-squared CIR transitions
        coupled with compound Poisson exponential jump arrivals.
        """
        from spxa.sim.exact import sample_jcir

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

        return sample_jcir(
            kappa=self._kappa,
            theta=self._theta,
            sigma=self._sigma,
            lambda_j=self._lambda_j,
            mu_j=self._mu_j,
            r0=self._r0,
            dt=dt,
            n_steps=final_steps,
            n_paths=final_paths,
            T=final_t,
            rng=rng,
        )

    def __repr__(self) -> str:
        return (
            f"JCIRProcess(kappa={self._kappa}, theta={self._theta}, sigma={self._sigma}, "
            f"lambda_j={self._lambda_j}, mu_j={self._mu_j}, r0={self._r0})"
        )


# Aliases
JumpCIRProcess = JCIRProcess
JCIRModel = JCIRProcess
JCIR = JCIRProcess
