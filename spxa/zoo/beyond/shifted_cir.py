"""Shifted Cox-Ingersoll-Ross (CIR++) Short-Rate Process.

The CIR++ model by Brigo & Mercurio (2001) shifts a standard CIR diffusion by a deterministic
function phi(t) to perfectly calibrate to market yield curves while retaining analytical
bond and option pricing tractability:
    r(t) = x(t) + phi(t)
    dx(t) = k * (theta - x(t)) dt + sigma * sqrt(x(t)) dW(t)

References:
    Brigo, D., & Mercurio, F. (2001). A deterministic-shift extension of analytically-tractable
        and positive short-rate models. Finance and Stochastics, 5(3), 369-388.
    Brigo, D., & Mercurio, F. (2006). Interest Rate Models - Theory and Practice. Springer, Sec. 3.8.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy import stats

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties


class ShiftedCIRProcess(Process):
    r"""Shifted Cox-Ingersoll-Ross (CIR++) short-rate process.

    .. math::
        r_t = x_t + \phi(t)
        dx_t = \kappa (\theta - x_t) dt + \sigma \sqrt{x_t} dW_t

    where :math:`\phi(t) = \phi_0 + \phi_1 e^{-\alpha t}` is an exponentially decaying deterministic shift.
    """

    def __init__(
        self,
        k: float,
        theta: float,
        sigma: float,
        x0: float = 0.03,
        phi0: float = 0.01,
        phi1: float = 0.0,
        alpha: float = 0.0,
    ) -> None:
        if k <= 0 or theta <= 0 or sigma <= 0:
            raise ValueError(f"Parameters must be positive: k={k}, theta={theta}, sigma={sigma}")
        if x0 < 0:
            raise ValueError(f"Initial factor x0 must be non-negative: x0={x0}")

        self.k = float(k)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.x0 = float(x0)
        self.phi0 = float(phi0)
        self.phi1 = float(phi1)
        self.alpha = float(alpha)
        self.r0 = self.x0 + self.phi(0.0)
        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_variance=True,
            has_finite_mean=True,
            notes=["Shifted CIR short-rate process (CIR++)"],
        )

    def _triplet(self):
        raise ExactnessError(
            "ShiftedCIRProcess has state-dependent diffusion coefficients and cannot be represented by a Levy triplet."
        )

    @property
    def feller_ratio(self) -> float:
        """Feller condition ratio 2*k*theta / sigma^2."""
        return 2.0 * self.k * self.theta / (self.sigma**2)

    def phi(self, t: float) -> float:
        """Deterministic shift phi(t) = phi0 + phi1 * exp(-alpha * t)."""
        if self.alpha == 0.0 or self.phi1 == 0.0:
            return self.phi0 + self.phi1
        return self.phi0 + self.phi1 * math.exp(-self.alpha * t)

    def phi_integral(self, t: float) -> float:
        r"""Integral \int_0^t phi(s) ds."""
        if t <= 0.0:
            return 0.0
        term0 = self.phi0 * t
        if self.alpha == 0.0 or self.phi1 == 0.0:
            term1 = self.phi1 * t
        else:
            term1 = self.phi1 * (1.0 - math.exp(-self.alpha * t)) / self.alpha
        return term0 + term1

    def mean(self, t: float) -> float:
        """Conditional expected short rate E[r(t) | r(0)]."""
        x_mean = self.theta + (self.x0 - self.theta) * math.exp(-self.k * t)
        return x_mean + self.phi(t)

    def variance(self, t: float) -> float:
        """Conditional variance Var[r(t) | r(0)] = Var[x(t) | x(0)]."""
        e_kt = math.exp(-self.k * t)
        term1 = self.x0 * (self.sigma**2 / self.k) * (e_kt - e_kt**2)
        term2 = self.theta * (self.sigma**2 / (2.0 * self.k)) * ((1.0 - e_kt) ** 2)
        return term1 + term2

    def _cir_bond_coeffs(self, tau: float) -> tuple[float, float]:
        """Compute A(tau), B(tau) for the underlying CIR factor."""
        gamma = math.sqrt(self.k**2 + 2.0 * self.sigma**2)
        exp_gt = math.exp(gamma * tau)
        denom = (gamma + self.k) * (exp_gt - 1.0) + 2.0 * gamma
        b = 2.0 * (exp_gt - 1.0) / denom
        num_a = 2.0 * gamma * math.exp((self.k + gamma) * tau / 2.0)
        power = 2.0 * self.k * self.theta / (self.sigma**2)
        a = (num_a / denom) ** power
        return a, b

    def zero_coupon_bond(self, tau: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Closed-form zero-coupon bond price P(0, tau)."""
        is_scalar = np.isscalar(tau)
        taus = np.atleast_1d(tau).astype(float)
        prices = np.zeros_like(taus)

        for i, t in enumerate(taus):
            if t <= 0.0:
                prices[i] = 1.0
                continue
            a, b = self._cir_bond_coeffs(t)
            shift_int = self.phi_integral(t)
            prices[i] = a * math.exp(-b * self.x0 - shift_int)

        return float(prices[0]) if is_scalar else prices

    def yield_curve(self, tau: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Zero-coupon yield curve y(0, tau) = -ln(P(0, tau)) / tau."""
        is_scalar = np.isscalar(tau)
        taus = np.atleast_1d(tau).astype(float)
        p = np.atleast_1d(self.zero_coupon_bond(taus))
        y = np.zeros_like(taus)
        pos = taus > 0.0
        y[pos] = -np.log(p[pos]) / taus[pos]
        y[~pos] = self.r0
        return float(y[0]) if is_scalar else y

    def european_bond_call(self, maturity: float, bond_maturity: float, strike: float) -> float:
        r"""European call option on zero-coupon bond P(t, T_bond) with strike K and maturity T_opt.

        Uses the exact non-central chi-square representation of Jamshidian / Brigo-Mercurio.
        """
        if maturity >= bond_maturity:
            raise ValueError(
                f"Option maturity {maturity} must be less than bond maturity {bond_maturity}"
            )

        tau_bond = bond_maturity - maturity
        a_tau, b_tau = self._cir_bond_coeffs(tau_bond)
        int_phi = self.phi_integral(bond_maturity) - self.phi_integral(maturity)

        # Strike threshold for x(T_opt)
        # K = a_tau * exp(-b_tau * x* - int_phi)
        ratio = strike / (a_tau * math.exp(-int_phi))
        if ratio <= 0.0:
            return float(
                self.zero_coupon_bond(bond_maturity) - strike * self.zero_coupon_bond(maturity)
            )
        x_star = -math.log(ratio) / b_tau
        if x_star <= 0.0:
            # Since r_t >= 0, r_t < x_star is impossible; option expires worthless
            return 0.0

        gamma = math.sqrt(self.k**2 + 2.0 * self.sigma**2)
        exp_gt = math.exp(gamma * maturity)
        d = 4.0 * self.k * self.theta / (self.sigma**2)

        rho = 2.0 * gamma / (self.sigma**2 * (exp_gt - 1.0))
        psi = (self.k + gamma) / self.sigma**2

        # Chi-squared CDF values
        df = d
        nc1 = 2.0 * (rho**2) * self.x0 * math.exp(gamma * maturity) / (rho + psi)
        chi1 = stats.ncx2.cdf(2.0 * x_star * (rho + psi), df, nc1)

        p_t = self.zero_coupon_bond(maturity)
        p_s = self.zero_coupon_bond(bond_maturity)

        # Brigo-Mercurio formula Call = P(0, S) * chi2_1 - K * P(0, T) * chi2_2
        nc2 = 2.0 * (rho**2) * self.x0 * math.exp(gamma * maturity) / (rho + psi + b_tau)
        chi2 = stats.ncx2.cdf(2.0 * x_star * (rho + psi + b_tau), df, nc2)

        call_price = p_s * chi2 - strike * p_t * chi1
        return float(max(0.0, call_price))

    def simulate(
        self,
        t_max: Optional[float] = None,
        num_steps: Optional[int] = None,
        num_paths: int = 1,
        seed: Optional[int] = None,
        T: Optional[float] = None,
        n_steps: Optional[int] = None,
        n_paths: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate short-rate paths with exact non-central chi-square transitions for x(t)."""
        if seed is not None:
            np.random.seed(seed)

        final_t = t_max if t_max is not None else (T if T is not None else 1.0)
        steps = num_steps if num_steps is not None else (n_steps if n_steps is not None else 100)
        paths_count = n_paths if n_paths is not None else num_paths

        dt = float(final_t) / steps
        t_grid = np.linspace(0.0, final_t, steps + 1)
        x_paths = np.zeros((paths_count, steps + 1))
        x_paths[:, 0] = self.x0

        d = 4.0 * self.k * self.theta / (self.sigma**2)
        c = (self.sigma**2 / (4.0 * self.k)) * (1.0 - math.exp(-self.k * dt))
        exp_kdt = math.exp(-self.k * dt)

        for step in range(steps):
            x_prev = x_paths[:, step]
            lambda_param = (4.0 * self.k * x_prev * exp_kdt) / (self.sigma**2 * (1.0 - exp_kdt))
            x_next = c * np.random.noncentral_chisquare(df=d, nonc=np.maximum(0.0, lambda_param))
            x_paths[:, step + 1] = np.maximum(0.0, x_next)

        phi_vals = np.array([self.phi(t) for t in t_grid])
        return x_paths + phi_vals


# Aliases
ShiftedCIR = ShiftedCIRProcess
CIRPlusPlus = ShiftedCIRProcess
