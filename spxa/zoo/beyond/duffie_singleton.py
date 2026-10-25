"""Duffie-Singleton hazard-rate default intensity diffusion model.

Implements the Duffie & Singleton (1999) term structure of defaultable bonds
under the Recovery of Market Value (RMV) convention with square-root (CIR)
stochastic default intensity (hazard rate) lambda_t:

    dlambda_t = kappa * (theta - lambda_t) dt + sigma * sqrt(lambda_t) * dW_t

The default-adjusted discount rate under RMV is:
    R_t = r + (1 - R) * lambda_t = r + L * lambda_t
where R is the fractional recovery rate in default and L = 1 - R is the loss given default.

Survival probability:
    S(0, T) = E[exp(-int_0^T lambda_s ds)] = A(T; 1) * exp(-B(T; 1) * lambda_0)

Defaultable zero-coupon bond price:
    P_def(0, T) = exp(-r * T) * A(T; L) * exp(-B(T; L) * lambda_0)

References:
    Duffie, D. & Singleton, K. J. (1999). Modeling Term Structures of Defaultable
        Bonds. Review of Financial Studies, 12(4), 687-720.
    Lando, D. (1998). On Cox processes and credit risky securities. Review of
        Derivatives Research, 2(2), 99-120.
    Schonbucher, P. J. (2003). Credit Derivatives Modeling. John Wiley & Sons.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy import integrate

from spxa.core.exactness import ExactnessLevel
from spxa.core.exceptions import ExactnessError
from spxa.core.process import Process
from spxa.core.properties import ProcessProperties
from spxa.core.triplet import LevyTriplet


class DuffieSingletonProcess(Process):
    r"""Duffie-Singleton hazard-rate default intensity credit process.

    Parameters
    ----------
    kappa : float
        Mean-reversion rate of the hazard intensity (must be positive).
    theta : float
        Long-term mean hazard intensity (must be positive).
    sigma : float
        Volatility parameter of the hazard intensity (must be positive).
    lambda0 : float
        Initial hazard intensity (must be positive).
    recovery : float
        Fractional recovery rate in default R in [0, 1] (default: 0.40).
    r : float
        Continuously compounded risk-free interest rate (default: 0.03).
    """

    def __init__(
        self,
        kappa: float = 0.8,
        theta: float = 0.03,
        sigma: float = 0.1,
        lambda0: float = 0.02,
        recovery: float = 0.40,
        r: float = 0.03,
    ) -> None:
        if kappa <= 0.0 or theta <= 0.0 or sigma <= 0.0:
            raise ValueError(
                f"Hazard rate parameters kappa, theta, sigma must be positive: "
                f"kappa={kappa}, theta={theta}, sigma={sigma}"
            )
        if lambda0 <= 0.0:
            raise ValueError(f"Initial hazard intensity lambda0 must be positive, got {lambda0}")
        if not (0.0 <= recovery <= 1.0):
            raise ValueError(f"Recovery rate must lie in [0, 1], got {recovery}")

        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.lambda0 = float(lambda0)
        self.recovery = float(recovery)
        self.loss = float(1.0 - recovery)
        self.r = float(r)

        super().__init__(exactness=ExactnessLevel.MOMENT_PROPAGATION)

    @property
    def exactness_level(self) -> ExactnessLevel:
        return ExactnessLevel.MOMENT_PROPAGATION

    def _triplet(self) -> LevyTriplet:
        raise ExactnessError(
            "DuffieSingletonProcess is an affine Cox default intensity process; "
            "it does not possess a stationary Levy triplet."
        )

    def _properties(self) -> ProcessProperties:
        return ProcessProperties(
            has_stationary_increments=False,
            has_independent_increments=False,
            is_martingale=False,
            has_finite_mean=True,
            has_finite_variance=True,
            notes=[
                "Duffie-Singleton default intensity credit model (Duffie & Singleton 1999)",
                f"Recovery = {self.recovery}, Loss = {self.loss}",
                f"Feller ratio 2*kappa*theta / sigma^2 = {self.feller_ratio:.4f}",
            ],
        )

    @property
    def feller_ratio(self) -> float:
        r"""Compute the Feller ratio 2 * kappa * theta / sigma^2."""
        return float(2.0 * self.kappa * self.theta / (self.sigma**2))

    @property
    def satisfies_feller(self) -> bool:
        r"""Check if hazard rate strictly avoids 0: 2 * kappa * theta > sigma^2."""
        return self.feller_ratio > 1.0

    def _cir_laplace_coefficients(self, tau: float, weight: float) -> Tuple[float, float]:
        r"""Compute (A(tau), B(tau)) such that E[exp(-weight * \int_0^tau lambda_s ds)] = A * exp(-B * lambda_0)."""
        if tau <= 0.0 or weight <= 0.0:
            return 1.0, 0.0

        k = self.kappa
        th = self.theta
        sig = self.sigma

        gamma = math.sqrt(k**2 + 2.0 * weight * (sig**2))
        exp_gt = math.exp(gamma * tau)
        denom = (gamma + k) * (exp_gt - 1.0) + 2.0 * gamma

        b_val = 2.0 * weight * (exp_gt - 1.0) / denom
        a_base = 2.0 * gamma * math.exp(0.5 * (k + gamma) * tau) / denom
        a_val = a_base ** (2.0 * k * th / (sig**2))

        return float(a_val), float(b_val)

    def survival_probability(self, tau: float) -> float:
        r"""Exact survival probability S(0, tau) = E[exp(-\int_0^tau lambda_s ds)]."""
        if tau <= 0.0:
            return 1.0
        a_val, b_val = self._cir_laplace_coefficients(tau, weight=1.0)
        return float(a_val * math.exp(-b_val * self.lambda0))

    def default_probability(self, tau: float) -> float:
        r"""Exact cumulative default probability: P(tau_default <= tau) = 1 - S(0, tau)."""
        if tau <= 0.0:
            return 0.0
        return float(max(0.0, 1.0 - self.survival_probability(tau)))

    def default_density(self, tau: float) -> float:
        r"""Exact default arrival probability density: f(tau) = -d/dtau S(0, tau)."""
        if tau <= 0.0:
            return self.lambda0
        # High precision central difference
        h = max(1e-5, 1e-4 * tau)
        s_plus = self.survival_probability(tau + h)
        s_minus = self.survival_probability(tau - h)
        return float(max(0.0, (s_minus - s_plus) / (2.0 * h)))

    def defaultable_bond(self, tau: float) -> float:
        r"""Defaultable zero-coupon bond price P_def(0, tau) under RMV:

        P_def(0, tau) = exp(-r * tau) * E[exp(-L * \int_0^tau lambda_s ds)].
        """
        if tau <= 0.0:
            return 1.0
        a_val, b_val = self._cir_laplace_coefficients(tau, weight=self.loss)
        credit_part = a_val * math.exp(-b_val * self.lambda0)
        risk_free_part = math.exp(-self.r * tau)
        return float(risk_free_part * credit_part)

    def credit_spread(self, tau: float) -> float:
        r"""Zero-coupon credit spread s(tau) = -ln(P_def(0, tau)) / tau - r."""
        if tau <= 0.0:
            return self.loss * self.lambda0
        p_def = self.defaultable_bond(tau)
        yield_def = -math.log(p_def) / tau
        return float(max(0.0, yield_def - self.r))

    def cds_par_spread(self, tau: float, n_points: int = 100) -> float:
        r"""Par Credit Default Swap (CDS) spread s_CDS(T).

        s_CDS(T) = (1 - R) * \int_0^T exp(-r*t) f(t) dt / \int_0^T exp(-r*t) S(0, t) dt.
        """
        if tau <= 0.0:
            return self.loss * self.lambda0

        t_grid = np.linspace(0.0, tau, n_points)

        protection_integrand = np.empty(n_points)
        premium_integrand = np.empty(n_points)

        for i, t_val in enumerate(t_grid):
            disc = math.exp(-self.r * t_val)
            protection_integrand[i] = disc * self.default_density(t_val)
            premium_integrand[i] = disc * self.survival_probability(t_val)

        prot_leg = self.loss * float(integrate.trapezoid(protection_integrand, t_grid))
        prem_leg = float(integrate.trapezoid(premium_integrand, t_grid))

        if prem_leg <= 1e-12:
            return self.loss * self.lambda0
        return float(prot_leg / prem_leg)

    def mean_hazard_rate(self, t: float) -> float:
        r"""Exact expected instantaneous hazard rate: E[lambda_t | lambda_0]."""
        if t <= 0.0:
            return self.lambda0
        return float(self.theta + (self.lambda0 - self.theta) * math.exp(-self.kappa * t))

    def variance_hazard_rate(self, t: float) -> float:
        r"""Exact conditional variance of hazard rate: Var[lambda_t | lambda_0]."""
        if t <= 0.0:
            return 0.0
        k = self.kappa
        th = self.theta
        sig2 = self.sigma**2
        e_kt = math.exp(-k * t)
        term1 = self.lambda0 * (sig2 / k) * (e_kt - e_kt**2)
        term2 = th * (sig2 / (2.0 * k)) * (1.0 - e_kt) ** 2
        return float(term1 + term2)

    def stationary_hazard_mean(self) -> float:
        r"""Long-term mean hazard rate: theta."""
        return self.theta

    def stationary_hazard_var(self) -> float:
        r"""Long-term variance of hazard rate: theta * sigma^2 / (2 * kappa)."""
        return float(self.theta * (self.sigma**2) / (2.0 * self.kappa))

    def simulate(
        self,
        t_max: float = 1.0,
        num_steps: int = 100,
        num_paths: int = 1000,
        seed: Optional[int] = None,
        return_default_times: bool = False,
    ) -> np.ndarray:
        r"""Simulate hazard intensity lambda_t paths using exact non-central chi-square transitions.

        Parameters
        ----------
        t_max : float
            Simulation time horizon.
        num_steps : int
            Number of discrete time steps.
        num_paths : int
            Number of sample paths.
        seed : int, optional
            RNG seed.
        return_default_times : bool
            If True, returns tuple (lambda_paths, default_times). Otherwise returns lambda_paths.
        """
        if t_max <= 0.0 or num_steps <= 0 or num_paths <= 0:
            raise ValueError("t_max, num_steps, and num_paths must be strictly positive.")

        rng = np.random.default_rng(seed)
        dt = t_max / num_steps

        lambda_paths = np.empty((num_paths, num_steps + 1), dtype=float)
        lambda_paths[:, 0] = self.lambda0

        k = self.kappa
        th = self.theta
        sig = self.sigma
        d = 4.0 * k * th / (sig**2)
        c = (sig**2) * (1.0 - math.exp(-k * dt)) / (4.0 * k)
        e_kdt = math.exp(-k * dt)

        # For default time generation
        # Integrated hazard Lambda_t = \int_0^t lambda_s ds
        # Generate unit exponential thresholds
        default_thresholds = rng.exponential(1.0, size=num_paths)
        default_times = np.full(num_paths, np.nan)
        cum_hazard = np.zeros(num_paths)

        for step in range(num_steps):
            lambda_curr = lambda_paths[:, step]
            # Exact non-central chi-square sampling
            nc = lambda_curr * e_kdt / c
            chi2_samples = rng.noncentral_chisquare(df=d, nonc=nc)
            lambda_next = c * chi2_samples
            lambda_paths[:, step + 1] = lambda_next

            # Trapezoidal integration of hazard rate
            dhazard = 0.5 * (lambda_curr + lambda_next) * dt
            new_cum = cum_hazard + dhazard

            # Check for defaults during this step
            defaulted_mask = (cum_hazard < default_thresholds) & (new_cum >= default_thresholds)
            if np.any(defaulted_mask):
                # Linear interpolation for default time within step
                t_step_start = step * dt
                frac = (default_thresholds[defaulted_mask] - cum_hazard[defaulted_mask]) / dhazard[
                    defaulted_mask
                ]
                default_times[defaulted_mask] = t_step_start + frac * dt

            cum_hazard = new_cum

        if return_default_times:
            return lambda_paths, default_times
        return lambda_paths


DuffieSingleton = DuffieSingletonProcess
DefaultIntensityDiffusion = DuffieSingletonProcess
HazardRateDiffusion = DuffieSingletonProcess
