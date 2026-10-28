r"""SABR Stochastic Volatility Model Surface Calibrator.

Calibrates the Hagan et al. (2002) SABR model parameters:
- alpha : initial stochastic volatility level (> 0)
- beta  : CEV elasticity parameter in [0, 1] (can be fixed or freely calibrated)
- rho   : correlation between forward price and volatility in (-1, 1)
- nu    : volatility of volatility (> 0)

Includes exact analytical inversion for initial alpha from market ATM volatility.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from spxa.calibration.surface.base import SurfaceCalibrator
from spxa.calibration.surface.data import MarketSurface
from spxa.calibration.surface.loss import SurfaceLoss
from spxa.calibration.surface.optimizer import SurfaceOptimizer


def sabr_hagan_implied_vol(
    f: float,
    k: float,
    t: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    r"""Compute Black-76 implied volatility using the Hagan et al. (2002) asymptotic expansion."""
    if k <= 0.0 or f <= 0.0 or t <= 0.0:
        return 0.0
    if alpha <= 0.0:
        return 0.0

    one_minus_beta = 1.0 - beta
    fk = f * k
    fk_beta = fk ** (one_minus_beta / 2.0)

    # ATM case: K == F
    if abs(f - k) < 1e-6 * f:
        term1 = alpha / (f**one_minus_beta)
        term2 = (
            (one_minus_beta**2 / 24.0) * (alpha**2 / (f ** (2.0 * one_minus_beta)))
            + 0.25 * (rho * beta * nu * alpha / (f**one_minus_beta))
            + ((2.0 - 3.0 * rho**2) / 24.0) * (nu**2)
        )
        return float(term1 * (1.0 + term2 * t))

    # OTM / ITM case
    log_fk = math.log(f / k)
    z = (nu / alpha) * fk_beta * log_fk

    # Numerical safeguard for x(z) calculation
    arg = math.sqrt(max(0.0, 1.0 - 2.0 * rho * z + z**2)) + z - rho
    if arg <= 0.0:
        x_z = z
    else:
        denom_xz = 1.0 - rho
        x_z = math.log(arg / max(1e-12, denom_xz))

    # Denominator expansion
    denom1 = fk_beta * (
        1.0 + (one_minus_beta**2 / 24.0) * (log_fk**2) + (one_minus_beta**4 / 1920.0) * (log_fk**4)
    )

    # Curvature / maturity adjustment
    factor = (
        (one_minus_beta**2 / 24.0) * (alpha**2 / (fk**one_minus_beta))
        + 0.25 * (rho * beta * nu * alpha / fk_beta)
        + ((2.0 - 3.0 * rho**2) / 24.0) * (nu**2)
    )

    if abs(z) < 1e-6:
        ratio = 1.0
    else:
        ratio = z / x_z

    vol = (alpha / denom1) * ratio * (1.0 + factor * t)
    return float(max(1e-6, vol))


class SABRCalibrator(SurfaceCalibrator):
    r"""Calibrates SABR model parameters (alpha, beta, rho, nu) to market implied volatility.

    Parameters
    ----------
    fixed_beta : float, optional
        If set (e.g. 1.0 for equities/FX or 0.5 for rates), beta is held fixed
        and only (alpha, rho, nu) are calibrated. If None, beta is calibrated freely.
    """

    def __init__(
        self,
        fixed_beta: Optional[float] = None,
        loss_fn: Optional[SurfaceLoss] = None,
        optimizer: Optional[SurfaceOptimizer] = None,
    ) -> None:
        super().__init__(loss_fn=loss_fn, optimizer=optimizer)
        self.fixed_beta = float(fixed_beta) if fixed_beta is not None else None

    @property
    def param_names(self) -> List[str]:
        if self.fixed_beta is not None:
            return ["alpha", "rho", "nu"]
        return ["alpha", "beta", "rho", "nu"]

    @property
    def default_bounds(self) -> List[Tuple[float, float]]:
        if self.fixed_beta is not None:
            return [(0.001, 5.0), (-0.999, 0.999), (0.001, 5.0)]
        return [(0.001, 5.0), (0.0, 1.0), (-0.999, 0.999), (0.001, 5.0)]

    @property
    def default_initial_guess(self) -> np.ndarray:
        if self.fixed_beta is not None:
            return np.array([0.20, -0.20, 0.40])
        return np.array([0.20, 0.70, -0.20, 0.40])

    def estimate_initial_alpha(self, atm_vol: float, forward: float, maturity: float) -> float:
        """Estimate initial alpha from ATM market volatility."""
        beta = self.fixed_beta if self.fixed_beta is not None else 0.7
        one_minus_beta = 1.0 - beta
        # alpha0 ~ atm_vol * f^(1 - beta)
        return float(atm_vol * (forward**one_minus_beta))

    def model_quote(
        self,
        strike: float,
        maturity: float,
        params: np.ndarray,
        surface: MarketSurface,
    ) -> float:
        if self.fixed_beta is not None:
            alpha, rho, nu = params[0], params[1], params[2]
            beta = self.fixed_beta
        else:
            alpha, beta, rho, nu = params[0], params[1], params[2], params[3]

        forward = surface.spot * math.exp((surface.rate - surface.dividend_yield) * maturity)
        return sabr_hagan_implied_vol(forward, strike, maturity, alpha, beta, rho, nu)
