r"""CEV-Heston Local-Stochastic Volatility Surface Calibrator.

Calibrates the 6 parameters of the combined CEV local volatility and Heston stochastic volatility model:
    dS_t = r * S_t * dt + S_t^gamma * \sqrt{v_t} * dW_t^S
    dv_t = \kappa * (\theta - v_t) * dt + \sigma * \sqrt{v_t} * dW_t^v

Parameters:
- v0    : initial variance level (> 0)
- kappa : mean reversion speed of variance (> 0)
- theta : long-term variance mean (> 0)
- sigma : volatility of variance (> 0)
- rho   : correlation between spot and variance in (-1, 1)
- gamma : CEV elasticity parameter in (0, 1.5] (gamma = 1 recovers pure Heston)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from spxa.calibration.surface.base import SurfaceCalibrator
from spxa.calibration.surface.data import MarketSurface
from spxa.calibration.surface.heston import bs_implied_vol, heston_call_price
from spxa.calibration.surface.loss import ArbitragePenalizedLoss, SurfaceLoss
from spxa.calibration.surface.optimizer import SurfaceOptimizer


def cev_heston_approx_implied_vol(
    s0: float,
    k: float,
    t: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    gamma: float,
) -> float:
    r"""Compute implied volatility for CEV-Heston using local-volatility perturbation expansion."""
    # Effective reference strike scaling: (S_0 * K)^{(gamma - 1) / 2}
    # When gamma == 1.0, scale == 1.0 (pure Heston)
    mid_scale = (s0 * k) ** (0.5 * (gamma - 1.0))
    effective_v0 = v0 * (mid_scale**2)
    effective_theta = theta * (mid_scale**2)

    call = heston_call_price(
        s0=s0,
        k=k,
        t=t,
        r=r,
        v0=effective_v0,
        kappa=kappa,
        theta=effective_theta,
        sigma=sigma,
        rho=rho,
    )
    return bs_implied_vol(call, s0, k, t, r)


class CEVHestonCalibrator(SurfaceCalibrator):
    r"""Calibrates CEV-Heston local-stochastic volatility model to market surfaces."""

    def __init__(
        self,
        enforce_feller: bool = True,
        loss_fn: Optional[SurfaceLoss] = None,
        optimizer: Optional[SurfaceOptimizer] = None,
    ) -> None:
        if enforce_feller and loss_fn is None:
            base_loss = ArbitragePenalizedLoss(feller_penalty_weight=50.0)
        else:
            base_loss = loss_fn
        super().__init__(loss_fn=base_loss, optimizer=optimizer)

    @property
    def param_names(self) -> List[str]:
        return ["v0", "kappa", "theta", "sigma", "rho", "gamma"]

    @property
    def default_bounds(self) -> List[Tuple[float, float]]:
        return [
            (0.001, 1.5),  # v0
            (0.05, 10.0),  # kappa
            (0.001, 1.5),  # theta
            (0.01, 2.5),  # sigma
            (-0.99, 0.99),  # rho
            (0.2, 1.5),  # gamma (CEV elasticity)
        ]

    @property
    def default_initial_guess(self) -> np.ndarray:
        return np.array([0.04, 1.5, 0.04, 0.3, -0.6, 1.0])

    def model_quote(
        self,
        strike: float,
        maturity: float,
        params: np.ndarray,
        surface: MarketSurface,
    ) -> float:
        v0, kappa, theta, sigma, rho, gamma = (
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5],
        )
        return cev_heston_approx_implied_vol(
            s0=surface.spot,
            k=strike,
            t=maturity,
            r=surface.rate,
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
            gamma=gamma,
        )
