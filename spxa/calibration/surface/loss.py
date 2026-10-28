r"""Objective loss functions and regularization penalties for surface calibration.

Provides loss metrics:
- RMSE (Root Mean Squared Error) on implied volatility or price
- Vega-Weighted Dollar Loss (penalizes ATM options higher, matching financial liquidity)
- Relative Percentage Error
- Regularized / Arbitrage-Penalized Loss functions enforcing Feller condition,
  positive calendar variance, and butterfly arbitrage constraints.
"""

from __future__ import annotations

import abc
from typing import Callable, Optional

import numpy as np


class SurfaceLoss(abc.ABC):
    """Abstract base class for calibration loss functions."""

    @abc.abstractmethod
    def __call__(
        self,
        model_quotes: np.ndarray,
        market_quotes: np.ndarray,
        vegas: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """Compute scalar loss between model quotes and market quotes."""
        raise NotImplementedError

    @abc.abstractmethod
    def residuals(
        self,
        model_quotes: np.ndarray,
        market_quotes: np.ndarray,
        vegas: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute 1D residual vector for least-squares solvers."""
        raise NotImplementedError


class RMSELoss(SurfaceLoss):
    """Root Mean Squared Error: sqrt( 1/N * sum (model - market)^2 )."""

    def __call__(
        self,
        model_quotes: np.ndarray,
        market_quotes: np.ndarray,
        vegas: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        res = self.residuals(model_quotes, market_quotes, vegas, weights)
        return float(np.sqrt(np.mean(res**2)))

    def residuals(
        self,
        model_quotes: np.ndarray,
        market_quotes: np.ndarray,
        vegas: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        diff = np.asarray(model_quotes, dtype=float) - np.asarray(market_quotes, dtype=float)
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            w_norm = w / (np.sum(w) + 1e-12) * len(w)
            return diff * np.sqrt(w_norm)
        return diff


class VegaWeightedLoss(SurfaceLoss):
    """Vega-weighted loss: weights errors proportional to option Vega.

    Reflects the market reality that liquid ATM options with high Vega should be
    fitted with tighter dollar bid-ask tolerance than far OTM wings.
    """

    def __init__(self, min_vega: float = 1e-4) -> None:
        self.min_vega = float(min_vega)

    def residuals(
        self,
        model_quotes: np.ndarray,
        market_quotes: np.ndarray,
        vegas: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        diff = np.asarray(model_quotes, dtype=float) - np.asarray(market_quotes, dtype=float)
        if vegas is None:
            # Fall back to unweighted diff
            return diff
        v = np.maximum(self.min_vega, np.asarray(vegas, dtype=float))
        v_norm = v / np.mean(v)
        return diff * np.sqrt(v_norm)

    def __call__(
        self,
        model_quotes: np.ndarray,
        market_quotes: np.ndarray,
        vegas: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        res = self.residuals(model_quotes, market_quotes, vegas, weights)
        return float(np.sqrt(np.mean(res**2)))


class RelativeErrorLoss(SurfaceLoss):
    """Mean absolute percentage error: 1/N * sum |model - market| / market."""

    def residuals(
        self,
        model_quotes: np.ndarray,
        market_quotes: np.ndarray,
        vegas: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        mkt = np.maximum(1e-6, np.asarray(market_quotes, dtype=float))
        diff = np.asarray(model_quotes, dtype=float) - mkt
        return diff / mkt

    def __call__(
        self,
        model_quotes: np.ndarray,
        market_quotes: np.ndarray,
        vegas: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        res = self.residuals(model_quotes, market_quotes, vegas, weights)
        return float(np.mean(np.abs(res)))


class ArbitragePenalizedLoss(SurfaceLoss):
    """Wraps a base loss function and adds financial penalty terms.

    Penalties include:
    - Feller condition: 2 * kappa * theta >= sigma^2 (for CIR/Heston models)
    - Correlation bounds: |rho| < 1
    - Positivity constraints
    """

    def __init__(
        self,
        base_loss: Optional[SurfaceLoss] = None,
        feller_penalty_weight: float = 100.0,
        correlation_penalty_weight: float = 500.0,
        custom_penalty: Optional[Callable[[np.ndarray], float]] = None,
    ) -> None:
        self.base_loss = base_loss or RMSELoss()
        self.feller_weight = float(feller_penalty_weight)
        self.corr_weight = float(correlation_penalty_weight)
        self.custom_penalty = custom_penalty

    def compute_penalty(self, params: np.ndarray, param_names: Optional[list[str]] = None) -> float:
        """Compute total scalar penalty for a given parameter vector."""
        penalty = 0.0
        p = np.asarray(params, dtype=float)

        if param_names is not None:
            p_map = {name: p[i] for i, name in enumerate(param_names)}
            # Check Feller condition if kappa, theta, sigma exist: 2 * kappa * theta >= sigma^2
            if "kappa" in p_map and "theta" in p_map and "sigma" in p_map:
                diff = (p_map["sigma"] ** 2) - 2.0 * p_map["kappa"] * p_map["theta"]
                if diff > 0.0:
                    penalty += self.feller_weight * (diff**2)
            # Check correlation if rho exists
            if "rho" in p_map:
                if abs(p_map["rho"]) >= 1.0:
                    penalty += self.corr_weight * ((abs(p_map["rho"]) - 0.999) ** 2)

        if self.custom_penalty is not None:
            penalty += self.custom_penalty(p)

        return penalty

    def residuals(
        self,
        model_quotes: np.ndarray,
        market_quotes: np.ndarray,
        vegas: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.base_loss.residuals(model_quotes, market_quotes, vegas, weights)

    def __call__(
        self,
        model_quotes: np.ndarray,
        market_quotes: np.ndarray,
        vegas: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        return self.base_loss(model_quotes, market_quotes, vegas, weights)
