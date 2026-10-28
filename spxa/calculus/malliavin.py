"""
Malliavin Calculus, Clark-Ocone Hedging & Bismut-Elworthy-Li Engine.

Implements variational stochastic calculus on Wiener space: the first variation
tangent process, Malliavin derivatives, the Bismut-Elworthy-Li formula for
singular/digital Greeks, the Clark-Ocone martingale representation, and the
Skorokhod divergence operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class BismutGreekResult:
    r"""Results of Bismut-Elworthy-Li integration by parts on Wiener space.

    Attributes
    ----------
    greek_value : float
        Estimated price sensitivity (e.g. Delta or Vega).
    standard_error : float
        Monte Carlo standard error of the Malliavin estimator.
    weights : np.ndarray
        Stochastic Malliavin weights :math:`\int_0^T \sigma^{-1}(X_t) Y_t dW_t`.
    """

    greek_value: float
    standard_error: float
    weights: np.ndarray


def simulate_first_variation(
    drift_derivative: Callable[[float, Any], Any],
    diffusion_derivative: Callable[[float, Any], Any],
    paths: np.ndarray,
    times: np.ndarray,
    bm_increments: np.ndarray,
) -> np.ndarray:
    r"""Simulate the tangent first variation process :math:`Y_t = \frac{\partial X_t}{\partial x_0}`.

    Satisfies the linearized variational SDE:

    .. math::
        dY_t = b'(t, X_t) Y_t dt + \sigma'(t, X_t) Y_t dW_t, \quad Y_0 = 1
    """
    x = np.atleast_2d(np.asarray(paths, dtype=float))
    t = np.asarray(times, dtype=float)
    dw = np.atleast_2d(np.asarray(bm_increments, dtype=float))
    dt = np.diff(t)

    n_paths, n_steps = dw.shape
    y = np.ones((n_paths, n_steps + 1))

    for k in range(n_steps):
        t_k = t[k]
        x_k = x[:, k]
        b_prime = np.asarray([drift_derivative(t_k, s) for s in x_k], dtype=float)
        sig_prime = np.asarray([diffusion_derivative(t_k, s) for s in x_k], dtype=float)

        d_log_y = (b_prime - 0.5 * sig_prime**2) * dt[k] + sig_prime * dw[:, k]
        y[:, k + 1] = y[:, k] * np.exp(d_log_y)

    return y if paths.ndim > 1 else y[0]


def malliavin_derivative_diffusion(
    s_idx: int,
    t_idx: int,
    first_variation_paths: np.ndarray,
    diffusion_at_s: np.ndarray,
) -> np.ndarray:
    r"""Evaluate the Malliavin derivative :math:`D_s X_t` for :math:`s \le t`:

    .. math::
        D_s X_t = Y_t Y_s^{-1} \sigma(s, X_s) \mathbf{1}_{\{s \le t\}}
    """
    if s_idx > t_idx:
        return np.zeros_like(diffusion_at_s)

    y = np.atleast_2d(first_variation_paths)
    y_t = y[:, t_idx]
    y_s = y[:, s_idx]
    sig_s = np.asarray(diffusion_at_s, dtype=float)
    return (y_t / (y_s + 1e-12)) * sig_s


def bismut_elworthy_li_delta(
    paths: np.ndarray,
    terminal_payoff: np.ndarray,
    times: np.ndarray,
    bm_increments: np.ndarray,
    diffusion_func: Callable[[float, Any], Any],
    first_variation_paths: Optional[np.ndarray] = None,
    discount_factor: float = 1.0,
) -> BismutGreekResult:
    r"""Compute the exact Delta :math:`\frac{\partial}{\partial x_0} \mathbb{E}[\Phi(X_T)]` via Bismut-Elworthy-Li.

    .. math::
        \nabla_x \mathbb{E}[\Phi(X_T)] = \frac{e^{-rT}}{T} \mathbb{E}\left[ \Phi(X_T) \int_0^T \sigma^{-1}(X_t) Y_t dW_t \right]

    Does NOT differentiate the payoff :math:`\Phi`, allowing exact Greeks for
    discontinuous payoffs (digital options, barrier crossings, binary bets).
    """
    x = np.atleast_2d(np.asarray(paths, dtype=float))
    t = np.asarray(times, dtype=float)
    dw = np.atleast_2d(np.asarray(bm_increments, dtype=float))
    payoff = np.asarray(terminal_payoff, dtype=float)

    n_paths, n_steps = dw.shape
    total_t = t[-1] - t[0]

    if first_variation_paths is None:
        y = np.ones((n_paths, n_steps + 1))
    else:
        y = np.atleast_2d(first_variation_paths)

    # Accumulate stochastic integral: int_0^T sigma^{-1}(X_t) Y_t dW_t
    int_weight = np.zeros(n_paths)
    for k in range(n_steps):
        t_k = t[k]
        x_k = x[:, k]
        sig_k = np.asarray([diffusion_func(t_k, s) for s in x_k], dtype=float)
        # Avoid division by zero
        sig_inv = 1.0 / np.where(np.abs(sig_k) < 1e-8, 1e-8, sig_k)
        int_weight += sig_inv * y[:, k] * dw[:, k]

    weights = int_weight / total_t
    estimator_samples = discount_factor * payoff * weights

    mean_greek = float(np.mean(estimator_samples))
    stderr = float(np.std(estimator_samples, ddof=1) / np.sqrt(n_paths))

    return BismutGreekResult(
        greek_value=mean_greek,
        standard_error=stderr,
        weights=weights,
    )


def clark_ocone_hedging_delta(
    terminal_payoff: np.ndarray,
    paths: np.ndarray,
    times: np.ndarray,
    volatility: float,
    discount_rate: float = 0.0,
) -> float:
    r"""Compute the replicating delta at :math:`t=0` via the Clark-Ocone formula:

    .. math::
        \Delta_0 = \frac{e^{-rT}}{\sigma S_0 T} \mathbb{E}\left[ \Phi(S_T) W_T \right]
    """
    payoff = np.asarray(terminal_payoff, dtype=float)
    x = np.asarray(paths, dtype=float)
    s0 = x[:, 0] if x.ndim > 1 else x[0]
    s_mean_0 = float(np.mean(s0))
    t_terminal = float(times[-1])

    # For standard lognormal stock price, Clark-Ocone weight at t=0:
    # W_T / (sigma * S_0 * T)
    # Estimate W_T from log returns
    log_st = np.log(x[:, -1] / s_mean_0)
    # Drift of log S is (r - 0.5*sigma^2)*T
    w_t = (log_st - (discount_rate - 0.5 * volatility**2) * t_terminal) / volatility

    weight = w_t / (volatility * s_mean_0 * t_terminal)
    delta_samples = np.exp(-discount_rate * t_terminal) * payoff * weight
    return float(np.mean(delta_samples))
