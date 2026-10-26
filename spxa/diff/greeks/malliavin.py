"""Malliavin Calculus sensitivity engine via Bismut-Elworthy-Li formulas."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from spxa.diff.config import require_jax
from spxa.diff.prng import generate_normal

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


def malliavin_delta_weight(
    s_traj: Any,
    dw: Any,
    t_grid: Any,
    sigma: float,
    t_total: float,
) -> Any:
    r"""Compute Bismut-Elworthy-Li Malliavin delta weight :math:`\pi^\Delta`:

    .. math::
        \pi^\Delta = \frac{1}{S_0 \sigma T} W_T
    """
    s0 = s_traj[:, 0, 0]
    w_t = jnp.sum(dw, axis=1)  # sum over time steps -> W_T
    # Malliavin weight
    return w_t / (s0[:, None] * sigma * t_total)


def compute_malliavin_greeks_gbm(
    payoff_fn: Callable[[Any], Any],
    s0: float,
    r: float,
    sigma: float,
    t_total: float,
    n_paths: int,
    key: Any,
) -> Tuple[float, float, float]:
    r"""Compute Malliavin Delta and Gamma for Black-Scholes-Merton Geometric Brownian Motion.

    Under Bismut-Elworthy-Li formula:
    .. math::
        \Delta = e^{-r T} \mathbb{E}\left[ f(S_T) \frac{W_T}{S_0 \sigma T} \right]
        \Gamma = e^{-r T} \mathbb{E}\left[ f(S_T) \frac{W_T^2 - \sigma W_T \sqrt{T} - T}{S_0^2 \sigma^2 T^2} \right]
    """
    require_jax()
    key_sample, _ = jax.random.split(key)

    # Standard normal shock
    z = jax.random.normal(key_sample, shape=(n_paths,))
    w_t = jnp.sqrt(t_total) * z

    st = s0 * jnp.exp((r - 0.5 * sigma**2) * t_total + sigma * w_t)
    payoffs = payoff_fn(st)
    discount = jnp.exp(-r * t_total)

    price = float(discount * jnp.mean(payoffs))

    # BEL weights
    weight_delta = w_t / (s0 * sigma * t_total)
    delta_val = float(discount * jnp.mean(payoffs * weight_delta))

    weight_gamma = (w_t**2 - sigma * t_total * w_t - t_total) / ((s0**2) * (sigma**2) * (t_total**2))
    gamma_val = float(discount * jnp.mean(payoffs * weight_gamma))

    return price, delta_val, gamma_val
