"""Likelihood Ratio Method (LRM) / Score Function Greeks for discontinuous payoffs."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from spxa.diff.config import require_jax

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


def lrm_log_gaussian_kernel(x_t: Any, mean: Any, var: Any) -> Any:
    r"""Log-likelihood kernel of normal density :math:`\ln p(x) = -\frac{1}{2}\ln(2\pi \sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}`."""
    return -0.5 * jnp.log(2.0 * jnp.pi * var) - 0.5 * ((x_t - mean) ** 2) / var


def compute_lrm_greeks_gbm(
    payoff_fn: Callable[[Any], Any],
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_paths: int,
    key: Any,
) -> Tuple[float, Dict[str, float]]:
    r"""Compute Greeks via Likelihood Ratio Method for discontinuous payoffs (e.g., Digitals, Barriers).

    .. math::
        \frac{\partial}{\partial \theta} \mathbb{E}[f(S_T)] = \mathbb{E}\left[ f(S_T) \frac{\partial \ln p(S_T; \theta)}{\partial \theta} \right]
    """
    require_jax()
    key_sample, _ = jax.random.split(key)

    # Sample terminal log-prices ln(S_T) ~ N(mu, var)
    mu = jnp.log(s0) + (r - 0.5 * sigma**2) * t
    std = sigma * jnp.sqrt(t)
    var = std**2

    z = jax.random.normal(key_sample, shape=(n_paths,))
    log_st = mu + std * z
    st = jnp.exp(log_st)

    payoffs = payoff_fn(st)
    discount = jnp.exp(-r * t)
    price = float(discount * jnp.mean(payoffs))

    # Score functions:
    # d ln p / d S_0 = (log(S_T) - mu) / (S_0 * sigma^2 * T) = z / (S_0 * sigma * sqrt(T))
    score_delta = z / (s0 * std)
    delta_lrm = float(discount * jnp.mean(payoffs * score_delta))

    # Score function for sigma:
    # ln p = -ln(std) - 0.5 * z^2 -> d ln p / d sigma = (z^2 - 1) / sigma - z * sqrt(T)
    # Exact derivative of log p(log_st; sigma):
    # d(log_st - mu)/d sigma = - (-sigma * T) = sigma * T
    # d ln p / d sigma = (z^2 - 1) / sigma - z * sqrt(T)
    score_vega = (z**2 - 1.0) / sigma - z * jnp.sqrt(t)
    vega_lrm = float(discount * jnp.mean(payoffs * score_vega))

    return price, {"delta": delta_lrm, "vega": vega_lrm}
