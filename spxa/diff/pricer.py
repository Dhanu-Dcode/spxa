"""Vectorized differentiable option pricing engine."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from spxa.diff.config import require_jax
from spxa.diff.sde_base import DifferentiableSDE
from spxa.diff.solvers.euler_maruyama import solve_euler_maruyama

try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import erf
except ImportError:
    pass


def _standard_normal_cdf(x: Any) -> Any:
    """Vectorized standard Gaussian cumulative distribution function."""
    return 0.5 * (1.0 + erf(x / jnp.sqrt(2.0)))


def bs_call_price(s0: Any, strike: Any, maturity: Any, rate: Any, sigma: Any) -> Any:
    r"""Vectorized Black-Scholes European call pricing formula."""
    sqrt_t = jnp.sqrt(maturity)
    d1 = (jnp.log(s0 / strike) + (rate + 0.5 * sigma**2) * maturity) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return s0 * _standard_normal_cdf(d1) - strike * jnp.exp(-rate * maturity) * _standard_normal_cdf(d2)


def bs_put_price(s0: Any, strike: Any, maturity: Any, rate: Any, sigma: Any) -> Any:
    r"""Vectorized Black-Scholes European put pricing formula."""
    sqrt_t = jnp.sqrt(maturity)
    d1 = (jnp.log(s0 / strike) + (rate + 0.5 * sigma**2) * maturity) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return strike * jnp.exp(-rate * maturity) * _standard_normal_cdf(-d2) - s0 * _standard_normal_cdf(-d1)


def mc_european_call_price(
    sde: DifferentiableSDE,
    params: Dict[str, Any],
    strike: float | Any,
    maturity: float,
    rate: float,
    n_steps: int = 100,
    n_paths: int = 2000,
    key: Any = None,
    antithetic: bool = True,
) -> Any:
    r"""Differentiable Monte Carlo price for European call option."""
    terminal_x, _ = solve_euler_maruyama(
        sde=sde,
        params=params,
        t_span=(0.0, maturity),
        n_steps=n_steps,
        n_paths=n_paths,
        key=key,
        antithetic=antithetic,
        return_trajectory=False,
    )
    s_t = terminal_x[..., 0]
    payoffs = jnp.maximum(s_t - strike, 0.0)
    discount = jnp.exp(-rate * maturity)
    return discount * jnp.mean(payoffs)


def mc_asian_call_price(
    sde: DifferentiableSDE,
    params: Dict[str, Any],
    strike: float | Any,
    maturity: float,
    rate: float,
    n_steps: int = 100,
    n_paths: int = 2000,
    key: Any = None,
    antithetic: bool = True,
) -> Any:
    r"""Differentiable Monte Carlo price for arithmetic average Asian call option."""
    trajectory, _ = solve_euler_maruyama(
        sde=sde,
        params=params,
        t_span=(0.0, maturity),
        n_steps=n_steps,
        n_paths=n_paths,
        key=key,
        antithetic=antithetic,
        return_trajectory=True,
    )
    # trajectory is (n_paths, n_steps + 1, state_dim)
    s_paths = trajectory[..., 0]
    s_avg = jnp.mean(s_paths, axis=-1)
    payoffs = jnp.maximum(s_avg - strike, 0.0)
    discount = jnp.exp(-rate * maturity)
    return discount * jnp.mean(payoffs)
