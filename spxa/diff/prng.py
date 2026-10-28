"""Differentiable pseudo-random number generator routines and variance reduction."""

from __future__ import annotations

import time
from typing import Any, Tuple

from spxa.diff.config import require_jax

try:
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
except ImportError:
    pass


def make_key(seed: int | None = None) -> Any:
    """Create a PRNGKey from a seed, or current nanosecond timestamp if None."""
    require_jax()
    if seed is None:
        seed = int(time.time_ns()) & 0x7FFFFFFF
    return jrandom.PRNGKey(seed)


def split_key(key: Any, num: int = 2) -> Tuple[Any, ...]:
    """Split a PRNG key into `num` subkeys."""
    require_jax()
    keys = jrandom.split(key, num)
    return tuple(keys[i] for i in range(num))


def generate_normal(
    key: Any,
    shape: Tuple[int, ...],
    dtype: Any = None,
    antithetic: bool = False,
) -> Tuple[Any, Any]:
    """Generate standard normal shocks with optional antithetic variance reduction.

    Returns:
        (samples, next_key) where samples has shape `shape`.
        If antithetic is True and the first dimension is even, pairs are exact negations.
    """
    require_jax()
    if dtype is None:
        dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    key_sample, next_key = jrandom.split(key)
    if antithetic:
        half_shape = (shape[0] // 2,) + shape[1:]
        z_half = jrandom.normal(key_sample, shape=half_shape, dtype=dtype)
        z = jnp.concatenate([z_half, -z_half], axis=0)
        # In case shape[0] was odd, generate remainder
        if shape[0] % 2 != 0:
            extra_key, next_key = jrandom.split(next_key)
            z_extra = jrandom.normal(extra_key, shape=(1,) + shape[1:], dtype=dtype)
            z = jnp.concatenate([z, z_extra], axis=0)
    else:
        z = jrandom.normal(key_sample, shape=shape, dtype=dtype)
    return z, next_key


def generate_uniform(
    key: Any,
    shape: Tuple[int, ...],
    dtype: Any = None,
    minval: float = 0.0,
    maxval: float = 1.0,
) -> Tuple[Any, Any]:
    """Generate uniform random variables on [minval, maxval]."""
    require_jax()
    if dtype is None:
        dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    key_sample, next_key = jrandom.split(key)
    u = jrandom.uniform(key_sample, shape=shape, dtype=dtype, minval=minval, maxval=maxval)
    return u, next_key


def generate_poisson(
    key: Any,
    lam: float | Any,
    shape: Tuple[int, ...],
    dtype: Any = None,
) -> Tuple[Any, Any]:
    """Generate Poisson distributed random counts with arrival rate `lam`."""
    require_jax()
    if dtype is None:
        dtype = jnp.int32

    key_sample, next_key = jrandom.split(key)
    p = jrandom.poisson(key_sample, lam=lam, shape=shape, dtype=dtype)
    return p, next_key


def generate_gamma(
    key: Any,
    shape_param: float | Any,
    shape: Tuple[int, ...],
    dtype: Any = None,
) -> Tuple[Any, Any]:
    """Generate standard Gamma distributed variables Gamma(a, 1)."""
    require_jax()
    if dtype is None:
        dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    key_sample, next_key = jrandom.split(key)
    g = jrandom.gamma(key_sample, a=shape_param, shape=shape, dtype=dtype)
    return g, next_key
