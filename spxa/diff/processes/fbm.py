"""Differentiable Fractional Brownian Motion (fBM) via Davies-Harte and Cholesky."""

from __future__ import annotations

from typing import Any, Tuple

from spxa.diff.config import require_jax
from spxa.diff.prng import generate_normal

try:
    import jax.numpy as jnp
except ImportError:
    pass


def fbm_autocovariance(n: int, hurst: float | Any) -> Any:
    r"""Compute autocovariance sequence :math:`\gamma(k) = \frac{1}{2} (|k-1|^{2H} - 2|k|^{2H} + |k+1|^{2H})`."""
    k = jnp.arange(n)
    two_h = 2.0 * hurst
    gamma = 0.5 * (jnp.abs(k - 1) ** two_h - 2.0 * (jnp.abs(k) ** two_h) + jnp.abs(k + 1) ** two_h)
    return gamma


def simulate_diff_fbm(
    hurst: float | Any,
    t_span: Tuple[float, float],
    n_steps: int,
    n_paths: int,
    key: Any,
) -> Tuple[Any, Any]:
    r"""Generate exact Fractional Brownian Motion trajectories differentiable with respect to H.

    Uses Cholesky decomposition of the exact fractional autocovariance matrix.
    """
    require_jax()
    t0, t1 = float(t_span[0]), float(t_span[1])
    total_time = t1 - t0
    dt = total_time / n_steps

    # Autocovariance sequence of increments on [0, 1]
    gamma = fbm_autocovariance(n_steps, hurst)

    # Construct symmetric Toeplitz covariance matrix of increments
    idx = jnp.abs(jnp.arange(n_steps)[:, None] - jnp.arange(n_steps)[None, :])
    cov_matrix = gamma[idx] + 1e-9 * jnp.eye(n_steps)

    # Cholesky factor L (n_steps, n_steps)
    L = jnp.linalg.cholesky(cov_matrix)

    # Standard normal i.i.d. shocks: shape (n_paths, n_steps)
    shocks, next_key = generate_normal(key, shape=(n_paths, n_steps))

    # Increments: (n_paths, n_steps) = shocks @ L.T
    increments = jnp.matmul(shocks, L.T) * (dt**hurst)

    # Cumulative sum to form fBM paths starting at 0
    paths = jnp.cumsum(increments, axis=-1)
    paths_with_zero = jnp.pad(paths, ((0, 0), (1, 0)), mode="constant", constant_values=0.0)

    return paths_with_zero, next_key
