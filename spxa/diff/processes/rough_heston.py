"""Differentiable Rough Heston Volterra Stochastic Volatility Model."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from spxa.diff.config import require_jax
from spxa.diff.prng import generate_normal

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


def simulate_diff_rough_heston(
    params: Dict[str, Any],
    t_span: Tuple[float, float],
    n_steps: int,
    n_paths: int,
    key: Any,
) -> Tuple[Any, Any]:
    r"""Simulate rough Heston model using fractional kernel convolution:

    .. math::
        v_t = v_0 + \frac{1}{\Gamma(\alpha)} \int_0^t (t-s)^{\alpha-1} \lambda (\theta - v_s) ds
            + \frac{1}{\Gamma(\alpha)} \int_0^t (t-s)^{\alpha-1} \nu \sqrt{v_s} dW_s^v
    where :math:`\alpha = H + 1/2 \in (0.5, 1.0)`.
    """
    require_jax()
    t0, t1 = float(t_span[0]), float(t_span[1])
    dt = (t1 - t0) / n_steps

    s0 = params.get("s0", 100.0)
    v0 = params.get("v0", 0.04)
    lam = params.get("lambda", 1.5)
    theta = params.get("theta", 0.04)
    nu = params.get("nu", 0.3)
    rho = params.get("rho", -0.7)
    h = params.get("hurst", 0.1)

    alpha = h + 0.5
    # Fractional weights b_k = ((k+1)^alpha - k^alpha) * dt^alpha / (alpha * Gamma(alpha))
    k_vec = jnp.arange(n_steps)
    gamma_val = float(math.gamma(alpha))
    weights = ((k_vec + 1.0) ** alpha - (k_vec**alpha)) * (dt**alpha) / (alpha * gamma_val)

    # Shocks
    key_shocks, next_key = generate_normal(key, shape=(n_steps, n_paths, 2))
    z1 = key_shocks[..., 0]
    z2 = key_shocks[..., 1]
    # Correlated driving Brownian increments
    dw_s = jnp.sqrt(dt) * z1
    dw_v = jnp.sqrt(dt) * (rho * z1 + jnp.sqrt(jnp.maximum(1.0 - rho**2, 1e-12)) * z2)

    # Sequential Volterra convolution via lax.scan
    def step_fn(carry: Tuple[Any, Any, Any], elem: Tuple[Any, Any, Any]) -> Tuple[Tuple[Any, Any, Any], Tuple[Any, Any]]:
        step_idx, v_hist, s_curr = carry
        w_curr, dw_s_curr, dw_v_curr = elem

        # Convolution sum over past
        # Current variance
        v_curr = v_hist[step_idx]
        v_pos = jnp.maximum(v_curr, 0.0)
        sqrt_v = jnp.sqrt(v_pos)

        # Update log-price
        r = params.get("r", 0.03)
        s_next = s_curr * jnp.exp((r - 0.5 * v_pos) * dt + sqrt_v * dw_s_curr)

        # Contribution of step_idx to all future times j > step_idx
        drift_term = lam * (theta - v_pos) * dt
        diff_term = nu * sqrt_v * dw_v_curr
        increment = drift_term + diff_term

        # Add contribution: for future step j, weight is weights[j - step_idx - 1]
        shift = jnp.arange(n_steps + 1) - (step_idx + 1)
        valid_mask = shift >= 0
        w_slice = jnp.where(valid_mask, weights[jnp.clip(shift, 0, n_steps - 1)], 0.0)

        # Outer product: w_slice[:, None] * increment[None, :]
        v_hist_next = v_hist + w_slice[:, None] * increment[None, :]

        return (step_idx + 1, v_hist_next, s_next), (s_next, v_pos)

    # Initial variance history initialized to v0
    v_hist_0 = jnp.full((n_steps + 1, n_paths), v0)
    s_0 = jnp.full((n_paths,), s0)

    _, (s_traj, v_traj) = jax.lax.scan(
        step_fn,
        (0, v_hist_0, s_0),
        (weights, dw_s, dw_v),
    )

    # Trajectory shape: (n_paths, n_steps + 1, 2)
    s_all = jnp.concatenate([s_0[None, :], s_traj], axis=0)
    v_all = jnp.concatenate([jnp.full((1, n_paths), v0), v_traj], axis=0)
    combined = jnp.stack([s_all, v_all], axis=-1)  # (n_steps + 1, n_paths, 2)
    return jnp.swapaxes(combined, 0, 1), next_key
