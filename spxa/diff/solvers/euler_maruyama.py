"""Differentiable Euler-Maruyama SDE integrator compiled via jax.lax.scan."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from spxa.diff.config import require_jax
from spxa.diff.prng import generate_normal
from spxa.diff.sde_base import DifferentiableSDE

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


def euler_maruyama_step(
    sde: DifferentiableSDE,
    t: Any,
    x: Any,
    dt: float | Any,
    dw: Any,
    params: Dict[str, Any],
) -> Any:
    r"""Single Euler-Maruyama numerical discretization step:

    .. math::
        \boldsymbol{X}_{t+\Delta t} = \Pi\left[ \boldsymbol{X}_t + \boldsymbol{f}(t, \boldsymbol{X}_t, \boldsymbol{\theta}) \Delta t
                                     + \boldsymbol{g}(t, \boldsymbol{X}_t, \boldsymbol{\theta}) \Delta \boldsymbol{W}_t \right]
    """
    drift_val = sde.drift(t, x, params)
    diff_val = sde.diffusion(t, x, params)

    # If diff_val is (batch, state_dim, noise_dim), perform batched matrix multiplication with dw (batch, noise_dim, 1)
    if jnp.ndim(diff_val) == 3:
        stoch_inc = jnp.squeeze(jnp.matmul(diff_val, dw[..., None]), axis=-1)
    else:
        stoch_inc = diff_val * dw

    x_next = x + drift_val * dt + stoch_inc
    return sde.project_state(x_next)


def solve_euler_maruyama(
    sde: DifferentiableSDE,
    params: Dict[str, Any],
    t_span: Tuple[float, float],
    n_steps: int,
    n_paths: int,
    key: Any,
    antithetic: bool = False,
    return_trajectory: bool = True,
) -> Tuple[Any, Any]:
    r"""Solve an SDE over `t_span = (t0, t1)` using vectorized Euler-Maruyama.

    Parameters:
        sde: DifferentiableSDE instance.
        params: parameter dictionary.
        t_span: tuple (t0, t1).
        n_steps: number of discrete time steps.
        n_paths: number of Monte Carlo paths.
        key: JAX PRNGKey.
        antithetic: if True, applies antithetic variance reduction.
        return_trajectory: if True returns full path (n_paths, n_steps + 1, state_dim),
                           else only terminal values (n_paths, state_dim).

    Returns:
        (trajectory_or_terminal, next_key)
    """
    require_jax()
    t0, t1 = float(t_span[0]), float(t_span[1])
    dt = (t1 - t0) / n_steps
    sqrt_dt = jnp.sqrt(dt)

    # Initial state
    x0 = sde.initial_state(batch_size=n_paths, params=params)

    # Pre-generate standard normal shocks: shape (n_steps, n_paths, noise_dim)
    shocks, next_key = generate_normal(
        key=key,
        shape=(n_steps, n_paths, sde.noise_dim),
        antithetic=antithetic,
    )
    dw_tensor = shocks * sqrt_dt

    t_grid = jnp.linspace(t0, t1 - dt, n_steps)

    def scan_fn(carry_x: Any, elem: Tuple[Any, Any]) -> Tuple[Any, Any]:
        t_curr, dw_curr = elem
        x_next = euler_maruyama_step(sde, t_curr, carry_x, dt, dw_curr, params)
        return x_next, x_next

    final_x, trajectory = jax.lax.scan(scan_fn, x0, (t_grid, dw_tensor))

    if return_trajectory:
        # Prepend initial state: shape (n_steps + 1, n_paths, state_dim) -> swap to (n_paths, n_steps + 1, state_dim)
        full_traj = jnp.concatenate([x0[None, ...], trajectory], axis=0)
        return jnp.swapaxes(full_traj, 0, 1), next_key
    return final_x, next_key
