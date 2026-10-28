"""Differentiable higher-order Milstein solver using automated JVP derivatives."""

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


def milstein_step(
    sde: DifferentiableSDE,
    t: Any,
    x: Any,
    dt: float | Any,
    dw: Any,
    params: Dict[str, Any],
) -> Any:
    r"""Single Milstein integration step with automated spatial diffusion derivative:

    .. math::
        \boldsymbol{X}_{t+\Delta t} = \boldsymbol{X}_t + \boldsymbol{f}(t, \boldsymbol{X}_t) \Delta t
                                     + \boldsymbol{g}(t, \boldsymbol{X}_t) \Delta \boldsymbol{W}_t
                                     + \frac{1}{2} \boldsymbol{g}(t, \boldsymbol{X}_t) \nabla_{\boldsymbol{x}} \boldsymbol{g}(t, \boldsymbol{X}_t) [(\Delta \boldsymbol{W}_t)^2 - \Delta t]
    """
    drift_val = sde.drift(t, x, params)

    # Use jax.jvp to compute g(x) * dg/dx in the direction of g(x)
    def diff_fn(state: Any) -> Any:
        return sde.diffusion(t, state, params)

    g_val, g_prime_g = jax.jvp(diff_fn, (x,), (diff_fn(x),))

    milstein_correction = 0.5 * g_prime_g * (dw**2 - dt)
    x_next = x + drift_val * dt + g_val * dw + milstein_correction
    return sde.project_state(x_next)


def solve_milstein(
    sde: DifferentiableSDE,
    params: Dict[str, Any],
    t_span: Tuple[float, float],
    n_steps: int,
    n_paths: int,
    key: Any,
    antithetic: bool = False,
    return_trajectory: bool = True,
) -> Tuple[Any, Any]:
    r"""Solve an SDE using JAX JIT-compiled Milstein scheme."""
    require_jax()
    t0, t1 = float(t_span[0]), float(t_span[1])
    dt = (t1 - t0) / n_steps
    sqrt_dt = jnp.sqrt(dt)

    x0 = sde.initial_state(batch_size=n_paths, params=params)

    shocks, next_key = generate_normal(
        key=key,
        shape=(n_steps, n_paths, sde.noise_dim),
        antithetic=antithetic,
    )
    dw_tensor = shocks * sqrt_dt
    t_grid = jnp.linspace(t0, t1 - dt, n_steps)

    def scan_fn(carry_x: Any, elem: Tuple[Any, Any]) -> Tuple[Any, Any]:
        t_curr, dw_curr = elem
        x_next = milstein_step(sde, t_curr, carry_x, dt, dw_curr, params)
        return x_next, x_next

    final_x, trajectory = jax.lax.scan(scan_fn, x0, (t_grid, dw_tensor))

    if return_trajectory:
        full_traj = jnp.concatenate([x0[None, ...], trajectory], axis=0)
        return jnp.swapaxes(full_traj, 0, 1), next_key
    return final_x, next_key
