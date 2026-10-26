"""Reversible Heun and Midpoint SDE integrators for memory-efficient backpropagation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from spxa.diff.config import require_jax
from spxa.diff.prng import generate_normal
from spxa.diff.sde_base import DifferentiableSDE

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


def reversible_heun_step(
    sde: DifferentiableSDE,
    t: Any,
    x: Any,
    dt: float | Any,
    dw: Any,
    params: Dict[str, Any],
) -> Any:
    r"""Second-order trapezoidal predictor-corrector step for Stratonovich/Itô systems:

    .. math::
        \boldsymbol{X}^{(1)} = \boldsymbol{X}_t + \boldsymbol{f}(t, \boldsymbol{X}_t) \Delta t + \boldsymbol{g}(t, \boldsymbol{X}_t) \Delta \boldsymbol{W}_t
        \boldsymbol{X}_{t+\Delta t} = \boldsymbol{X}_t + \frac{1}{2} \left[ \boldsymbol{f}(t, \boldsymbol{X}_t) + \boldsymbol{f}(t+\Delta t, \boldsymbol{X}^{(1)}) \right] \Delta t
                                     + \frac{1}{2} \left[ \boldsymbol{g}(t, \boldsymbol{X}_t) + \boldsymbol{g}(t+\Delta t, \boldsymbol{X}^{(1)}) \right] \Delta \boldsymbol{W}_t
    """
    f0 = sde.drift(t, x, params)
    g0 = sde.diffusion(t, x, params)

    x1 = sde.project_state(x + f0 * dt + g0 * dw)

    f1 = sde.drift(t + dt, x1, params)
    g1 = sde.diffusion(t + dt, x1, params)

    x_next = x + 0.5 * (f0 + f1) * dt + 0.5 * (g0 + g1) * dw
    return sde.project_state(x_next)


def solve_reversible_heun(
    sde: DifferentiableSDE,
    params: Dict[str, Any],
    t_span: Tuple[float, float],
    n_steps: int,
    n_paths: int,
    key: Any,
    antithetic: bool = False,
    return_trajectory: bool = True,
) -> Tuple[Any, Any]:
    r"""Solve an SDE using Reversible Heun scheme."""
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
        x_next = reversible_heun_step(sde, t_curr, carry_x, dt, dw_curr, params)
        return x_next, x_next

    final_x, trajectory = jax.lax.scan(scan_fn, x0, (t_grid, dw_tensor))

    if return_trajectory:
        full_traj = jnp.concatenate([x0[None, ...], trajectory], axis=0)
        return jnp.swapaxes(full_traj, 0, 1), next_key
    return final_x, next_key
