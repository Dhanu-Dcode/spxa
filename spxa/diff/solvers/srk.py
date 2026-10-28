"""Differentiable Stochastic Runge-Kutta (SRK) higher-order solvers."""

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


def srk_step(
    sde: DifferentiableSDE,
    t: Any,
    x: Any,
    dt: float | Any,
    dw: Any,
    params: Dict[str, Any],
) -> Any:
    r"""Two-stage explicit Stochastic Runge-Kutta predictor-corrector step.

    Strong order 1.0 derivative-free scheme (Kloeden & Platen, 1992):
    .. math::
        \boldsymbol{K}_1 = \boldsymbol{f}(t, \boldsymbol{X}_t) \Delta t
        \boldsymbol{G}_1 = \boldsymbol{g}(t, \boldsymbol{X}_t)
        \hat{\boldsymbol{X}} = \boldsymbol{X}_t + \boldsymbol{K}_1 + \boldsymbol{G}_1 \sqrt{\Delta t}
        \boldsymbol{G}_2 = \boldsymbol{g}(t + \Delta t, \hat{\boldsymbol{X}})
        \boldsymbol{X}_{t+\Delta t} = \boldsymbol{X}_t + \boldsymbol{K}_1 + \boldsymbol{G}_1 \Delta \boldsymbol{W}_t
                                     + \frac{1}{2\sqrt{\Delta t}} (\boldsymbol{G}_2 - \boldsymbol{G}_1) [(\Delta \boldsymbol{W}_t)^2 - \Delta t]
    """
    sqrt_dt = jnp.sqrt(dt)

    k1 = sde.drift(t, x, params) * dt
    g1 = sde.diffusion(t, x, params)

    # Support state
    x_hat = x + k1 + g1 * sqrt_dt
    g2 = sde.diffusion(t + dt, x_hat, params)

    srk_corr = 0.5 / sqrt_dt * (g2 - g1) * (dw**2 - dt)
    x_next = x + k1 + g1 * dw + srk_corr
    return sde.project_state(x_next)


def solve_srk(
    sde: DifferentiableSDE,
    params: Dict[str, Any],
    t_span: Tuple[float, float],
    n_steps: int,
    n_paths: int,
    key: Any,
    antithetic: bool = False,
    return_trajectory: bool = True,
) -> Tuple[Any, Any]:
    r"""Solve an SDE using two-stage explicit Stochastic Runge-Kutta."""
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
        x_next = srk_step(sde, t_curr, carry_x, dt, dw_curr, params)
        return x_next, x_next

    final_x, trajectory = jax.lax.scan(scan_fn, x0, (t_grid, dw_tensor))

    if return_trajectory:
        full_traj = jnp.concatenate([x0[None, ...], trajectory], axis=0)
        return jnp.swapaxes(full_traj, 0, 1), next_key
    return final_x, next_key
