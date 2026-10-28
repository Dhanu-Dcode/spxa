"""Higher-order Greeks and cross-sensitivities using JAX Hessian computation."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from spxa.diff.config import require_jax
from spxa.diff.sde_base import DifferentiableSDE
from spxa.diff.solvers.euler_maruyama import solve_euler_maruyama

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


def compute_higher_order_greeks(
    payoff_fn: Callable[[Any], Any],
    sde: DifferentiableSDE,
    params: Dict[str, Any],
    param_keys: List[str],
    t_span: Tuple[float, float] = (0.0, 1.0),
    n_steps: int = 100,
    n_paths: int = 2000,
    key: Any = None,
    antithetic: bool = True,
    discount_rate: float = 0.0,
) -> Tuple[float, Dict[str, float], Dict[Tuple[str, str], float]]:
    r"""Compute price, first-order gradient vector, and full second-order Hessian matrix:

    .. math::
        H_{ij} = \frac{\partial^2 V}{\partial \theta_i \partial \theta_j}
    """
    require_jax()

    def loss_fn(p: Dict[str, Any]) -> Any:
        terminal_x, _ = solve_euler_maruyama(
            sde=sde,
            params=p,
            t_span=t_span,
            n_steps=n_steps,
            n_paths=n_paths,
            key=key,
            antithetic=antithetic,
            return_trajectory=False,
        )
        payoffs = payoff_fn(terminal_x)
        total_time = t_span[1] - t_span[0]
        discount = jnp.exp(-discount_rate * total_time)
        return discount * jnp.mean(payoffs)

    # Compute value, grad, and full Hessian
    val = float(loss_fn(params))
    grad_dict = jax.grad(loss_fn)(params)
    hessian_dict = jax.hessian(loss_fn)(params)

    first_order = {k: float(grad_dict[k]) for k in param_keys if k in grad_dict}
    second_order = {}
    for k1 in param_keys:
        for k2 in param_keys:
            if k1 in hessian_dict and k2 in hessian_dict[k1]:
                second_order[(k1, k2)] = float(hessian_dict[k1][k2])

    return val, first_order, second_order
