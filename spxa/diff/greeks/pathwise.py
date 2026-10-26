"""Pathwise automatic differentiation sensitivity engine."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from spxa.diff.config import require_jax
from spxa.diff.sde_base import DifferentiableSDE
from spxa.diff.solvers.euler_maruyama import solve_euler_maruyama

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


def pathwise_expected_payoff(
    payoff_fn: Callable[[Any], Any],
    sde: DifferentiableSDE,
    params: Dict[str, Any],
    t_span: Tuple[float, float],
    n_steps: int,
    n_paths: int,
    key: Any,
    antithetic: bool = False,
    discount_rate: float = 0.0,
) -> Any:
    r"""Compute discounted expected payoff :math:`e^{-r T} \mathbb{E}[f(X_T)]` across Monte Carlo paths."""
    terminal_x, _ = solve_euler_maruyama(
        sde=sde,
        params=params,
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


def compute_pathwise_greeks(
    payoff_fn: Callable[[Any], Any],
    sde: DifferentiableSDE,
    params: Dict[str, Any],
    param_keys: list[str],
    t_span: Tuple[float, float] = (0.0, 1.0),
    n_steps: int = 100,
    n_paths: int = 2000,
    key: Any = None,
    antithetic: bool = True,
    discount_rate: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    r"""Compute price and pathwise sensitivities (Greeks) with respect to specified parameter names.

    Returns:
        (price, {param_name: gradient_value})
    """
    require_jax()

    def loss_fn(p: Dict[str, Any]) -> Any:
        return pathwise_expected_payoff(
            payoff_fn=payoff_fn,
            sde=sde,
            params=p,
            t_span=t_span,
            n_steps=n_steps,
            n_paths=n_paths,
            key=key,
            antithetic=antithetic,
            discount_rate=discount_rate,
        )

    val, grad_dict = jax.value_and_grad(loss_fn)(params)
    filtered_grads = {k: float(grad_dict[k]) for k in param_keys if k in grad_dict}
    return float(val), filtered_grads
