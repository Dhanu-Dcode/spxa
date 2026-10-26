"""End-to-end differentiable SDE calibrator using gradient descent."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from spxa.diff.config import require_jax

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


def calibrate_parameters(
    loss_fn: Callable[[Dict[str, Any]], Any],
    initial_params: Dict[str, Any],
    trainable_keys: List[str],
    learning_rate: float = 0.01,
    n_iterations: int = 50,
) -> Tuple[Dict[str, Any], List[float]]:
    r"""Calibrate differentiable model parameters via gradient descent:

    .. math::
        \boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_k)
    """
    require_jax()
    params = {k: jnp.array(v, dtype=jnp.float64) for k, v in initial_params.items()}
    loss_history: List[float] = []

    grad_fn = jax.value_and_grad(loss_fn)

    for i in range(n_iterations):
        loss_val, grads = grad_fn(params)
        loss_history.append(float(loss_val))

        # Update trainable parameters
        for k in trainable_keys:
            if k in grads:
                params[k] = params[k] - learning_rate * grads[k]

    return {k: float(v) if jnp.ndim(v) == 0 else v for k, v in params.items()}, loss_history
