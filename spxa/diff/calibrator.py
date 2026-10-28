"""End-to-end differentiable SDE calibrator using gradient descent."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

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
    learning_rate: float = 0.001,
    n_iterations: int = 50,
    optimizer: str = "adam",
) -> Tuple[Dict[str, Any], List[float]]:
    r"""Calibrate differentiable model parameters via gradient descent or Adam:

    .. math::
        \boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \eta \boldsymbol{m}_k / (\sqrt{\boldsymbol{v}_k} + \epsilon)
    """
    require_jax()
    params = {k: jnp.array(v) for k, v in initial_params.items()}
    loss_history: List[float] = []

    grad_fn = jax.value_and_grad(loss_fn)

    # Adam states
    m = {k: jnp.zeros_like(params[k]) for k in trainable_keys}
    v = {k: jnp.zeros_like(params[k]) for k in trainable_keys}
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for t_step in range(1, n_iterations + 1):
        loss_val, grads = grad_fn(params)
        loss_history.append(float(loss_val))

        for k in trainable_keys:
            if k in grads:
                g = grads[k]
                if optimizer == "adam":
                    m[k] = beta1 * m[k] + (1.0 - beta1) * g
                    v[k] = beta2 * v[k] + (1.0 - beta2) * (g**2)
                    m_hat = m[k] / (1.0 - (beta1**t_step))
                    v_hat = v[k] / (1.0 - (beta2**t_step))
                    params[k] = params[k] - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
                else:
                    params[k] = params[k] - learning_rate * g

    return {k: float(val) if jnp.ndim(val) == 0 else val for k, val in params.items()}, loss_history
