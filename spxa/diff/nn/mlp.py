"""Pure functional Multi-Layer Perceptron (MLP) for JAX."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from spxa.diff.config import require_jax

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


def init_mlp_params(
    layer_sizes: List[int],
    key: Any,
    scale: float = 0.05,
) -> List[Tuple[Any, Any]]:
    r"""Initialize weights and biases for an MLP network with Xavier/Glorot-like scaling."""
    require_jax()
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = []
    for in_dim, out_dim, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
        w_key, b_key = jax.random.split(k)
        w = jax.random.normal(w_key, shape=(in_dim, out_dim)) * jnp.sqrt(2.0 / (in_dim + out_dim))
        b = jnp.zeros((out_dim,))
        params.append((w, b))
    return params


def mlp_forward(
    params: List[Tuple[Any, Any]],
    x: Any,
    activation: Optional[Callable[[Any], Any]] = None,
) -> Any:
    r"""Forward pass through an MLP with arbitrary activation."""
    require_jax()
    if activation is None:
        activation = jax.nn.silu

    h = x
    for i, (w, b) in enumerate(params):
        h = jnp.matmul(h, w) + b
        if i < len(params) - 1:
            h = activation(h)
    return h
