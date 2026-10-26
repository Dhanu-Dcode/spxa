"""Differentiable Neural Stochastic Differential Equations (Neural SDEs)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from spxa.diff.config import require_jax
from spxa.diff.nn.mlp import init_mlp_params, mlp_forward
from spxa.diff.sde_base import DifferentiableSDE

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


class DifferentiableNeuralSDE(DifferentiableSDE):
    r"""Neural SDE with drift and diffusion parametrized by deep neural networks:

    .. math::
        d\boldsymbol{X}_t = \boldsymbol{f}_{\boldsymbol{\theta}}(t, \boldsymbol{X}_t) dt
                           + \boldsymbol{g}_{\boldsymbol{\phi}}(t, \boldsymbol{X}_t) d\boldsymbol{W}_t
    """

    def __init__(self, state_dim: int = 1, noise_dim: int = 1) -> None:
        super().__init__(state_dim=state_dim, noise_dim=noise_dim, has_jumps=False)

    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        drift_weights = params["drift_weights"]
        # Input features: [t, x]
        t_col = jnp.full((x.shape[0], 1), t)
        features = jnp.concatenate([t_col, x], axis=-1)
        return mlp_forward(drift_weights, features)

    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        diff_weights = params["diffusion_weights"]
        t_col = jnp.full((x.shape[0], 1), t)
        features = jnp.concatenate([t_col, x], axis=-1)
        raw_diff = mlp_forward(diff_weights, features)
        # Enforce positive diffusion via softplus
        return jax.nn.softplus(raw_diff) + 1e-4

    @staticmethod
    def init_params(
        state_dim: int,
        hidden_dim: int,
        key: Any,
    ) -> Dict[str, Any]:
        """Initialize parameters for both drift and diffusion networks."""
        require_jax()
        k1, k2 = jax.random.split(key)
        in_dim = state_dim + 1
        drift_weights = init_mlp_params([in_dim, hidden_dim, hidden_dim, state_dim], k1)
        diff_weights = init_mlp_params([in_dim, hidden_dim, hidden_dim, state_dim], k2)
        return {
            "drift_weights": drift_weights,
            "diffusion_weights": diff_weights,
            "x0": 0.0,
        }
