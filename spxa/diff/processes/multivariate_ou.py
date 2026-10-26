"""Differentiable Vector (Multivariate) Ornstein-Uhlenbeck process."""

from __future__ import annotations

from typing import Any, Dict

from spxa.diff.config import require_jax
from spxa.diff.sde_base import DifferentiableSDE

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


class DifferentiableVectorOU(DifferentiableSDE):
    r"""Differentiable d-dimensional Vector Ornstein-Uhlenbeck process:

    .. math::
        d\boldsymbol{X}_t = \boldsymbol{\Theta} (\boldsymbol{\mu} - \boldsymbol{X}_t) dt + \boldsymbol{\Sigma} d\boldsymbol{W}_t
    """

    def __init__(self, dim: int = 2) -> None:
        super().__init__(state_dim=dim, noise_dim=dim, has_jumps=False)

    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        # theta_matrix: (dim, dim)
        theta_mat = params.get("theta_matrix", jnp.eye(self.state_dim))
        mu_vec = params.get("mu_vector", jnp.zeros((self.state_dim,)))

        diff = mu_vec - x
        # x is (batch, dim), compute diff @ theta_mat.T -> (batch, dim)
        return jnp.matmul(diff, theta_mat.T)

    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        # sigma_matrix: (dim, dim)
        sigma_mat = params.get("sigma_matrix", 0.2 * jnp.eye(self.state_dim))
        # Broadcast to (batch, dim, dim)
        batch_size = x.shape[0]
        return jnp.broadcast_to(sigma_mat, (batch_size, self.state_dim, self.state_dim))

    def initial_state(self, batch_size: int, params: Dict[str, Any]) -> Any:
        x0 = params.get("x0", jnp.zeros((self.state_dim,)))
        return jnp.broadcast_to(x0, (batch_size, self.state_dim))
