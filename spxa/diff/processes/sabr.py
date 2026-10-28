"""Differentiable Hagan SABR Stochastic Volatility Model."""

from __future__ import annotations

from typing import Any, Dict

from spxa.diff.sde_base import DifferentiableSDE

try:
    import jax.numpy as jnp
except ImportError:
    pass


class DifferentiableSABR(DifferentiableSDE):
    r"""Differentiable 2D SABR Stochastic Volatility Model:

    .. math::
        dF_t = \sigma_t F_t^\beta dW_t^1
        d\sigma_t = \alpha \sigma_t (\rho dW_t^1 + \sqrt{1-\rho^2} dW_t^2)
    """

    def __init__(self) -> None:
        super().__init__(state_dim=2, noise_dim=2, has_jumps=False)

    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        return jnp.zeros_like(x)

    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        f = jnp.maximum(x[..., 0], 1e-12)
        sig = jnp.maximum(x[..., 1], 1e-12)

        beta = params.get("beta", 0.5)
        alpha = params.get("alpha", 0.4)
        rho = params.get("rho", -0.3)

        # Cholesky factor matrix (batch, 2, 2)
        g11 = sig * (f**beta)
        g12 = jnp.zeros_like(f)
        g21 = rho * alpha * sig
        g22 = jnp.sqrt(jnp.maximum(1.0 - rho**2, 1e-12)) * alpha * sig

        row1 = jnp.stack([g11, g12], axis=-1)
        row2 = jnp.stack([g21, g22], axis=-1)
        return jnp.stack([row1, row2], axis=-2)

    def project_state(self, x: Any) -> Any:
        f = jnp.maximum(x[..., 0], 1e-12)
        sig = jnp.maximum(x[..., 1], 1e-12)
        return jnp.stack([f, sig], axis=-1)

    def initial_state(self, batch_size: int, params: Dict[str, Any]) -> Any:
        f0 = params.get("f0", 100.0)
        sigma0 = params.get("sigma0", 0.2)
        init_vec = jnp.array([f0, sigma0])
        return jnp.broadcast_to(init_vec, (batch_size, 2))
