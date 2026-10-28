"""Differentiable 3/2 Stochastic Volatility Model."""

from __future__ import annotations

from typing import Any, Dict

from spxa.diff.sde_base import DifferentiableSDE

try:
    import jax.numpy as jnp
except ImportError:
    pass


class DifferentiableThreeHalves(DifferentiableSDE):
    r"""Differentiable 3/2 Stochastic Volatility Model:

    .. math::
        dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^S
        dv_t = \kappa v_t (\theta - v_t) dt + \epsilon v_t^{3/2} dW_t^v
    """

    def __init__(self) -> None:
        super().__init__(state_dim=2, noise_dim=2, has_jumps=False)

    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        s = x[..., 0]
        v = jnp.maximum(x[..., 1], 1e-8)

        mu = params.get("mu", 0.05)
        kappa = params.get("kappa", 2.0)
        theta = params.get("theta", 0.04)

        ds = mu * s
        dv = kappa * v * (theta - v)
        return jnp.stack([ds, dv], axis=-1)

    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        s = jnp.maximum(x[..., 0], 1e-12)
        v = jnp.maximum(x[..., 1], 1e-8)
        sqrt_v = jnp.sqrt(v)
        v_3_2 = v * sqrt_v

        epsilon = params.get("epsilon", 0.4)
        rho = params.get("rho", -0.5)

        g11 = sqrt_v * s
        g12 = jnp.zeros_like(s)
        g21 = rho * epsilon * v_3_2
        g22 = jnp.sqrt(jnp.maximum(1.0 - rho**2, 1e-12)) * epsilon * v_3_2

        row1 = jnp.stack([g11, g12], axis=-1)
        row2 = jnp.stack([g21, g22], axis=-1)
        return jnp.stack([row1, row2], axis=-2)

    def project_state(self, x: Any) -> Any:
        s = jnp.maximum(x[..., 0], 1e-12)
        v = jnp.maximum(x[..., 1], 1e-8)
        return jnp.stack([s, v], axis=-1)

    def initial_state(self, batch_size: int, params: Dict[str, Any]) -> Any:
        s0 = params.get("s0", 100.0)
        v0 = params.get("v0", 0.04)
        init_vec = jnp.array([s0, v0])
        return jnp.broadcast_to(init_vec, (batch_size, 2))
