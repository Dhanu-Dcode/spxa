"""Differentiable 2D Heston Stochastic Volatility Model."""

from __future__ import annotations

from typing import Any, Dict

from spxa.diff.sde_base import DifferentiableSDE

try:
    import jax.numpy as jnp
except ImportError:
    pass


class DifferentiableHeston(DifferentiableSDE):
    r"""Differentiable 2-Dimensional Heston Stochastic Volatility Model:

    .. math::
        d\boldsymbol{X}_t = \begin{pmatrix} dS_t \\ dv_t \end{pmatrix}
        = \begin{pmatrix} \mu S_t \\ \kappa(\theta - v_t) \end{pmatrix} dt
        + \begin{pmatrix} \sqrt{v_t} S_t & 0 \\ \rho \sigma_v \sqrt{v_t} & \sqrt{1 - \rho^2}\sigma_v \sqrt{v_t} \end{pmatrix}
          \begin{pmatrix} dW_t^1 \\ dW_t^2 \end{pmatrix}
    """

    def __init__(self) -> None:
        super().__init__(state_dim=2, noise_dim=2, has_jumps=False)

    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        s = x[..., 0]
        v = jnp.maximum(x[..., 1], 0.0)

        mu = params.get("mu", 0.05)
        kappa = params.get("kappa", 1.5)
        theta = params.get("theta", 0.04)

        ds = mu * s
        dv = kappa * (theta - v)
        return jnp.stack([ds, dv], axis=-1)

    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        s = jnp.maximum(x[..., 0], 1e-12)
        v = jnp.maximum(x[..., 1], 0.0)
        sqrt_v = jnp.sqrt(v)

        sigma_v = params.get("sigma_v", 0.3)
        rho = params.get("rho", -0.7)

        # Cholesky factor matrix (batch, 2, 2)
        g11 = sqrt_v * s
        g12 = jnp.zeros_like(s)
        g21 = rho * sigma_v * sqrt_v
        g22 = jnp.sqrt(jnp.maximum(1.0 - rho**2, 1e-12)) * sigma_v * sqrt_v

        row1 = jnp.stack([g11, g12], axis=-1)
        row2 = jnp.stack([g21, g22], axis=-1)
        return jnp.stack([row1, row2], axis=-2)

    def project_state(self, x: Any) -> Any:
        s = jnp.maximum(x[..., 0], 1e-12)
        v = jnp.maximum(x[..., 1], 0.0)
        return jnp.stack([s, v], axis=-1)

    def initial_state(self, batch_size: int, params: Dict[str, Any]) -> Any:
        s0 = params.get("s0", 100.0)
        v0 = params.get("v0", 0.04)
        init_vec = jnp.array([s0, v0])
        return jnp.broadcast_to(init_vec, (batch_size, 2))
