"""Differentiable Bates Stochastic Volatility Jump-Diffusion Model."""

from __future__ import annotations

from typing import Any, Dict

from spxa.diff.config import require_jax
from spxa.diff.sde_base import DifferentiableSDE

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


class DifferentiableBates(DifferentiableSDE):
    r"""Differentiable Bates Stochastic Volatility Jump-Diffusion Model:

    .. math::
        \frac{dS_t}{S_{t^-}} = (r - q - \lambda \kappa_J) dt + \sqrt{v_t} dW_t^S + (e^J - 1) dN_t
        dv_t = \kappa(\theta - v_t) dt + \sigma_v \sqrt{v_t} dW_t^v
    """

    def __init__(self) -> None:
        super().__init__(state_dim=2, noise_dim=2, has_jumps=True)

    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        s = x[..., 0]
        v = jnp.maximum(x[..., 1], 0.0)

        r = params.get("r", 0.03)
        q = params.get("q", 0.0)
        kappa = params.get("kappa", 1.5)
        theta = params.get("theta", 0.04)

        # Jump compensation
        lam = params.get("lambda_j", 0.1)
        mu_j = params.get("mu_j", -0.05)
        sigma_j = params.get("sigma_j", 0.1)
        k_j = jnp.exp(mu_j + 0.5 * sigma_j**2) - 1.0

        ds = (r - q - lam * k_j) * s
        dv = kappa * (theta - v)
        return jnp.stack([ds, dv], axis=-1)

    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        s = jnp.maximum(x[..., 0], 1e-12)
        v = jnp.maximum(x[..., 1], 0.0)
        sqrt_v = jnp.sqrt(v)

        sigma_v = params.get("sigma_v", 0.3)
        rho = params.get("rho", -0.7)

        g11 = sqrt_v * s
        g12 = jnp.zeros_like(s)
        g21 = rho * sigma_v * sqrt_v
        g22 = jnp.sqrt(jnp.maximum(1.0 - rho**2, 1e-12)) * sigma_v * sqrt_v

        row1 = jnp.stack([g11, g12], axis=-1)
        row2 = jnp.stack([g21, g22], axis=-1)
        return jnp.stack([row1, row2], axis=-2)

    def jump_intensity(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        lam = params.get("lambda_j", 0.1)
        return jnp.full_like(x[..., 0], lam)

    def jump_transform(self, t: Any, x: Any, shock: Any, params: Dict[str, Any]) -> Any:
        s = x[..., 0]
        mu_j = params.get("mu_j", -0.05)
        sigma_j = params.get("sigma_j", 0.1)

        jump_size = jnp.exp(mu_j + sigma_j * shock) - 1.0
        delta_s = s * jump_size
        delta_v = jnp.zeros_like(s)
        return jnp.stack([delta_s, delta_v], axis=-1)

    def project_state(self, x: Any) -> Any:
        s = jnp.maximum(x[..., 0], 1e-12)
        v = jnp.maximum(x[..., 1], 0.0)
        return jnp.stack([s, v], axis=-1)

    def initial_state(self, batch_size: int, params: Dict[str, Any]) -> Any:
        s0 = params.get("s0", 100.0)
        v0 = params.get("v0", 0.04)
        init_vec = jnp.array([s0, v0])
        return jnp.broadcast_to(init_vec, (batch_size, 2))
