"""Differentiable Constant Elasticity of Variance (CEV) diffusion model."""

from __future__ import annotations

from typing import Any, Dict

from spxa.diff.config import require_jax
from spxa.diff.sde_base import DifferentiableSDE

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


class DifferentiableCEV(DifferentiableSDE):
    r"""Differentiable Constant Elasticity of Variance (CEV) model:

    .. math::
        dS_t = \mu S_t dt + \sigma S_t^\gamma dW_t
    """

    def __init__(self) -> None:
        super().__init__(state_dim=1, noise_dim=1, has_jumps=False)

    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        mu = params.get("mu", 0.05)
        return mu * x

    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        sigma = params.get("sigma", 0.25)
        gamma = params.get("gamma", 0.75)
        pos_x = jnp.maximum(x, 1e-12)
        return sigma * (pos_x**gamma)

    def project_state(self, x: Any) -> Any:
        return jnp.maximum(x, 1e-12)
