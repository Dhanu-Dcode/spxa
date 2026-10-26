"""Differentiable Self-Exciting Hawkes Process."""

from __future__ import annotations

from typing import Any, Dict

from spxa.diff.config import require_jax
from spxa.diff.sde_base import DifferentiableSDE

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


class DifferentiableHawkes(DifferentiableSDE):
    r"""Differentiable Self-Exciting Hawkes Intensity Process:

    .. math::
        d\lambda_t = \alpha (\mu - \lambda_t) dt + \beta dN_t
    where :math:`dN_t \sim \text{Poisson}(\lambda_t dt)`.
    """

    def __init__(self) -> None:
        super().__init__(state_dim=1, noise_dim=1, has_jumps=True)

    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        alpha = params.get("alpha", 1.5)
        mu = params.get("mu", 0.5)
        return alpha * (mu - x)

    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        # Zero continuous Brownian diffusion
        return jnp.zeros_like(x)

    def jump_intensity(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        # Self-exciting intensity is the state itself
        return jnp.maximum(x[..., 0], 1e-6)

    def jump_transform(self, t: Any, x: Any, shock: Any, params: Dict[str, Any]) -> Any:
        beta = params.get("beta", 0.8)
        return jnp.full_like(x, beta)

    def project_state(self, x: Any) -> Any:
        return jnp.maximum(x, 1e-6)
