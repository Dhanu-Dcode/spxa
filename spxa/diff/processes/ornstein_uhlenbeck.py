"""Differentiable Ornstein-Uhlenbeck and Vasicek mean-reverting SDE."""

from __future__ import annotations

from typing import Any, Dict

from spxa.diff.config import require_jax
from spxa.diff.sde_base import DifferentiableSDE

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


class DifferentiableOU(DifferentiableSDE):
    r"""Differentiable Ornstein-Uhlenbeck (Vasicek) mean-reverting process:

    .. math::
        dX_t = \theta (\mu - X_t) dt + \sigma dW_t
    """

    def __init__(self) -> None:
        super().__init__(state_dim=1, noise_dim=1, has_jumps=False)

    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        theta = params.get("theta", 1.0)
        mu = params.get("mu", 0.0)
        return theta * (mu - x)

    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        sigma = params.get("sigma", 0.3)
        return jnp.full_like(x, sigma)

    def exact_step(self, t: Any, x: Any, dt: float | Any, z: Any, params: Dict[str, Any]) -> Any:
        r"""Exact Gaussian transition map."""
        theta = params.get("theta", 1.0)
        mu = params.get("mu", 0.0)
        sigma = params.get("sigma", 0.3)

        exp_neg = jnp.exp(-theta * dt)
        cond_mean = x * exp_neg + mu * (1.0 - exp_neg)
        cond_var = (sigma**2 / (2.0 * theta)) * (1.0 - jnp.exp(-2.0 * theta * dt))
        cond_std = jnp.sqrt(jnp.maximum(cond_var, 1e-14))
        return cond_mean + cond_std * z
