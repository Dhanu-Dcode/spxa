"""Differentiable Merton Jump Diffusion Model."""

from __future__ import annotations

from typing import Any, Dict

from spxa.diff.sde_base import DifferentiableSDE

try:
    import jax.numpy as jnp
except ImportError:
    pass


class DifferentiableMerton(DifferentiableSDE):
    r"""Differentiable Merton (1976) Jump Diffusion Model:

    .. math::
        \frac{dS_t}{S_{t^-}} = (r - q - \lambda \kappa_J) dt + \sigma dW_t + (e^Y - 1) dN_t
    """

    def __init__(self) -> None:
        super().__init__(state_dim=1, noise_dim=1, has_jumps=True)

    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        r = params.get("r", 0.05)
        q = params.get("q", 0.0)
        lam = params.get("lambda_j", 0.2)
        mu_j = params.get("mu_j", -0.1)
        sigma_j = params.get("sigma_j", 0.15)
        k_j = jnp.exp(mu_j + 0.5 * sigma_j**2) - 1.0
        return (r - q - lam * k_j) * x

    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        sigma = params.get("sigma", 0.2)
        return sigma * x

    def jump_intensity(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        lam = params.get("lambda_j", 0.2)
        return jnp.full_like(x, lam)

    def jump_transform(self, t: Any, x: Any, shock: Any, params: Dict[str, Any]) -> Any:
        mu_j = params.get("mu_j", -0.1)
        sigma_j = params.get("sigma_j", 0.15)
        jump_factor = jnp.exp(mu_j + sigma_j * shock) - 1.0
        return x * jump_factor

    def project_state(self, x: Any) -> Any:
        return jnp.maximum(x, 1e-12)
