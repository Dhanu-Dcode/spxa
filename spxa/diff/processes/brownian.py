"""Differentiable Geometric Brownian Motion (Black-Scholes-Merton) SDE."""

from __future__ import annotations

from typing import Any, Dict

from spxa.diff.sde_base import DifferentiableSDE

try:
    import jax.numpy as jnp
except ImportError:
    pass


class DifferentiableGBM(DifferentiableSDE):
    r"""Differentiable Geometric Brownian Motion:

    .. math::
        dS_t = \mu S_t dt + \sigma S_t dW_t
    """

    def __init__(self) -> None:
        super().__init__(state_dim=1, noise_dim=1, has_jumps=False)

    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        mu = params.get("mu", 0.05)
        return mu * x

    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        sigma = params.get("sigma", 0.20)
        return sigma * x

    def project_state(self, x: Any) -> Any:
        """Enforce positivity via smooth softplus or non-negative clamp."""
        return jnp.maximum(x, 1e-12)

    def exact_step(self, t: Any, x: Any, dt: float | Any, dw: Any, params: Dict[str, Any]) -> Any:
        r"""Exact log-normal transition:

        .. math::
            S_{t+\Delta t} = S_t \exp\left( (\mu - \frac{1}{2}\sigma^2)\Delta t + \sigma \Delta W_t \right)
        """
        mu = params.get("mu", 0.05)
        sigma = params.get("sigma", 0.20)
        return x * jnp.exp((mu - 0.5 * sigma**2) * dt + sigma * dw)
