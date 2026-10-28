"""Differentiable Cox-Ingersoll-Ross (CIR) square-root diffusion process."""

from __future__ import annotations

from typing import Any, Dict

from spxa.diff.sde_base import DifferentiableSDE

try:
    import jax.numpy as jnp
except ImportError:
    pass


class DifferentiableCIR(DifferentiableSDE):
    r"""Differentiable Cox-Ingersoll-Ross (CIR) process:

    .. math::
        dv_t = \kappa (\theta - v_t) dt + \sigma \sqrt{\max(v_t, 0)} dW_t
    """

    def __init__(self) -> None:
        super().__init__(state_dim=1, noise_dim=1, has_jumps=False)

    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        kappa = params.get("kappa", 1.5)
        theta = params.get("theta", 0.04)
        return kappa * (theta - jnp.maximum(x, 0.0))

    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        sigma = params.get("sigma", 0.2)
        pos_x = jnp.maximum(x, 0.0)
        return sigma * jnp.sqrt(pos_x)

    def project_state(self, x: Any) -> Any:
        """Full truncation non-negative projection."""
        return jnp.maximum(x, 0.0)
