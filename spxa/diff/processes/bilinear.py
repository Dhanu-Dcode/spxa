"""Differentiable Bilinear Diffusion SDE."""

from __future__ import annotations

from typing import Any, Dict

from spxa.diff.config import require_jax
from spxa.diff.sde_base import DifferentiableSDE

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


class DifferentiableBilinear(DifferentiableSDE):
    r"""Differentiable Bilinear Diffusion:

    .. math::
        dX_t = (a + b X_t) dt + (c + d X_t) dW_t
    """

    def __init__(self) -> None:
        super().__init__(state_dim=1, noise_dim=1, has_jumps=False)

    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        a = params.get("a", 0.1)
        b = params.get("b", -0.5)
        return a + b * x

    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        c = params.get("c", 0.05)
        d = params.get("d", 0.2)
        return c + d * x
