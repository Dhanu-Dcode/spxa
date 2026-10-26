"""Differentiable Bessel and Squared Bessel (BESQ) diffusions."""

from __future__ import annotations

from typing import Any, Dict

from spxa.diff.config import require_jax
from spxa.diff.sde_base import DifferentiableSDE

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


class DifferentiableBESQ(DifferentiableSDE):
    r"""Differentiable Squared Bessel Process BESQ(dim):

    .. math::
        dX_t = \delta dt + 2 \sqrt{X_t} dW_t
    """

    def __init__(self) -> None:
        super().__init__(state_dim=1, noise_dim=1, has_jumps=False)

    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        delta = params.get("dim", 2.0)
        return jnp.full_like(x, delta)

    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        pos_x = jnp.maximum(x, 0.0)
        return 2.0 * jnp.sqrt(pos_x)

    def project_state(self, x: Any) -> Any:
        return jnp.maximum(x, 0.0)


class DifferentiableBessel(DifferentiableSDE):
    r"""Differentiable Bessel Process BES(dim):

    .. math::
        dR_t = \frac{\delta - 1}{2 R_t} dt + dW_t
    """

    def __init__(self) -> None:
        super().__init__(state_dim=1, noise_dim=1, has_jumps=False)

    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        delta = params.get("dim", 2.0)
        pos_x = jnp.maximum(x, 1e-8)
        return (delta - 1.0) / (2.0 * pos_x)

    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        return jnp.ones_like(x)

    def project_state(self, x: Any) -> Any:
        return jnp.maximum(x, 1e-8)
