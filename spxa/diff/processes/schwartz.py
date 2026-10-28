"""Differentiable Schwartz 1-Factor and 2-Factor Commodity Models."""

from __future__ import annotations

from typing import Any, Dict

from spxa.diff.sde_base import DifferentiableSDE

try:
    import jax.numpy as jnp
except ImportError:
    pass


class DifferentiableSchwartz1Factor(DifferentiableSDE):
    r"""Differentiable Schwartz (1997) 1-Factor Commodity Model:

    .. math::
        d\ln S_t = \kappa (\mu - \ln S_t) dt + \sigma dW_t
        \implies dS_t = \kappa (\mu - \ln S_t + \frac{\sigma^2}{2\kappa}) S_t dt + \sigma S_t dW_t
    """

    def __init__(self) -> None:
        super().__init__(state_dim=1, noise_dim=1, has_jumps=False)

    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        kappa = params.get("kappa", 1.0)
        mu = params.get("mu", 3.0)
        sigma = params.get("sigma", 0.3)
        pos_x = jnp.maximum(x, 1e-12)
        log_s = jnp.log(pos_x)
        return kappa * (mu - log_s + (sigma**2) / (2.0 * kappa)) * pos_x

    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        sigma = params.get("sigma", 0.3)
        pos_x = jnp.maximum(x, 1e-12)
        return sigma * pos_x

    def project_state(self, x: Any) -> Any:
        return jnp.maximum(x, 1e-12)


class DifferentiableSchwartz2Factor(DifferentiableSDE):
    r"""Differentiable Schwartz (1997) 2-Factor Commodity Model:

    .. math::
        dS_t = (r - \delta_t) S_t dt + \sigma_1 S_t dW_t^1
        d\delta_t = \kappa (\alpha - \delta_t) dt + \sigma_2 (\rho dW_t^1 + \sqrt{1-\rho^2} dW_t^2)
    """

    def __init__(self) -> None:
        super().__init__(state_dim=2, noise_dim=2, has_jumps=False)

    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        s = x[..., 0]
        delta = x[..., 1]

        r = params.get("r", 0.05)
        kappa = params.get("kappa", 1.2)
        alpha = params.get("alpha", 0.08)

        ds = (r - delta) * s
        ddelta = kappa * (alpha - delta)
        return jnp.stack([ds, ddelta], axis=-1)

    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        s = jnp.maximum(x[..., 0], 1e-12)

        sigma1 = params.get("sigma1", 0.25)
        sigma2 = params.get("sigma2", 0.35)
        rho = params.get("rho", 0.3)

        g11 = sigma1 * s
        g12 = jnp.zeros_like(s)
        g21 = rho * sigma2 * jnp.ones_like(s)
        g22 = jnp.sqrt(jnp.maximum(1.0 - rho**2, 1e-12)) * sigma2 * jnp.ones_like(s)

        row1 = jnp.stack([g11, g12], axis=-1)
        row2 = jnp.stack([g21, g22], axis=-1)
        return jnp.stack([row1, row2], axis=-2)

    def project_state(self, x: Any) -> Any:
        s = jnp.maximum(x[..., 0], 1e-12)
        delta = x[..., 1]
        return jnp.stack([s, delta], axis=-1)

    def initial_state(self, batch_size: int, params: Dict[str, Any]) -> Any:
        s0 = params.get("s0", 50.0)
        delta0 = params.get("delta0", 0.05)
        init_vec = jnp.array([s0, delta0])
        return jnp.broadcast_to(init_vec, (batch_size, 2))
