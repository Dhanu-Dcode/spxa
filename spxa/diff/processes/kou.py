"""Differentiable Kou (2002) Double-Exponential Jump Diffusion Model."""

from __future__ import annotations

from typing import Any, Dict

from spxa.diff.config import require_jax
from spxa.diff.sde_base import DifferentiableSDE

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


class DifferentiableKou(DifferentiableSDE):
    r"""Differentiable Kou Double-Exponential Jump Diffusion:

    .. math::
        \frac{dS_t}{S_{t^-}} = (r - q - \lambda \kappa_J) dt + \sigma dW_t + (e^Y - 1) dN_t
    where :math:`Y` has density :math:`f_Y(y) = p \eta_1 e^{-\eta_1 y} \mathbf{1}_{y \ge 0} + (1-p) \eta_2 e^{\eta_2 y} \mathbf{1}_{y < 0}`.
    """

    def __init__(self) -> None:
        super().__init__(state_dim=1, noise_dim=1, has_jumps=True)

    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        r = params.get("r", 0.05)
        q = params.get("q", 0.0)
        lam = params.get("lambda_j", 0.2)
        p = params.get("p_up", 0.4)
        eta1 = params.get("eta1", 10.0)  # > 1
        eta2 = params.get("eta2", 5.0)

        # Expected jump size kappa_J = p * eta1 / (eta1 - 1) + (1-p) * eta2 / (eta2 + 1) - 1
        k_j = p * (eta1 / (eta1 - 1.0)) + (1.0 - p) * (eta2 / (eta2 + 1.0)) - 1.0
        return (r - q - lam * k_j) * x

    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        sigma = params.get("sigma", 0.2)
        return sigma * x

    def jump_intensity(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        lam = params.get("lambda_j", 0.2)
        return jnp.full_like(x, lam)

    def jump_transform(self, t: Any, x: Any, shock: Any, params: Dict[str, Any]) -> Any:
        # shock is standard uniform U in (0, 1)
        p = params.get("p_up", 0.4)
        eta1 = params.get("eta1", 10.0)
        eta2 = params.get("eta2", 5.0)

        # Double exponential quantile transform
        u = jnp.clip(shock, 1e-7, 1.0 - 1e-7)
        # if u < p: Y = -ln(1 - u/p) / eta1; else: Y = ln((u-p)/(1-p)) / eta2
        y_pos = -jnp.log(jnp.maximum(1.0 - u / p, 1e-7)) / eta1
        y_neg = jnp.log(jnp.maximum((u - p) / (1.0 - p), 1e-7)) / eta2
        y = jnp.where(u < p, y_pos, y_neg)

        jump_factor = jnp.exp(y) - 1.0
        return x * jump_factor

    def project_state(self, x: Any) -> Any:
        return jnp.maximum(x, 1e-12)
