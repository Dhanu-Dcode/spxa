"""Abstract base class and protocol for differentiable SDE specifications."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

from spxa.diff.config import require_jax

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass


class DifferentiableSDE(ABC):
    r"""Abstract base class for all differentiable Itô and Jump-Diffusion SDEs.

    Represents multidimensional SDE systems of the form:
    .. math::
        d\boldsymbol{X}_t = \boldsymbol{f}(t, \boldsymbol{X}_t, \boldsymbol{\theta}) dt
                           + \boldsymbol{g}(t, \boldsymbol{X}_t, \boldsymbol{\theta}) d\boldsymbol{W}_t
                           + \boldsymbol{J}(t, \boldsymbol{X}_t, \boldsymbol{Z}_t, \boldsymbol{\theta}) dN_t
    """

    def __init__(
        self,
        state_dim: int = 1,
        noise_dim: int = 1,
        has_jumps: bool = False,
    ) -> None:
        require_jax()
        self._state_dim = int(state_dim)
        self._noise_dim = int(noise_dim)
        self._has_jumps = bool(has_jumps)

    @property
    def state_dim(self) -> int:
        """Dimensionality of the continuous state vector X."""
        return self._state_dim

    @property
    def noise_dim(self) -> int:
        """Dimensionality of the driving Brownian motion W."""
        return self._noise_dim

    @property
    def has_jumps(self) -> bool:
        """True if the SDE incorporates jump discontinuities."""
        return self._has_jumps

    @abstractmethod
    def drift(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        r"""Compute deterministic drift vector :math:`\boldsymbol{f}(t, \boldsymbol{X}_t, \boldsymbol{\theta})`."""
        pass

    @abstractmethod
    def diffusion(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        r"""Compute diffusion matrix/vector :math:`\boldsymbol{g}(t, \boldsymbol{X}_t, \boldsymbol{\theta})`."""
        pass

    def jump_intensity(self, t: Any, x: Any, params: Dict[str, Any]) -> Any:
        r"""Arrival intensity rate :math:`\lambda(t, \boldsymbol{X}_t, \boldsymbol{\theta})`."""
        return jnp.zeros_like(x[..., 0])

    def jump_transform(self, t: Any, x: Any, shock: Any, params: Dict[str, Any]) -> Any:
        r"""Jump size map :math:`\boldsymbol{J}(t, \boldsymbol{X}_t, \boldsymbol{Z}_t, \boldsymbol{\theta})`."""
        return jnp.zeros_like(x)

    def project_state(self, x: Any) -> Any:
        r"""Optional state boundary projection / constraint enforcement (e.g. non-negativity)."""
        return x

    def initial_state(self, batch_size: int, params: Dict[str, Any]) -> Any:
        r"""Construct initial state tensor :math:`\boldsymbol{X}_0` of shape `(batch_size, state_dim)`."""
        x0_val = params.get("x0", 0.0)
        if jnp.ndim(x0_val) == 0:
            return jnp.full((batch_size, self.state_dim), x0_val)
        return jnp.broadcast_to(x0_val, (batch_size, self.state_dim))
