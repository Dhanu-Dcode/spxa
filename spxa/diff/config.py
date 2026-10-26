"""Runtime configuration and device/precision management for differentiable SDEs."""

from __future__ import annotations

import os
from typing import Any, Optional

_HAS_JAX: Optional[bool] = None


def has_jax() -> bool:
    """Return True if JAX and JAXLib are installed and available."""
    global _HAS_JAX
    if _HAS_JAX is None:
        try:
            import jax  # noqa: F401
            import jax.numpy as jnp  # noqa: F401

            _HAS_JAX = True
        except ImportError:
            _HAS_JAX = False
    return _HAS_JAX


def enable_x64(enabled: bool = True) -> None:
    """Enable or disable 64-bit floating point precision in JAX."""
    if has_jax():
        import jax

        jax.config.update("jax_enable_x64", enabled)


def get_available_devices() -> list[str]:
    """Return list of available device types (e.g. ['cpu', 'gpu', 'tpu'])."""
    if not has_jax():
        return ["cpu"]
    import jax

    devices = jax.devices()
    return sorted(list({d.platform for d in devices}))


def default_backend() -> str:
    """Return the default hardware acceleration backend ('gpu', 'tpu', or 'cpu')."""
    if not has_jax():
        return "cpu"
    import jax

    try:
        backend = jax.default_backend()
        return str(backend)
    except Exception:
        return "cpu"


def require_jax() -> None:
    """Raise ImportError if JAX is not available."""
    if not has_jax():
        raise ImportError(
            "The spxa.diff module requires JAX. "
            "Please install JAX via `pip install spxa[diff]` or `pip install jax jaxlib`."
        )
