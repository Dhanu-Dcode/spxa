"""Tests for spxa.diff configuration and PRNG utilities."""

import pytest

from spxa.diff.config import (
    default_backend,
    enable_x64,
    get_available_devices,
    has_jax,
    require_jax,
)
from spxa.diff.prng import (
    generate_gamma,
    generate_normal,
    generate_poisson,
    generate_uniform,
    make_key,
    split_key,
)


def test_config_jax_presence():
    assert has_jax() is True
    require_jax()  # should not raise
    backend = default_backend()
    assert isinstance(backend, str)
    devices = get_available_devices()
    assert len(devices) >= 1
    enable_x64(True)


def test_prng_generation():
    key = make_key(42)
    k1, k2 = split_key(key, 2)
    assert k1 is not None and k2 is not None

    # Normal samples
    z, key = generate_normal(key, shape=(100, 2), antithetic=False)
    assert z.shape == (100, 2)

    # Antithetic normal samples
    z_anti, key = generate_normal(key, shape=(100, 2), antithetic=True)
    assert z_anti.shape == (100, 2)
    # Check antithetic property: z_anti[:50] == -z_anti[50:]
    import jax.numpy as jnp

    assert jnp.allclose(z_anti[:50], -z_anti[50:])

    # Uniform samples
    u, key = generate_uniform(key, shape=(50,), minval=1.0, maxval=5.0)
    assert u.shape == (50,)
    assert jnp.all(u >= 1.0) and jnp.all(u <= 5.0)

    # Poisson samples
    p, key = generate_poisson(key, lam=2.5, shape=(50,))
    assert p.shape == (50,)
    assert jnp.all(p >= 0)

    # Gamma samples
    g, key = generate_gamma(key, shape_param=2.0, shape=(50,))
    assert g.shape == (50,)
    assert jnp.all(g > 0.0)
