"""Tests for differentiable rough volatility and fractional Brownian motion."""

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from spxa.diff.processes.fbm import simulate_diff_fbm
from spxa.diff.processes.rough_heston import simulate_diff_rough_heston
from spxa.diff.prng import make_key


def test_fractional_brownian_motion():
    key = make_key(707)
    # Test subdiffusive rough H = 0.3
    paths, key = simulate_diff_fbm(hurst=0.3, t_span=(0.0, 1.0), n_steps=20, n_paths=30, key=key)
    assert paths.shape == (30, 21)
    assert jnp.allclose(paths[:, 0], 0.0)

    # Test Hurst parameter differentiability
    def mean_fbm_norm(h_val):
        p, _ = simulate_diff_fbm(hurst=h_val, t_span=(0.0, 1.0), n_steps=10, n_paths=15, key=key)
        return jnp.mean(p[:, -1] ** 2)

    grad_h = jax.grad(mean_fbm_norm)(0.35)
    assert jnp.isfinite(grad_h)


def test_rough_heston_simulation():
    key = make_key(808)
    params = {
        "s0": 100.0,
        "v0": 0.04,
        "lambda": 1.5,
        "theta": 0.04,
        "nu": 0.3,
        "rho": -0.7,
        "hurst": 0.15,
        "r": 0.03,
    }
    traj, _ = simulate_diff_rough_heston(params, (0.0, 1.0), n_steps=15, n_paths=20, key=key)
    assert traj.shape == (20, 16, 2)
    # Verify spot positive and variance non-negative
    assert jnp.all(traj[..., 0] > 0.0)
    assert jnp.all(traj[..., 1] >= 0.0)
