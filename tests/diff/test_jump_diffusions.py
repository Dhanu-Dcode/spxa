"""Tests for differentiable jump-diffusion and point processes."""

import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from spxa.diff.processes.bates import DifferentiableBates
from spxa.diff.processes.hawkes import DifferentiableHawkes
from spxa.diff.processes.kou import DifferentiableKou
from spxa.diff.processes.merton import DifferentiableMerton
from spxa.diff.prng import make_key
from spxa.diff.solvers import solve_euler_maruyama


def test_jump_diffusions():
    key = make_key(505)

    # Merton
    merton = DifferentiableMerton()
    p_merton = {"r": 0.05, "sigma": 0.2, "lambda_j": 0.2, "mu_j": -0.1, "sigma_j": 0.1, "x0": 100.0}
    traj_m, key = solve_euler_maruyama(merton, p_merton, (0.0, 1.0), 30, 50, key)
    assert traj_m.shape == (50, 31, 1)
    assert jnp.all(traj_m > 0.0)

    # Kou
    kou = DifferentiableKou()
    p_kou = {
        "r": 0.05,
        "sigma": 0.2,
        "lambda_j": 0.2,
        "p_up": 0.4,
        "eta1": 10.0,
        "eta2": 5.0,
        "x0": 100.0,
    }
    traj_k, key = solve_euler_maruyama(kou, p_kou, (0.0, 1.0), 30, 50, key)
    assert traj_k.shape == (50, 31, 1)
    assert jnp.all(traj_k > 0.0)

    # Bates
    bates = DifferentiableBates()
    p_bates = {
        "r": 0.03,
        "kappa": 1.5,
        "theta": 0.04,
        "sigma_v": 0.3,
        "rho": -0.7,
        "lambda_j": 0.1,
        "mu_j": -0.05,
        "sigma_j": 0.1,
        "s0": 100.0,
        "v0": 0.04,
    }
    traj_b, key = solve_euler_maruyama(bates, p_bates, (0.0, 1.0), 30, 50, key)
    assert traj_b.shape == (50, 31, 2)
    assert jnp.all(traj_b[..., 0] > 0.0)
    assert jnp.all(traj_b[..., 1] >= 0.0)

    # Hawkes
    hawkes = DifferentiableHawkes()
    p_hawkes = {"alpha": 1.5, "mu": 0.5, "beta": 0.8, "x0": 0.5}
    traj_h, key = solve_euler_maruyama(hawkes, p_hawkes, (0.0, 1.0), 30, 50, key)
    assert traj_h.shape == (50, 31, 1)
    assert jnp.all(traj_h > 0.0)
