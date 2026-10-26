"""Tests for differentiable stochastic volatility systems (Heston, SABR, 3/2)."""

import jax
import jax.numpy as jnp
import pytest

from spxa.diff.processes.heston import DifferentiableHeston
from spxa.diff.processes.sabr import DifferentiableSABR
from spxa.diff.processes.three_halves import DifferentiableThreeHalves
from spxa.diff.prng import make_key
from spxa.diff.solvers import solve_euler_maruyama


def test_stochastic_vol_models():
    key = make_key(404)

    # Heston
    heston = DifferentiableHeston()
    p_heston = {
        "mu": 0.05,
        "kappa": 1.5,
        "theta": 0.04,
        "sigma_v": 0.3,
        "rho": -0.7,
        "s0": 100.0,
        "v0": 0.04,
    }
    traj_h, key = solve_euler_maruyama(heston, p_heston, (0.0, 1.0), 30, 50, key)
    assert traj_h.shape == (50, 31, 2)
    # Check S > 0 and v >= 0
    assert jnp.all(traj_h[..., 0] > 0.0)
    assert jnp.all(traj_h[..., 1] >= 0.0)

    # SABR
    sabr = DifferentiableSABR()
    p_sabr = {"beta": 0.7, "alpha": 0.4, "rho": -0.3, "f0": 100.0, "sigma0": 0.2}
    traj_s, key = solve_euler_maruyama(sabr, p_sabr, (0.0, 1.0), 30, 50, key)
    assert traj_s.shape == (50, 31, 2)
    assert jnp.all(traj_s > 0.0)

    # 3/2
    three_halves = DifferentiableThreeHalves()
    p_th = {
        "mu": 0.05,
        "kappa": 2.0,
        "theta": 0.04,
        "epsilon": 0.3,
        "rho": -0.5,
        "s0": 100.0,
        "v0": 0.04,
    }
    traj_th, key = solve_euler_maruyama(three_halves, p_th, (0.0, 1.0), 30, 50, key)
    assert traj_th.shape == (50, 31, 2)
    assert jnp.all(traj_th > 0.0)
