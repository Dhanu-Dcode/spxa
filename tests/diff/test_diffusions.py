"""Tests for 1D differentiable continuous diffusion processes."""

import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from spxa.diff.processes.bessel import DifferentiableBESQ
from spxa.diff.processes.bilinear import DifferentiableBilinear
from spxa.diff.processes.cev import DifferentiableCEV
from spxa.diff.processes.cir import DifferentiableCIR
from spxa.diff.processes.ornstein_uhlenbeck import DifferentiableOU
from spxa.diff.prng import make_key
from spxa.diff.solvers import solve_euler_maruyama


def test_diffusions_execution():
    key = make_key(303)

    # CIR
    cir = DifferentiableCIR()
    p_cir = {"kappa": 1.5, "theta": 0.04, "sigma": 0.2, "x0": 0.04}
    traj_cir, key = solve_euler_maruyama(cir, p_cir, (0.0, 1.0), 50, 100, key)
    assert jnp.all(traj_cir >= 0.0)

    # OU
    ou = DifferentiableOU()
    p_ou = {"theta": 1.0, "mu": 2.0, "sigma": 0.3, "x0": 1.0}
    traj_ou, key = solve_euler_maruyama(ou, p_ou, (0.0, 1.0), 50, 100, key)
    assert traj_ou.shape == (100, 51, 1)

    # BESQ & Bessel
    besq = DifferentiableBESQ()
    p_besq = {"dim": 3.0, "x0": 1.0}
    traj_besq, key = solve_euler_maruyama(besq, p_besq, (0.0, 1.0), 50, 100, key)
    assert jnp.all(traj_besq >= 0.0)

    # CEV
    cev = DifferentiableCEV()
    p_cev = {"mu": 0.05, "sigma": 0.2, "gamma": 0.8, "x0": 100.0}
    traj_cev, key = solve_euler_maruyama(cev, p_cev, (0.0, 1.0), 50, 100, key)
    assert jnp.all(traj_cev > 0.0)

    # Bilinear
    bilin = DifferentiableBilinear()
    p_bilin = {"a": 0.1, "b": -0.5, "c": 0.05, "d": 0.2, "x0": 1.0}
    traj_bilin, key = solve_euler_maruyama(bilin, p_bilin, (0.0, 1.0), 50, 100, key)
    assert traj_bilin.shape == (100, 51, 1)
