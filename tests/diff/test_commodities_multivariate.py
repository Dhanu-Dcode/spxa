"""Tests for differentiable commodity and multivariate diffusion processes."""

import jax
import jax.numpy as jnp
import pytest

from spxa.diff.processes.multivariate_ou import DifferentiableVectorOU
from spxa.diff.processes.schwartz import (
    DifferentiableSchwartz1Factor,
    DifferentiableSchwartz2Factor,
)
from spxa.diff.prng import make_key
from spxa.diff.solvers import solve_euler_maruyama


def test_schwartz_and_vector_ou():
    key = make_key(606)

    # Schwartz 1-Factor
    sch1 = DifferentiableSchwartz1Factor()
    p1 = {"kappa": 1.0, "mu": 3.0, "sigma": 0.3, "x0": 20.0}
    traj1, key = solve_euler_maruyama(sch1, p1, (0.0, 1.0), 30, 50, key)
    assert traj1.shape == (50, 31, 1)
    assert jnp.all(traj1 > 0.0)

    # Schwartz 2-Factor
    sch2 = DifferentiableSchwartz2Factor()
    p2 = {"r": 0.05, "kappa": 1.2, "alpha": 0.08, "sigma1": 0.25, "sigma2": 0.35, "rho": 0.3, "s0": 50.0, "delta0": 0.05}
    traj2, key = solve_euler_maruyama(sch2, p2, (0.0, 1.0), 30, 50, key)
    assert traj2.shape == (50, 31, 2)
    assert jnp.all(traj2[..., 0] > 0.0)

    # Vector OU
    dim = 3
    vec_ou = DifferentiableVectorOU(dim=dim)
    p_vec = {
        "theta_matrix": jnp.eye(dim),
        "mu_vector": jnp.zeros((dim,)),
        "sigma_matrix": 0.1 * jnp.eye(dim),
        "x0": jnp.ones((dim,)),
    }
    traj_vec, key = solve_euler_maruyama(vec_ou, p_vec, (0.0, 1.0), 30, 50, key)
    assert traj_vec.shape == (50, 31, dim)
