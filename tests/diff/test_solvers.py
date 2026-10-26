"""Tests for differentiable numerical SDE solvers."""

import jax
import jax.numpy as jnp
import pytest

from spxa.diff.processes.brownian import DifferentiableGBM
from spxa.diff.prng import make_key
from spxa.diff.solvers import (
    solve_euler_maruyama,
    solve_milstein,
    solve_reversible_heun,
    solve_sde,
    solve_srk,
)


def test_solvers_gbm_trajectory():
    sde = DifferentiableGBM()
    params = {"mu": 0.05, "sigma": 0.2, "x0": 100.0}
    key = make_key(101)

    for method in ["euler", "milstein", "srk", "heun"]:
        traj, _ = solve_sde(
            sde=sde,
            params=params,
            t_span=(0.0, 1.0),
            n_steps=50,
            n_paths=200,
            key=key,
            method=method,
            return_trajectory=True,
        )
        assert traj.shape == (200, 51, 1)
        # All simulated prices should remain positive
        assert jnp.all(traj > 0.0)


def test_solvers_gradient_backprop():
    sde = DifferentiableGBM()
    key = make_key(202)

    def mean_terminal(sigma_val):
        p = {"mu": 0.05, "sigma": sigma_val, "x0": 100.0}
        term, _ = solve_euler_maruyama(
            sde=sde,
            params=p,
            t_span=(0.0, 1.0),
            n_steps=20,
            n_paths=50,
            key=key,
            return_trajectory=False,
        )
        return jnp.mean(term)

    # Autodiff wrt sigma
    grad_sigma = jax.grad(mean_terminal)(0.2)
    assert jnp.isfinite(grad_sigma)
