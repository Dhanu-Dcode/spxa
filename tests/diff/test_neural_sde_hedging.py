"""Tests for Neural SDEs and Deep Hedging networks."""

import jax
import jax.numpy as jnp
import pytest

from spxa.diff.nn.deep_hedging import (
    deep_hedging_step,
    entropic_risk_measure,
    mean_squared_hedging_error,
    simulate_deep_hedging_pnl,
)
from spxa.diff.nn.mlp import init_mlp_params, mlp_forward
from spxa.diff.nn.neural_sde import DifferentiableNeuralSDE
from spxa.diff.prng import make_key
from spxa.diff.solvers import solve_euler_maruyama


def test_neural_sde():
    key = make_key(1313)
    k_init, k_sim = jax.random.split(key)

    nsde = DifferentiableNeuralSDE(state_dim=1, noise_dim=1)
    params = nsde.init_params(state_dim=1, hidden_dim=16, key=k_init)

    traj, _ = solve_euler_maruyama(
        sde=nsde,
        params=params,
        t_span=(0.0, 1.0),
        n_steps=10,
        n_paths=20,
        key=k_sim,
        return_trajectory=True,
    )
    assert traj.shape == (20, 11, 1)
    assert jnp.all(jnp.isfinite(traj))


def test_deep_hedging_pnl_and_optimization():
    key = make_key(1414)
    k_net, k_paths = jax.random.split(key)

    # Initialize policy network [in=3 -> hidden=16 -> out=1]
    policy_params = init_mlp_params([3, 16, 1], k_net)

    # Synthetic geometric Brownian motion price paths
    n_paths = 50
    n_steps = 10
    s0 = 100.0
    dt = 1.0 / n_steps
    z = jax.random.normal(k_paths, shape=(n_paths, n_steps))
    log_increments = (0.05 - 0.5 * 0.2**2) * dt + 0.2 * jnp.sqrt(dt) * z
    log_paths = jnp.cumsum(log_increments, axis=-1)
    log_paths_with_0 = jnp.pad(log_paths, ((0, 0), (1, 0)), mode="constant", constant_values=0.0)
    price_paths = s0 * jnp.exp(log_paths_with_0)

    # European call payoffs
    option_payoffs = jnp.maximum(price_paths[:, -1] - 100.0, 0.0)

    # Compute PnL
    pnl = simulate_deep_hedging_pnl(
        policy_params=policy_params,
        price_paths=price_paths,
        option_payoffs=option_payoffs,
        cost_rate=0.001,
    )
    assert pnl.shape == (n_paths,)
    assert jnp.all(jnp.isfinite(pnl))

    # Evaluate risk measures
    entropic_loss = entropic_risk_measure(pnl, risk_aversion=1.0)
    assert jnp.isfinite(entropic_loss)

    mse_loss = mean_squared_hedging_error(pnl)
    assert jnp.isfinite(mse_loss)

    # Differentiate loss wrt policy parameters
    def total_loss(p):
        hedging_pnl = simulate_deep_hedging_pnl(
            policy_params=p,
            price_paths=price_paths,
            option_payoffs=option_payoffs,
            cost_rate=0.001,
        )
        return mean_squared_hedging_error(hedging_pnl)

    grads = jax.grad(total_loss)(policy_params)
    assert len(grads) == len(policy_params)
    assert jnp.isfinite(grads[0][0]).all()
